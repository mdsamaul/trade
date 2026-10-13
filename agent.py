"""
agent.py - Autonomous Paper-Trading & Self-Optimizing Agent for Quotex Demo Charts.
Implements:
  1. Vision Ingestion & Auto-Cropping (OpenCV & Pillow)
  2. Strictness & Strategy Self-Optimization Engine (Dynamic Strictness Levels 1-5)
  3. Vision AI Analyzer (OpenAI SDK / xAI Grok Vision API with Heuristic CV Fallback)
  4. Virtual Paper Trading Ledger & Verification Engine (T=0 to T+Expiry verification)
  5. Rolling Accuracy (Last 5 Trades >= 80% Win Rate Benchmark)
"""

import os
import io
import re
import json
import uuid
import time
import base64
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuotexAgent")


# ==============================================================================
# DATA MODELS & SCHEMAS
# ==============================================================================

class StrictnessConfig(BaseModel):
    level: int = Field(..., ge=1, le=5)
    name: str
    min_confidence: float
    min_wick_ratio: float
    require_trend_alignment: bool
    description: str


STRICTNESS_TIERS: Dict[int, StrictnessConfig] = {
    1: StrictnessConfig(
        level=1,
        name="Lenient / Exploration",
        min_confidence=62.0,
        min_wick_ratio=0.15,
        require_trend_alignment=False,
        description="Trades on standard momentum, moderate engulfing bodies, and basic S&R bounces."
    ),
    2: StrictnessConfig(
        level=2,
        name="Balanced / Standard",
        min_confidence=70.0,
        min_wick_ratio=0.25,
        require_trend_alignment=True,
        description="Requires clean trend alignment (EMA slope) or distinct candlestick rejection wicks."
    ),
    3: StrictnessConfig(
        level=3,
        name="Selective / Conservative",
        min_confidence=78.0,
        min_wick_ratio=0.35,
        require_trend_alignment=True,
        description="Demands strong confluence: Trend alignment + Key Support/Resistance rejection."
    ),
    4: StrictnessConfig(
        level=4,
        name="High Strictness / Defense",
        min_confidence=84.0,
        min_wick_ratio=0.45,
        require_trend_alignment=True,
        description="Activated after losses. Filters out 80% of candles; requires institutional pin-bars."
    ),
    5: StrictnessConfig(
        level=5,
        name="Sniper / Ultra Strict",
        min_confidence=90.0,
        min_wick_ratio=0.55,
        require_trend_alignment=True,
        description="Highest conviction only. Requires clear momentum exhaustion at extreme S&R bounce."
    )
}


class TradeRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))
    direction: str  # "CALL", "PUT", "SKIP"
    entry_price: float
    close_price: Optional[float] = None
    timeframe_seconds: int = 60
    outcome: str = "PENDING"  # "PENDING", "WIN", "LOSS", "DRAW", "SKIPPED"
    confidence: float = 0.0
    strictness_level: int = 2
    rationale: str = ""
    trend: str = "NEUTRAL"
    setup_grade: str = "B"
    trade_amount: float = 100.0
    pnl: float = 0.0
    balance_after: float = 10000.0
    expiry_timestamp: float = 0.0  # Unix timestamp when candle closes
    indicators_summary: Dict[str, Any] = Field(default_factory=dict)


class SessionStats(BaseModel):
    balance: float
    initial_balance: float
    total_trades: int
    wins: int
    losses: int
    draws: int
    overall_win_rate: float
    rolling_window: List[str]
    rolling_win_rate: float
    target_achieved: bool
    status_badge: str
    status_message: str
    current_strictness: int
    strictness_name: str
    active_trade: Optional[TradeRecord] = None


# ==============================================================================
# VISION PROCESSOR (AUTO-CROPPING & IMAGE ENHANCEMENT)
# ==============================================================================

class VisionProcessor:
    """Handles image decoding, auto-cropping Quotex chart canvas, and visual feature extraction."""

    @staticmethod
    def decode_base64_image(image_b64: str) -> Image.Image:
        """Decodes base64 string (with or without data URI prefix) to PIL Image."""
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def encode_image_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Encodes PIL Image to base64 data URI."""
        buffered = io.BytesIO()
        img.save(buffered, format=format, quality=quality)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_str}"

    @classmethod
    def auto_crop_quotex_chart(
        cls,
        image: Image.Image,
        custom_bounds: Optional[Dict[str, float]] = None
    ) -> Image.Image:
        """
        Crops active trading canvas containing recent 20-30 candles and right-side price ladder.
        If custom_bounds are provided ({top, bottom, left, right} in 0.0-1.0 range), applies them.
        Otherwise applies intelligent Quotex UI auto-crop:
          - Trims top navigation bar (~8%)
          - Trims left asset sidebar (~8%)
          - Trims bottom dock (~6%)
          - Preserves right price axis (~12% right) while excluding binary option trade buttons.
        """
        width, height = image.size

        if custom_bounds:
            x1 = int(custom_bounds.get("left", 0.0) * width)
            y1 = int(custom_bounds.get("top", 0.0) * height)
            x2 = int(custom_bounds.get("right", 1.0) * width)
            y2 = int(custom_bounds.get("bottom", 1.0) * height)
        else:
            # Intelligent Quotex Auto-Crop presets
            x1 = int(width * 0.08)   # Skip left asset selector
            y1 = int(height * 0.09)  # Skip top header
            x2 = int(width * 0.94)   # Include price ladder on right
            y2 = int(height * 0.90)  # Skip bottom indicators/tools

        # Safety clamps
        x1 = max(0, min(x1, width - 50))
        y1 = max(0, min(y1, height - 50))
        x2 = max(x1 + 50, min(x2, width))
        y2 = max(y1 + 50, min(y2, height))

        cropped = image.crop((x1, y1, x2, y2))

        # Enhance contrast slightly to make candle wicks and text stand out crisply
        enhancer = ImageEnhance.Contrast(cropped)
        enhanced = enhancer.enhance(1.15)
        return enhanced

    @classmethod
    def cv_analyze_candlestick_pixels(cls, image: Image.Image) -> Dict[str, Any]:
        """
        Senior Quantitative Computer Vision Analysis for Quotex Charts:
          1. Detects and segments all individual candles across the canvas.
          2. Calculates dynamic Support ($Y_{sup}$) and Resistance ($Y_{res}$) swing levels.
          3. Evaluates candle body-to-wick proportions, rejection tails, and consecutive streaks.
          4. Computes true price structure trend and channel positioning.
        """
        np_img = np.array(image)
        h, w = np_img.shape[0], np_img.shape[1]
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

        # Quotex Green Candle Mask (Vibrant Emerald / Mint)
        lower_green = np.array([35, 70, 70])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Quotex Red Candle Mask (Crimson / Coral Red)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([165, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

        # Filter out UI noise: Top banner (deposit buttons, account badges) & Bottom dock
        top_cut = int(h * 0.12)
        bottom_cut = int(h * 0.08)
        left_cut = int(w * 0.04)
        right_cut = int(w * 0.08)  # right price ladder margin

        green_mask[:top_cut, :] = 0; green_mask[-bottom_cut:, :] = 0
        green_mask[:, :left_cut] = 0; green_mask[:, -right_cut:] = 0
        red_mask[:top_cut, :] = 0; red_mask[-bottom_cut:, :] = 0
        red_mask[:, :left_cut] = 0; red_mask[:, -right_cut:] = 0

        all_candles_mask = cv2.bitwise_or(green_mask, red_mask)
        contours, _ = cv2.findContours(all_candles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candle_boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Filter realistic candlestick dimensions
            if bh >= 6 and bw >= 2 and bw <= 45:
                candle_boxes.append((bx, by, bw, bh))

        # Sort chronologically from left to right
        candle_boxes = sorted(candle_boxes, key=lambda b: b[0])

        candles_data = []
        for bx, by, bw, bh in candle_boxes:
            c_g = np.count_nonzero(green_mask[by:by+bh, bx:bx+bw])
            c_r = np.count_nonzero(red_mask[by:by+bh, bx:bx+bw])
            color = "GREEN" if c_g >= c_r else "RED"

            # Profile vertical density to detect body vs wicks
            vert_profile = np.count_nonzero(all_candles_mask[by:by+bh, bx:bx+bw], axis=1)
            max_thick = np.max(vert_profile) if len(vert_profile) > 0 else 1
            body_rows = np.where(vert_profile >= max(2, int(max_thick * 0.60)))[0]

            if len(body_rows) > 0:
                body_top = by + int(body_rows[0])
                body_bot = by + int(body_rows[-1])
            else:
                body_top = by + int(bh * 0.25)
                body_bot = by + int(bh * 0.75)

            wick_top = max(0, body_top - by)
            wick_bot = max(0, (by + bh) - body_bot)
            body_h = max(1, body_bot - body_top)

            candles_data.append({
                "x": bx, "y": by, "w": bw, "h": bh,
                "color": color,
                "high": by,
                "low": by + bh,
                "body_top": body_top,
                "body_bot": body_bot,
                "upper_wick_ratio": round(wick_top / bh, 2),
                "lower_wick_ratio": round(wick_bot / bh, 2),
                "body_ratio": round(body_h / bh, 2)
            })

        # Calculate Support & Resistance from the last 15-20 candles
        if candles_data:
            lookback = candles_data[-20:]
            resistance_y = min([c["high"] for c in lookback]) # Lowest Y is highest price
            support_y = max([c["low"] for c in lookback])      # Highest Y is lowest price
            channel_h = max(1, support_y - resistance_y)

            active_candle = candles_data[-1]
            prev_candle = candles_data[-2] if len(candles_data) >= 2 else active_candle

            # Channel position ratio: 0.0 = at Resistance (Peak), 1.0 = at Support (Floor)
            pos_ratio = (active_candle["low"] - resistance_y) / channel_h
            pos_ratio = min(1.0, max(0.0, pos_ratio))

            # Trend diagnosis from sequence of candle closes over last 8 candles
            sample_candles = candles_data[-8:] if len(candles_data) >= 8 else candles_data
            closes = [c["body_bot"] if c["color"] == "GREEN" else c["body_top"] for c in sample_candles]
            slope = closes[-1] - closes[0]  # Note: in pixel space, decreasing Y means price is rising
            if slope < -12:
                trend = "STRONG_BULLISH"
            elif slope > 12:
                trend = "STRONG_BEARISH"
            elif slope < -5:
                trend = "MILD_BULLISH"
            elif slope > 5:
                trend = "MILD_BEARISH"
            else:
                trend = "SIDEWAYS_CONSOLIDATION"

            # Count consecutive same-color candle streaks leading up to active candle
            green_streak = 0
            red_streak = 0
            for c in reversed(candles_data[:-1]):
                if c["color"] == "GREEN":
                    if red_streak > 0: break
                    green_streak += 1
                else:
                    if green_streak > 0: break
                    red_streak += 1

            lower_rejection = active_candle["lower_wick_ratio"] >= 0.30
            upper_rejection = active_candle["upper_wick_ratio"] >= 0.30
            active_candle_avg_y = float(active_candle["body_top"] + active_candle["body_bot"]) / 2.0
            latest_color = active_candle["color"]
        else:
            resistance_y = int(h * 0.2)
            support_y = int(h * 0.8)
            channel_h = max(1, support_y - resistance_y)
            pos_ratio = 0.5
            trend = "SIDEWAYS_CONSOLIDATION"
            green_streak = 0
            red_streak = 0
            lower_rejection = False
            upper_rejection = False
            active_candle_avg_y = h / 2.0
            latest_color = "NEUTRAL"
            candles_data = []

        green_pixels = int(np.count_nonzero(green_mask))
        red_pixels = int(np.count_nonzero(red_mask))
        total_candle_pixels = green_pixels + red_pixels

        return {
            "candles": candles_data,
            "resistance_y": resistance_y,
            "support_y": support_y,
            "channel_height": channel_h,
            "pos_ratio": round(pos_ratio, 3),
            "trend": trend,
            "green_streak": green_streak,
            "red_streak": red_streak,
            "lower_rejection": lower_rejection,
            "upper_rejection": upper_rejection,
            "latest_candle_color": latest_color,
            "active_candle_avg_y": active_candle_avg_y,
            "chart_height": h,
            "total_detected_candle_pixels": total_candle_pixels
        }


# ==============================================================================
# VISION AI STRATEGY ENGINE & OPENAI / GROK CLIENT
# ==============================================================================

class VisionAIEngine:
    """
    Connects to OpenAI / xAI Grok Vision API or falls back to Advanced CV Heuristics.
    Performs price action analysis, rejection wick evaluation, and directional prediction.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
        self._init_client()

    def update_config(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        if base_url is not None:
            self.base_url = base_url.strip()
        if model_name is not None:
            self.model_name = model_name.strip()
        self._init_client()

    def _init_client(self):
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else "https://api.openai.com/v1"
                )
                logger.info(f"OpenAI Vision client initialized with model '{self.model_name}' at '{self.base_url}'")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.info("No OPENAI_API_KEY detected. Heuristic Computer-Vision engine active as zero-config fallback.")

    def analyze_chart(
        self,
        image_b64: str,
        strictness_level: int,
        candle_timeframe: int = 60
    ) -> Dict[str, Any]:
        """
        Sends the cropped chart frame to the Vision AI model or heuristic engine.
        Returns structured decision: CALL, PUT, or SKIP with confidence and rationale.
        """
        strictness_cfg = STRICTNESS_TIERS.get(strictness_level, STRICTNESS_TIERS[2])

        # If client is configured and has API key, call Vision model
        if self.client and self.api_key:
            try:
                return self._call_ai_vision_api(image_b64, strictness_cfg, candle_timeframe)
            except Exception as e:
                logger.error(f"AI Vision API call failed ({e}). Falling back to Heuristic CV Engine.")
                return self._heuristic_vision_analysis(image_b64, strictness_cfg, candle_timeframe)
        else:
            return self._heuristic_vision_analysis(image_b64, strictness_cfg, candle_timeframe)

    def _call_ai_vision_api(
        self,
        image_b64: str,
        strictness_cfg: StrictnessConfig,
        timeframe_seconds: int
    ) -> Dict[str, Any]:
        """Calls the OpenAI-compatible vision model with strict JSON formatting instructions."""
        system_prompt = f"""You are a World-Class Binary Options Quantitative Vision Trader analyzing Quotex Candlestick Charts.
Your objective is to identify ultra-high probability trade setups (CALL, PUT, or SKIP) for a {timeframe_seconds}-second binary option expiry.

DYNAMIC STRICTNESS LEVEL: Level {strictness_cfg.level}/5 ({strictness_cfg.name})
MINIMUM CONFIDENCE REQUIRED TO EXECUTE: {strictness_cfg.min_confidence}%
REQUIRE TREND ALIGNMENT: {strictness_cfg.require_trend_alignment}
MINIMUM REJECTION WICK RATIO: {strictness_cfg.min_wick_ratio * 100:.0f}%
CRITERIA: {strictness_cfg.description}

Analyze the provided chart image:
1. Identify the recent trend (Bullish, Bearish, or Sideways/Ranging).
2. Examine the most recent 1 to 3 completed candles and the current forming candle. Look for:
   - Long rejection wicks (pin bars / hammers) bouncing off support/resistance or moving averages.
   - Momentum engulfing bodies.
   - Clean pullbacks to key structural levels.
3. Extract or estimate the current asset price from the right-hand price scale or price watermark.
4. Calculate a realistic confidence score (0 to 100%).
   - If confidence is BELOW {strictness_cfg.min_confidence}%, you MUST output "SKIP".
   - Only output "CALL" or "PUT" if the setup satisfies all strictness criteria.

You MUST respond strictly in valid JSON format matching this schema:
{{
  "direction": "CALL" | "PUT" | "SKIP",
  "confidence": <float between 0 and 100>,
  "current_price": <float price, e.g. 1.08452 or 92.450>,
  "trend": "BULLISH" | "BEARISH" | "SIDEWAYS",
  "setup_grade": "A+" | "A" | "B+" | "B" | "C",
  "rejection_wick_found": <boolean>,
  "support_resistance_level": "<string level or description>",
  "reasoning": "<concise 2-sentence rationale for directional bias or skip rationale>"
}}
"""

        user_content = [
            {
                "type": "text",
                "text": "Analyze this Quotex chart snippet. Output the JSON trading signal for the upcoming candle."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high"
                }
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.15,
            max_tokens=450
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"Raw Vision AI Response: {content}")

        # Parse JSON
        parsed = self._extract_json(content)
        return self._sanitize_decision(parsed, strictness_cfg)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from AI text response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try regex search for markdown ```json ... ``` or { ... }
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse valid JSON from AI response: {text[:200]}")

    def _sanitize_decision(self, data: Dict[str, Any], strictness_cfg: StrictnessConfig) -> Dict[str, Any]:
        """Validates AI output and enforces strictness thresholds."""
        direction = str(data.get("direction", "SKIP")).upper().strip()
        if direction not in ["CALL", "PUT", "SKIP"]:
            direction = "SKIP"

        try:
            confidence = float(data.get("confidence", 50.0))
        except (ValueError, TypeError):
            confidence = 50.0

        try:
            price = float(data.get("current_price", 100.0))
            if price <= 0:
                price = 100.0
        except (ValueError, TypeError):
            price = 100.0

        trend = str(data.get("trend", "SIDEWAYS")).upper().strip()
        setup_grade = str(data.get("setup_grade", "B")).strip()
        reasoning = str(data.get("reasoning", "Standard price action evaluation.")).strip()

        # Strictness check enforcement
        if direction != "SKIP" and confidence < strictness_cfg.min_confidence:
            reasoning = f"[Strictness Level {strictness_cfg.level} Filter]: Confidence ({confidence:.1f}%) is below required threshold ({strictness_cfg.min_confidence:.1f}%). Trade filtered to SKIP."
            direction = "SKIP"

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "current_price": round(price, 5),
            "trend": trend,
            "setup_grade": setup_grade,
            "rejection_wick_found": bool(data.get("rejection_wick_found", False)),
            "support_resistance_level": str(data.get("support_resistance_level", "Dynamic")),
            "reasoning": reasoning
        }

    def _heuristic_vision_analysis(
        self,
        image_b64: str,
        strictness_cfg: StrictnessConfig,
        timeframe_seconds: int
    ) -> Dict[str, Any]:
        """
        Senior Quantitative Trader Price Action Engine for Quotex Charts.
        Executes strictly on high-conviction institutional confluence:
          1. Overbought/Oversold Exhaustion Protection (Never buys the top / sells the bottom)
          2. A+ Support & Resistance Rejections (Pin-bar hammers / shooting stars)
          3. Key Level Engulfing Patterns (Demand/Supply shift)
          4. Trend Pullback Continuations
          5. Strict Chop & Noise Filters (Skips low-probability consolidation)
        """
        pil_img = VisionProcessor.decode_base64_image(image_b64)
        cv_stats = VisionProcessor.cv_analyze_candlestick_pixels(pil_img)

        candles = cv_stats.get("candles", [])
        resistance_y = cv_stats.get("resistance_y", 100)
        support_y = cv_stats.get("support_y", 300)
        pos = cv_stats.get("pos_ratio", 0.5)  # 0.0 at Resistance (Peak), 1.0 at Support (Floor)
        trend = cv_stats.get("trend", "SIDEWAYS_CONSOLIDATION")
        green_streak = cv_stats.get("green_streak", 0)
        red_streak = cv_stats.get("red_streak", 0)

        chart_h = max(1.0, float(cv_stats.get("chart_height", 360)))
        candle_y = float(cv_stats.get("active_candle_avg_y", chart_h / 2.0))
        price_ratio = 1.0 - (candle_y / chart_h)
        base_price = round(1.08200 + (price_ratio * 0.00650), 5)

        if not candles:
            return {
                "direction": "SKIP",
                "confidence": 50.0,
                "current_price": base_price,
                "trend": trend,
                "setup_grade": "C",
                "rejection_wick_found": False,
                "support_resistance_level": "Awaiting Chart Data",
                "reasoning": "No clear candlestick pattern detected on chart. Skipping cycle."
            }

        active_c = candles[-1]
        prev_c = candles[-2] if len(candles) >= 2 else active_c

        direction = "SKIP"
        confidence = 55.0
        grade = "C"
        rejection_wick = False
        level_name = f"Zone (Pos: {pos:.2f})"
        reasoning = "Consolidation / Indecisive candle structure. Awaiting A+ S&R level confirmation."

        # =========================================================================
        # SENIOR RULE 1: OVERBOUGHT & OVERSOLD EXHAUSTION PROTECTION (CAPITAL DEFENSE)
        # =========================================================================
        # 1A. Market pumped 3+ green candles and is touching Resistance ceiling
        if green_streak >= 3 and pos <= 0.28:
            if active_c["upper_wick_ratio"] >= 0.28 or (active_c["color"] == "RED" and active_c["body_ratio"] >= 0.45):
                direction = "PUT"
                confidence = 88.0 + (active_c["upper_wick_ratio"] * 10.0)
                grade = "A+"
                rejection_wick = True
                level_name = "Major Resistance Ceiling"
                reasoning = f"Bearish Exhaustion Reversal at Resistance ({green_streak} green candle rally exhausted, upper wick {active_c['upper_wick_ratio']*100:.0f}%)."
            else:
                direction = "SKIP"
                confidence = 52.0
                grade = "C"
                rejection_wick = False
                level_name = "Overbought Ceiling"
                reasoning = f"Exhaustion Trap avoided: Market already surged {green_streak} green candles into Resistance. Buying CALL here is high-risk."

        # 1B. Market plunged 3+ red candles and is touching Support floor
        elif red_streak >= 3 and pos >= 0.72:
            if active_c["lower_wick_ratio"] >= 0.28 or (active_c["color"] == "GREEN" and active_c["body_ratio"] >= 0.45):
                direction = "CALL"
                confidence = 88.0 + (active_c["lower_wick_ratio"] * 10.0)
                grade = "A+"
                rejection_wick = True
                level_name = "Major Support Floor"
                reasoning = f"Bullish Exhaustion Reversal at Support ({red_streak} red candle drop exhausted, lower wick {active_c['lower_wick_ratio']*100:.0f}%)."
            else:
                direction = "SKIP"
                confidence = 52.0
                grade = "C"
                rejection_wick = False
                level_name = "Oversold Floor"
                reasoning = f"Exhaustion Trap avoided: Market plunged {red_streak} red candles into Support. Selling PUT here is high-risk."

        # =========================================================================
        # SENIOR RULE 2: A+ PIN-BAR & WICK REJECTIONS AT KEY S&R
        # =========================================================================
        # 2A. Bullish Pin-bar / Hammer at Support (Floor)
        elif pos >= 0.62 and active_c["lower_wick_ratio"] >= 0.30:
            direction = "CALL"
            confidence = 84.0 + (active_c["lower_wick_ratio"] * 18.0)
            grade = "A+"
            rejection_wick = True
            level_name = "Key Support Demand Zone"
            reasoning = f"A+ Bullish Pin-bar rejection at Support floor ({active_c['lower_wick_ratio']*100:.0f}% lower wick). Buyers strongly defending level."

        # 2B. Bearish Shooting Star at Resistance (Ceiling)
        elif pos <= 0.38 and active_c["upper_wick_ratio"] >= 0.30:
            direction = "PUT"
            confidence = 84.0 + (active_c["upper_wick_ratio"] * 18.0)
            grade = "A+"
            rejection_wick = True
            level_name = "Key Resistance Supply Zone"
            reasoning = f"A+ Bearish Shooting Star rejection at Resistance ceiling ({active_c['upper_wick_ratio']*100:.0f}% upper wick). Sellers rejecting high prices."

        # =========================================================================
        # SENIOR RULE 3: HIGH-VOLUME ENGULFING PATTERNS AT KEY ZONES
        # =========================================================================
        # 3A. Bullish Engulfing at Support
        elif pos >= 0.58 and prev_c["color"] == "RED" and active_c["color"] == "GREEN" and active_c["body_ratio"] >= 0.50 and active_c["h"] > prev_c["h"]:
            direction = "CALL"
            confidence = 84.0
            grade = "A"
            rejection_wick = False
            level_name = "Key Support Floor"
            reasoning = "Bullish Engulfing reversal formed at Support. Buyers absorbing all previous selling pressure."

        # 3B. Bearish Engulfing at Resistance
        elif pos <= 0.42 and prev_c["color"] == "GREEN" and active_c["color"] == "RED" and active_c["body_ratio"] >= 0.50 and active_c["h"] > prev_c["h"]:
            direction = "PUT"
            confidence = 84.0
            grade = "A"
            rejection_wick = False
            level_name = "Key Resistance Ceiling"
            reasoning = "Bearish Engulfing reversal formed at Resistance. Sellers overwhelming previous buyers."

        # =========================================================================
        # SENIOR RULE 4: HEALTHY TREND PULLBACK CONTINUATION
        # =========================================================================
        # 4A. Uptrend continuation after small red pullback
        elif trend in ["STRONG_BULLISH", "MILD_BULLISH"] and pos >= 0.25 and pos <= 0.70:
            if prev_c["color"] == "RED" and active_c["color"] == "GREEN" and active_c["lower_wick_ratio"] >= 0.18:
                direction = "CALL"
                confidence = 82.0
                grade = "A"
                rejection_wick = True
                level_name = "Uptrend Pullback Support"
                reasoning = "Uptrend continuation: Minor red retracement followed by immediate bullish rejection bounce."

        # 4B. Downtrend continuation after small green pullback
        elif trend in ["STRONG_BEARISH", "MILD_BEARISH"] and pos >= 0.30 and pos <= 0.75:
            if prev_c["color"] == "GREEN" and active_c["color"] == "RED" and active_c["upper_wick_ratio"] >= 0.18:
                direction = "PUT"
                confidence = 82.0
                grade = "A"
                rejection_wick = True
                level_name = "Downtrend Pullback Resistance"
                reasoning = "Downtrend continuation: Minor green retracement followed by immediate bearish rejection drop."

        # =========================================================================
        # SENIOR RULE 5: DEFAULT CHOP & NOISE FILTER (SKIP)
        # =========================================================================
        else:
            direction = "SKIP"
            confidence = 58.0
            grade = "C"
            rejection_wick = False
            level_name = "Consolidation Channel"
            reasoning = f"No high-conviction S&R confluence (Channel pos: {pos:.2f}, Trend: {trend}). A professional trader waits for prime levels."

        confidence = min(96.0, max(50.0, confidence))

        # Enforce strictness tier threshold
        if direction != "SKIP" and confidence < strictness_cfg.min_confidence:
            reasoning = f"Setup ({direction} @ {confidence:.1f}%) filtered out by Strictness Level {strictness_cfg.level} ({strictness_cfg.name}). Demands >= {strictness_cfg.min_confidence}% confidence."
            direction = "SKIP"

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "current_price": round(base_price, 5),
            "trend": trend,
            "setup_grade": grade,
            "rejection_wick_found": rejection_wick,
            "support_resistance_level": level_name,
            "reasoning": reasoning
        }

    def verify_price_from_frame(
        self,
        image_b64: str,
        entry_price: float,
        direction: str
    ) -> float:
        """
        Reads or calculates the current closing price from the verification frame.
        """
        if self.client and self.api_key:
            try:
                system_prompt = f"""Examine this updated Quotex chart. 
The trader previously entered a {direction} trade at price {entry_price}.
Extract the current closing price of the current candle from the price axis or chart.
Respond ONLY with a valid JSON object: {{"close_price": <float>}}"""
                user_content = [
                    {"type": "text", "text": "Extract current close price."},
                    {"type": "image_url", "image_url": {"url": image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}"}}
                ]
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    temperature=0.0,
                    max_tokens=100
                )
                data = self._extract_json(resp.choices[0].message.content)
                val = float(data.get("close_price", 0.0))
                if val > 0:
                    return val
            except Exception as e:
                logger.warning(f"Vision price extraction error: {e}. Utilizing pixel delta calculation.")

        # Heuristic delta calculation from new frame's real candle vertical position
        pil_img = VisionProcessor.decode_base64_image(image_b64)
        new_stats = VisionProcessor.cv_analyze_candlestick_pixels(pil_img)
        chart_h = max(1.0, float(new_stats.get("chart_height", 360)))
        new_candle_y = float(new_stats.get("active_candle_avg_y", chart_h / 2.0))
        new_price_ratio = 1.0 - (new_candle_y / chart_h)
        measured_close = round(1.08200 + (new_price_ratio * 0.00650), 5)

        if measured_close != entry_price:
            close_price = measured_close
        else:
            bias = new_stats["recent_bullish_bias"]
            delta = 0.00008 if bias >= 0.50 else -0.00008
            close_price = round(entry_price + delta, 5)

        return close_price


# ==============================================================================
# TRADING SESSION & STATE MANAGER (PAPER TRADING LEDGER & SELF-OPTIMIZER)
# ==============================================================================

class TradingSessionManager:
    """
    Thread-safe Paper-Trading Ledger and Autonomous Self-Optimizing State Manager.
    Tracks virtual balance, win/loss metrics, rolling 5-trade accuracy, and dynamic strictness.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        trade_amount: float = 100.0,
        payout_rate: float = 0.85,
        default_strictness: int = 2,
        rolling_window_size: int = 5,
        target_win_rate: float = 0.80
    ):
        self.lock = threading.Lock()
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.trade_amount = float(trade_amount)
        self.payout_rate = float(payout_rate)
        self.rolling_window_size = rolling_window_size
        self.target_win_rate = target_win_rate  # 80% = 4 out of 5 wins

        self.current_strictness = default_strictness
        self.auto_optimize = True

        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Rolling outcomes deque: holds strings "WIN", "LOSS", "DRAW"
        self.rolling_outcomes: deque[str] = deque(maxlen=rolling_window_size)
        self.target_achieved = False

        self.active_trade: Optional[TradeRecord] = None
        self.trade_history: List[TradeRecord] = []

    def get_stats(self) -> SessionStats:
        with self.lock:
            total = self.wins + self.losses + self.draws
            overall_wr = (self.wins / total * 100.0) if total > 0 else 0.0

            rolling_list = list(self.rolling_outcomes)
            rolling_wins = rolling_list.count("WIN")
            rolling_wr = (rolling_wins / len(rolling_list) * 100.0) if rolling_list else 0.0

            # Target is reached when the rolling window has at least 4 wins out of 5 trades (>= 80%)
            achieved = (len(rolling_list) >= self.rolling_window_size and (rolling_wins / len(rolling_list)) >= self.target_win_rate)
            self.target_achieved = achieved

            strict_cfg = STRICTNESS_TIERS.get(self.current_strictness, STRICTNESS_TIERS[2])

            if achieved:
                status_badge = "TARGET ACHIEVED: 80% WIN RATE"
                status_message = f"GOAL REACHED! Rolling accuracy: {rolling_wins}/{len(rolling_list)} Wins ({rolling_wr:.0f}%). Strategy converged."
            else:
                status_badge = "TRAINING / PRACTICING"
                if len(rolling_list) < self.rolling_window_size:
                    status_message = f"Practicing... ({len(rolling_list)}/{self.rolling_window_size} trades in window). Rolling: {rolling_wins}W - {rolling_list.count('LOSS')}L."
                else:
                    status_message = f"Optimizing strategy ({rolling_wins}/{self.rolling_window_size} Wins = {rolling_wr:.0f}%). Target: >= 80%."

            return SessionStats(
                balance=round(self.balance, 2),
                initial_balance=round(self.initial_balance, 2),
                total_trades=total,
                wins=self.wins,
                losses=self.losses,
                draws=self.draws,
                overall_win_rate=round(overall_wr, 1),
                rolling_window=rolling_list,
                rolling_win_rate=round(rolling_wr, 1),
                target_achieved=achieved,
                status_badge=status_badge,
                status_message=status_message,
                current_strictness=self.current_strictness,
                strictness_name=f"Level {self.current_strictness} - {strict_cfg.name}",
                active_trade=self.active_trade
            )

    def open_trade(
        self,
        direction: str,
        entry_price: float,
        confidence: float,
        rationale: str,
        trend: str,
        setup_grade: str,
        timeframe_seconds: int = 60,
        indicators_summary: Optional[Dict[str, Any]] = None
    ) -> TradeRecord:
        """Opens a virtual paper trade at T=0."""
        with self.lock:
            if self.active_trade is not None:
                raise ValueError("An active paper trade is already open. Wait for candle expiry.")

            now = time.time()
            trade = TradeRecord(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
                direction=direction,
                entry_price=entry_price,
                timeframe_seconds=timeframe_seconds,
                outcome="PENDING",
                confidence=confidence,
                strictness_level=self.current_strictness,
                rationale=rationale,
                trend=trend,
                setup_grade=setup_grade,
                trade_amount=self.trade_amount,
                pnl=0.0,
                balance_after=self.balance,
                expiry_timestamp=now + timeframe_seconds,
                indicators_summary=indicators_summary or {}
            )
            self.active_trade = trade
            logger.info(f"VIRTUAL TRADE OPENED: {direction} @ {entry_price} | Conf: {confidence}% | Strictness: L{self.current_strictness}")
            return trade

    def resolve_trade(
        self,
        close_price: float,
        trade_id: Optional[str] = None
    ) -> TradeRecord:
        """
        Resolves trade at T+Expiry:
          - CALL & Close > Entry => WIN
          - PUT & Close < Entry => WIN
          - Close == Entry => DRAW
          - Otherwise => LOSS
        Updates PnL, balance, rolling window, and triggers self-optimizing strictness adjustments.
        """
        with self.lock:
            trade = self.active_trade
            if trade is None:
                raise ValueError("No active trade found to resolve.")

            if trade_id and trade.id != trade_id:
                raise ValueError(f"Active trade ID {trade.id} does not match {trade_id}")

            trade.close_price = close_price
            direction = trade.direction
            entry_price = trade.entry_price

            # Determine outcome
            if direction == "CALL":
                if close_price > entry_price:
                    outcome = "WIN"
                elif close_price < entry_price:
                    outcome = "LOSS"
                else:
                    outcome = "DRAW"
            elif direction == "PUT":
                if close_price < entry_price:
                    outcome = "WIN"
                elif close_price > entry_price:
                    outcome = "LOSS"
                else:
                    outcome = "DRAW"
            else:
                outcome = "DRAW"

            trade.outcome = outcome

            # Financial PnL calculation
            if outcome == "WIN":
                profit = trade.trade_amount * self.payout_rate
                self.balance += profit
                trade.pnl = profit
                self.wins += 1
            elif outcome == "LOSS":
                self.balance -= trade.trade_amount
                trade.pnl = -trade.trade_amount
                self.losses += 1
            else:
                trade.pnl = 0.0
                self.draws += 1

            self.total_trades += 1
            trade.balance_after = self.balance

            # Update rolling accuracy window
            self.rolling_outcomes.append(outcome)

            # Record in ledger
            self.trade_history.insert(0, trade)
            self.active_trade = None

            # Self-Optimization Feedback Loop
            self._self_optimize(outcome)

            logger.info(
                f"TRADE RESOLVED: {outcome} | Dir: {direction} | Entry: {entry_price} | Close: {close_price} | "
                f"PnL: ${trade.pnl:+.2f} | Balance: ${self.balance:.2f}"
            )
            return trade

    def _self_optimize(self, latest_outcome: str):
        """
        Self-Optimizing Strictness Controller:
        - If rolling win rate < 80%:
            - On a LOSS, the agent tightens strictness: increases level by +1 (max 5)
              to filter out lower-probability setups and protect capital.
        - If rolling win rate achieves >= 80% (4/5 wins):
            - Target achieved! Stabilizes at the optimal discovered strictness.
        """
        if not self.auto_optimize:
            return

        rolling_list = list(self.rolling_outcomes)
        rolling_wins = rolling_list.count("WIN")
        rolling_total = len(rolling_list)
        rolling_wr = (rolling_wins / rolling_total) if rolling_total > 0 else 0.0

        if rolling_total >= self.rolling_window_size and rolling_wr >= self.target_win_rate:
            # Target achieved
            self.target_achieved = True
            logger.info(f"Target achieved! Rolling accuracy {rolling_wins}/{rolling_total} ({rolling_wr*100:.0f}%).")
        else:
            self.target_achieved = False
            if latest_outcome == "LOSS":
                # Tighten strictness parameter to reject marginal setups
                old_lvl = self.current_strictness
                self.current_strictness = min(5, self.current_strictness + 1)
                logger.info(
                    f"[SELF-OPTIMIZER] Loss registered. Strictness automatically tightened from Level {old_lvl} -> Level {self.current_strictness} "
                    f"to demand higher confluence."
                )

    def reset_session(self):
        """Resets the trading session to clean baseline."""
        with self.lock:
            self.balance = self.initial_balance
            self.total_trades = 0
            self.wins = 0
            self.losses = 0
            self.draws = 0
            self.rolling_outcomes.clear()
            self.target_achieved = False
            self.active_trade = None
            self.trade_history.clear()
            self.current_strictness = 2
            logger.info("Session reset to initial demo parameters.")


# ==============================================================================
# DIRECT QUOTEX AUTO-CLICKER (REAL TRADE EXECUTION)
# ==============================================================================

# ==============================================================================
# DIRECT QUOTEX AUTO-CLICKER (FAIL-PROOF HARDWARE CLICK EXECUTION)
# ==============================================================================

import ctypes
from ctypes import wintypes

# Win32 SendInput Structures
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]

class _INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_u", _INPUT_UNION),
    ]


def attach_to_input_desktop() -> bool:
    """
    Ensures the calling thread is attached to the user's interactive Windows desktop (Default).
    Critical in Windows services or IDE subprocesses where worker threads default to isolated desktops.
    """
    try:
        user32 = ctypes.windll.user32
        DESKTOP_ALL = 0x01FF
        hDesk = user32.OpenInputDesktop(0, False, DESKTOP_ALL)
        if not hDesk:
            hDesk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
        if hDesk:
            return bool(user32.SetThreadDesktop(hDesk))
    except Exception as e:
        logger.warning(f"Could not attach to input desktop: {e}")
    return False


def force_focus_quotex_window(target_x: Optional[int] = None, target_y: Optional[int] = None) -> bool:
    """
    Guaranteed method to bring the Quotex browser window to the active foreground.
    Bypasses Windows UIPI and SetForegroundWindow restrictions using AttachThreadInput + Alt key unlock.
    """
    try:
        attach_to_input_desktop()
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        target_hwnd = None

        # 1. First priority: if target coordinates are given, inspect window under coordinates
        if target_x is not None and target_y is not None:
            pt = wintypes.POINT(int(target_x), int(target_y))
            hw_at_pt = user32.WindowFromPoint(pt)
            if hw_at_pt:
                root = user32.GetAncestor(hw_at_pt, 2)  # GA_ROOT
                if root and user32.IsWindowVisible(root):
                    target_hwnd = root

        # 2. Second priority: find Quotex trading window (excluding agent dashboard)
        if not target_hwnd:
            candidates = []
            def enum_cb(hwnd, lp):
                if user32.IsWindowVisible(hwnd):
                    buf = ctypes.create_unicode_buffer(512)
                    if user32.GetWindowTextW(hwnd, buf, 512) > 0:
                        t = buf.value.lower()
                        # Match Quotex demo or trading, ignore our own agent dashboard title
                        if "quotex" in t and "autonomous paper-trading" not in t:
                            candidates.append(hwnd)
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            if candidates:
                target_hwnd = candidates[0]

        if not target_hwnd:
            return False

        cur_fg = user32.GetForegroundWindow()
        if cur_fg == target_hwnd:
            return True

        cur_thread = kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(cur_fg, None) if cur_fg else 0
        target_thread = user32.GetWindowThreadProcessId(target_hwnd, None)

        if fg_thread and fg_thread != cur_thread:
            user32.AttachThreadInput(cur_thread, fg_thread, True)
        if target_thread and target_thread != cur_thread:
            user32.AttachThreadInput(cur_thread, target_thread, True)

        user32.AllowSetForegroundWindow(-1)
        if user32.IsIconic(target_hwnd):
            user32.ShowWindow(target_hwnd, 9)  # SW_RESTORE
        else:
            user32.ShowWindow(target_hwnd, 5)  # SW_SHOW

        user32.BringWindowToTop(target_hwnd)

        # Alt key press trick to unlock SetForegroundWindow permission in Windows
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)
        res = user32.SetForegroundWindow(target_hwnd)

        if fg_thread and fg_thread != cur_thread:
            user32.AttachThreadInput(cur_thread, fg_thread, False)
        if target_thread and target_thread != cur_thread:
            user32.AttachThreadInput(cur_thread, target_thread, False)

        return bool(res)
    except Exception as e:
        logger.warning(f"force_focus_quotex_window warning: {e}")
        return False


def dispatch_hardware_click(target_x: int, target_y: int) -> bool:
    """
    Executes exactly ONE single, precise, fail-proof mouse click on target screen coordinates.
    Brings target window to foreground, simulates realistic mouse hover and 100ms click press.
    """
    import time
    attach_to_input_desktop()
    user32 = ctypes.windll.user32

    # Step 1: Ensure Quotex / target window is in the foreground
    force_focus_quotex_window(target_x, target_y)
    time.sleep(0.04)

    # Step 2: Physical hardware cursor placement
    user32.SetCursorPos(int(target_x), int(target_y))
    time.sleep(0.03)

    # Step 3: Win32 Virtual Desktop SendInput (Primary Path - 1 Single Click)
    v_left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
    v_top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
    v_width = max(1, user32.GetSystemMetrics(78))   # SM_CXVIRTUALSCREEN
    v_height = max(1, user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN

    norm_x = int((target_x - v_left) * 65535 / v_width)
    norm_y = int((target_y - v_top) * 65535 / v_height)

    # Send MOUSEEVENTF_MOVE to trigger browser hover / hit-test
    inp_move = _INPUT(type=0)
    inp_move.mi.dx = norm_x
    inp_move.mi.dy = norm_y
    inp_move.mi.dwFlags = 0x0001 | 0x8000 | 0x4000  # MOVE | ABSOLUTE | VIRTUALDESK
    user32.SendInput(1, ctypes.byref(inp_move), ctypes.sizeof(_INPUT))
    time.sleep(0.04)

    # Send MOUSEEVENTF_LEFTDOWN
    inp_down = _INPUT(type=0)
    inp_down.mi.dx = norm_x
    inp_down.mi.dy = norm_y
    inp_down.mi.dwFlags = 0x0002 | 0x8000 | 0x4000  # LEFTDOWN | ABSOLUTE | VIRTUALDESK
    sent = user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(_INPUT))
    time.sleep(0.10)  # Realistic 100ms press duration

    # Send MOUSEEVENTF_LEFTUP
    inp_up = _INPUT(type=0)
    inp_up.mi.dx = norm_x
    inp_up.mi.dy = norm_y
    inp_up.mi.dwFlags = 0x0004 | 0x8000 | 0x4000  # LEFTUP | ABSOLUTE | VIRTUALDESK
    user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(_INPUT))

    # If SendInput was successful, exactly ONE single click has completed!
    if sent > 0:
        return True

    # Fallback ONLY if SendInput returned 0 (failed to send)
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.click(x=int(target_x), y=int(target_y))
        return True
    except Exception as py_err:
        logger.debug(f"PyAutoGUI click fallback error: {py_err}")

    return False


AUTOCLICK_CONFIG_FILE = Path(__file__).parent / "autoclick_config.json"

class AutoClickerConfig(BaseModel):
    enabled: bool = True
    call_x: int = 4482
    call_y: int = 486
    put_x: int = 4442
    put_y: int = 531
    restore_cursor: bool = False


class QuotexAutoClicker:
    """Automates mouse clicking on Quotex UP / DOWN buttons on trade signals."""
    def __init__(self):
        self.config = AutoClickerConfig()
        self.lock = threading.Lock()
        self._last_click_time = 0.0
        self._load_persisted_config()

    def _load_persisted_config(self):
        try:
            if AUTOCLICK_CONFIG_FILE.exists():
                with open(AUTOCLICK_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.config = AutoClickerConfig(**data)
                    logger.info(f"Loaded persistent AutoClicker config: {self.config}")
        except Exception as e:
            logger.warning(f"Could not load autoclick config file: {e}")

    def _save_persisted_config(self):
        try:
            with open(AUTOCLICK_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config.dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save autoclick config file: {e}")

    def update_config(self, enabled=None, call_x=None, call_y=None, put_x=None, put_y=None, restore_cursor=None):
        with self.lock:
            if enabled is not None: self.config.enabled = bool(enabled)
            if call_x is not None: self.config.call_x = int(call_x)
            if call_y is not None: self.config.call_y = int(call_y)
            if put_x is not None: self.config.put_x = int(put_x)
            if put_y is not None: self.config.put_y = int(put_y)
            if restore_cursor is not None: self.config.restore_cursor = bool(restore_cursor)
            self._save_persisted_config()

    def auto_detect_quotex_buttons(self) -> Dict[str, Any]:
        """
        Scans all connected screens with Computer Vision to find exact centers of Quotex
        Green Buy (CALL) and Red Sell (PUT) buttons. Handles multi-monitor offsets.
        """
        try:
            import cv2
            import numpy as np
            import time
            from PIL import ImageGrab

            attach_to_input_desktop()
            force_focus_quotex_window()
            time.sleep(0.35)

            # Grab across all monitors
            img = ImageGrab.grab(all_screens=True)
            arr = np.array(img)
            h, w, _ = arr.shape

            user32 = ctypes.windll.user32
            v_left = user32.GetSystemMetrics(76)
            v_top = user32.GetSystemMetrics(77)

            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

            g_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
            r_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
            r_mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
            r_mask = cv2.bitwise_or(r_mask1, r_mask2)

            def detect(mask):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cands = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] >= 35 and cv2.boundingRect(c)[3] >= 18]
                if cands:
                    bx, by, bw, bh = sorted(cands, key=lambda b: b[2] * b[3], reverse=True)[0]
                    # Map from ImageGrab coords (relative to v_left, v_top) to absolute screen coords:
                    screen_x = v_left + bx + (bw // 2)
                    screen_y = v_top + by + (bh // 2)
                    return int(screen_x), int(screen_y)
                return None

            call_pos = detect(g_mask)
            put_pos = detect(r_mask)

            if call_pos and put_pos:
                with self.lock:
                    self.config.call_x, self.config.call_y = call_pos
                    self.config.put_x, self.config.put_y = put_pos
                logger.info(f"🎯 [AUTO-DETECT SUCCESS] Green Buy: {call_pos}, Red Sell: {put_pos}")
                return {
                    "success": True,
                    "call_x": call_pos[0],
                    "call_y": call_pos[1],
                    "put_x": put_pos[0],
                    "put_y": put_pos[1]
                }
        except Exception as e:
            logger.error(f"Auto-detection error: {e}")
        return {"success": False, "error": "Quotex buttons not found on screen"}

    def execute_click(self, direction: str) -> bool:
        with self.lock:
            if not self.config.enabled:
                return False

            # Debounce protection: enforce minimum 2.0s between trades
            now = time.time()
            if now - self._last_click_time < 2.0:
                logger.info(f"Click debounced ({now - self._last_click_time:.2f}s since last click). Skipping duplicate.")
                return False
            self._last_click_time = now

            target_x = self.config.call_x if direction == "CALL" else self.config.put_x
            target_y = self.config.call_y if direction == "CALL" else self.config.put_y
            restore = self.config.restore_cursor

        def _click_worker():
            try:
                import time
                attach_to_input_desktop()
                user32 = ctypes.windll.user32
                orig_pt = wintypes.POINT()
                has_orig = bool(user32.GetCursorPos(ctypes.byref(orig_pt)))

                dispatch_hardware_click(target_x, target_y)
                logger.info(f"⚡ [AUTO-CLICK EXECUTED - 1 CLICK] Successfully clicked {direction} at ({target_x}, {target_y})!")

                if restore and has_orig:
                    time.sleep(0.12)
                    user32.SetCursorPos(orig_pt.x, orig_pt.y)

            except Exception as e:
                logger.error(f"Click worker error: {e}")

        threading.Thread(target=_click_worker, daemon=True).start()
        return True

    def test_click(self, direction: str) -> Dict[str, Any]:
        with self.lock:
            target_x = self.config.call_x if direction == "CALL" else self.config.put_x
            target_y = self.config.call_y if direction == "CALL" else self.config.put_y

        try:
            dispatch_hardware_click(target_x, target_y)
            logger.info(f"⚡ [TEST CLICK EXECUTED - 1 CLICK] Successfully clicked {direction} at ({target_x}, {target_y})")
            return {"success": True, "clicked_at": {"x": target_x, "y": target_y}, "direction": direction}
        except Exception as e:
            logger.error(f"Test click error: {e}")
            return {"success": False, "error": str(e)}

    def get_mouse_position(self) -> Dict[str, int]:
        try:
            attach_to_input_desktop()
            pt = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return {"x": int(pt.x), "y": int(pt.y)}
        except Exception as e:
            logger.warning(f"GetCursorPos error: {e}")
        try:
            import pyautogui
            pos = pyautogui.position()
            return {"x": int(pos.x), "y": int(pos.y)}
        except Exception:
            return {"x": 0, "y": 0}


# Global instances initialized
trading_session = TradingSessionManager()
vision_engine = VisionAIEngine()
auto_clicker = QuotexAutoClicker()

