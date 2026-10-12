"""
main.py - FastAPI Application for Quotex Autonomous Paper-Trading & Self-Optimizing Agent.
Exposes REST APIs for screen frame analysis, virtual paper-trading execution,
automated trade verification, live session monitoring, and settings management.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from agent import (
    trading_session,
    vision_engine,
    auto_clicker,
    VisionProcessor,
    STRICTNESS_TIERS,
    SessionStats,
    TradeRecord
)

logger = logging.getLogger("QuotexAPI")

app = FastAPI(
    title="Quotex Autonomous Paper-Trading Agent API",
    description="Vision AI & Self-Optimizing Quantitative Agent for Quotex Demo Charts",
    version="1.0.0"
)

# Enable CORS for local development and browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# REQUEST & RESPONSE SCHEMAS
# ==============================================================================

class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded frame captured from browser")
    timeframe_seconds: int = Field(default=60, ge=5, le=3600, description="Candle timeframe in seconds")
    crop_bounds: Optional[Dict[str, float]] = Field(default=None, description="Custom crop fractions: top, bottom, left, right")
    auto_execute: bool = Field(default=True, description="Whether to execute paper trade automatically on CALL/PUT")


class VerifyRequest(BaseModel):
    trade_id: Optional[str] = Field(default=None, description="Trade ID to verify")
    image_base64: Optional[str] = Field(default=None, description="Base64 frame at candle expiry")
    manual_close_price: Optional[float] = Field(default=None, description="Explicit manual close price if desired")


class AutoCycleRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 frame captured from browser")
    timeframe_seconds: int = Field(default=60)
    crop_bounds: Optional[Dict[str, float]] = None
    force_verify: bool = Field(default=False, description="Force verification of active trade regardless of timer")


class ConfigUpdateRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    trade_amount: Optional[float] = Field(default=None, ge=1.0)
    payout_rate: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    strictness_level: Optional[int] = Field(default=None, ge=1, le=5)
    auto_optimize: Optional[bool] = None


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Serves the main dashboard application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status", response_model=SessionStats)
async def get_session_status():
    """Returns current live session metrics, rolling accuracy, and active trade."""
    return trading_session.get_stats()


@app.get("/api/trades", response_model=List[TradeRecord])
async def get_trade_history():
    """Returns all recorded trades in the current session."""
    with trading_session.lock:
        return list(trading_session.trade_history)


@app.post("/api/analyze-and-trade")
async def analyze_and_trade(payload: AnalyzeRequest):
    """
    Core Vision Ingestion & Decision Endpoint:
      1. Crops the Quotex chart frame.
      2. Runs Vision AI / Heuristic analysis.
      3. If decision is CALL/PUT and auto_execute is True, opens a virtual trade.
    """
    try:
        # 1. Ingest and Crop Chart Frame
        raw_pil = VisionProcessor.decode_base64_image(payload.image_base64)
        cropped_pil = VisionProcessor.auto_crop_quotex_chart(raw_pil, payload.crop_bounds)
        cropped_b64 = VisionProcessor.encode_image_base64(cropped_pil, format="JPEG", quality=85)

        # 2. Run Vision AI / Strategy Analysis
        decision_data = vision_engine.analyze_chart(
            image_b64=cropped_b64,
            strictness_level=trading_session.current_strictness,
            candle_timeframe=payload.timeframe_seconds
        )

        direction = decision_data["direction"]
        confidence = decision_data["confidence"]
        entry_price = decision_data["current_price"]
        trend = decision_data["trend"]
        setup_grade = decision_data["setup_grade"]
        reasoning = decision_data["reasoning"]

        trade_opened = None

        # 3. Virtual Paper Execution (T=0)
        if direction in ["CALL", "PUT"] and payload.auto_execute:
            with trading_session.lock:
                active_trade = trading_session.active_trade

            if active_trade is not None:
                # Active trade in flight, do not open another until expired
                logger.info("Trade signal generated but an active paper trade is already in progress.")
            else:
                trade_opened = trading_session.open_trade(
                    direction=direction,
                    entry_price=entry_price,
                    confidence=confidence,
                    rationale=reasoning,
                    trend=trend,
                    setup_grade=setup_grade,
                    timeframe_seconds=payload.timeframe_seconds,
                    indicators_summary={
                        "rejection_wick_found": decision_data.get("rejection_wick_found", False),
                        "key_level": decision_data.get("support_resistance_level", "")
                    }
                )
                # Execute direct mouse click on Quotex UP / DOWN button if enabled
                if auto_clicker.config.enabled:
                    auto_clicker.execute_click(direction)

        return {
            "success": True,
            "decision": decision_data,
            "trade_opened": trade_opened.dict() if trade_opened else None,
            "cropped_preview_b64": cropped_b64,
            "session": trading_session.get_stats().dict()
        }

    except Exception as e:
        logger.exception("Error during analyze_and_trade execution")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify-trade")
async def verify_trade(payload: VerifyRequest):
    """
    Outcome Verification Endpoint (T+Expiry):
      1. Determines closing price from the verification frame or input.
      2. Resolves trade as WIN, LOSS, or DRAW.
      3. Updates rolling accuracy and triggers strategy self-optimization.
    """
    try:
        active = trading_session.active_trade
        if not active:
            raise HTTPException(status_code=400, detail="No active paper trade to verify.")

        close_price = payload.manual_close_price
        if close_price is None:
            if not payload.image_base64:
                raise HTTPException(status_code=400, detail="Either manual_close_price or image_base64 must be provided.")
            
            # Extract close price from verification image
            raw_pil = VisionProcessor.decode_base64_image(payload.image_base64)
            cropped_pil = VisionProcessor.auto_crop_quotex_chart(raw_pil)
            cropped_b64 = VisionProcessor.encode_image_base64(cropped_pil)

            close_price = vision_engine.verify_price_from_frame(
                image_b64=cropped_b64,
                entry_price=active.entry_price,
                direction=active.direction
            )

        resolved_trade = trading_session.resolve_trade(
            close_price=close_price,
            trade_id=payload.trade_id or active.id
        )

        return {
            "success": True,
            "trade": resolved_trade.dict(),
            "session": trading_session.get_stats().dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during verify_trade execution")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-cycle")
async def auto_cycle(payload: AutoCycleRequest):
    """
    Automated Continuous Cycle Endpoint:
      - Checks if active trade is expired: if so, verifies it using current frame.
      - If no active trade: executes analysis and opens trade if high conviction.
    """
    try:
        active = trading_session.active_trade
        now = time.time()

        # Step 1: Check if pending trade needs verification
        if active and (now >= active.expiry_timestamp or payload.force_verify):
            # Verify pending trade
            raw_pil = VisionProcessor.decode_base64_image(payload.image_base64)
            cropped_pil = VisionProcessor.auto_crop_quotex_chart(raw_pil, payload.crop_bounds)
            cropped_b64 = VisionProcessor.encode_image_base64(cropped_pil)

            close_price = vision_engine.verify_price_from_frame(
                image_b64=cropped_b64,
                entry_price=active.entry_price,
                direction=active.direction
            )
            resolved = trading_session.resolve_trade(close_price=close_price, trade_id=active.id)
            return {
                "cycle_action": "TRADE_VERIFIED",
                "trade": resolved.dict(),
                "session": trading_session.get_stats().dict()
            }

        # Step 2: If trade is still in progress, wait
        if active and now < active.expiry_timestamp:
            remaining = int(active.expiry_timestamp - now)
            return {
                "cycle_action": "TRADE_IN_PROGRESS",
                "remaining_seconds": remaining,
                "active_trade": active.dict(),
                "session": trading_session.get_stats().dict()
            }

        # Step 3: No active trade -> Ingest and analyze frame
        raw_pil = VisionProcessor.decode_base64_image(payload.image_base64)
        cropped_pil = VisionProcessor.auto_crop_quotex_chart(raw_pil, payload.crop_bounds)
        cropped_b64 = VisionProcessor.encode_image_base64(cropped_pil)

        decision = vision_engine.analyze_chart(
            image_b64=cropped_b64,
            strictness_level=trading_session.current_strictness,
            candle_timeframe=payload.timeframe_seconds
        )

        direction = decision["direction"]
        opened_trade = None

        if direction in ["CALL", "PUT"]:
            opened_trade = trading_session.open_trade(
                direction=direction,
                entry_price=decision["current_price"],
                confidence=decision["confidence"],
                rationale=decision["reasoning"],
                trend=decision["trend"],
                setup_grade=decision["setup_grade"],
                timeframe_seconds=payload.timeframe_seconds
            )
            action = f"TRADE_OPENED_{direction}"
            # Execute direct mouse click on Quotex UP / DOWN button if enabled
            if auto_clicker.config.enabled:
                auto_clicker.execute_click(direction)
        else:
            action = "SKIPPED_FILTERED"

        return {
            "cycle_action": action,
            "decision": decision,
            "trade": opened_trade.dict() if opened_trade else None,
            "cropped_preview_b64": cropped_b64,
            "session": trading_session.get_stats().dict()
        }

    except Exception as e:
        logger.exception("Error during auto_cycle execution")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# DIRECT QUOTEX AUTO-CLICKER ENDPOINTS
# ==============================================================================

class AutoClickUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    call_x: Optional[int] = None
    call_y: Optional[int] = None
    put_x: Optional[int] = None
    put_y: Optional[int] = None
    restore_cursor: Optional[bool] = None


class TestClickRequest(BaseModel):
    direction: str = Field(..., description="CALL or PUT")


@app.get("/api/autoclick/config")
async def get_autoclick_config():
    """Returns current auto-clicker status and target coordinates."""
    return auto_clicker.config.dict()


@app.post("/api/autoclick/config")
async def update_autoclick_config(payload: AutoClickUpdateRequest):
    """Updates auto-clicker coordinates and enable/disable state."""
    auto_clicker.update_config(
        enabled=payload.enabled,
        call_x=payload.call_x,
        call_y=payload.call_y,
        put_x=payload.put_x,
        put_y=payload.put_y,
        restore_cursor=payload.restore_cursor
    )
    return {"success": True, "config": auto_clicker.config.dict()}


@app.get("/api/autoclick/mouse-pos")
async def get_mouse_position():
    """Returns live screen coordinates of the user's cursor for calibration."""
    return auto_clicker.get_mouse_position()


@app.post("/api/autoclick/test-click")
async def test_autoclick(payload: TestClickRequest):
    """Performs an immediate single test click on the target button."""
    return auto_clicker.test_click(payload.direction.upper())


@app.post("/api/autoclick/auto-detect")
async def auto_detect_buttons():
    """Automatically scans visible screen to find Quotex Green Buy and Red Sell buttons."""
    return auto_clicker.auto_detect_quotex_buttons()



@app.post("/api/reset")
async def reset_session():
    """Resets paper trading balance, ledger history, and rolling accuracy."""
    trading_session.reset_session()
    return {"success": True, "message": "Session reset successfully", "session": trading_session.get_stats().dict()}


@app.get("/api/config")
async def get_config():
    """Returns runtime AI and strategy configuration."""
    api_key_masked = "Not Set (Using Heuristic CV)"
    if vision_engine.api_key:
        api_key_masked = f"{vision_engine.api_key[:4]}...{vision_engine.api_key[-4:]}"

    return {
        "api_key_configured": bool(vision_engine.api_key),
        "api_key_masked": api_key_masked,
        "base_url": vision_engine.base_url,
        "model_name": vision_engine.model_name,
        "trade_amount": trading_session.trade_amount,
        "payout_rate": trading_session.payout_rate,
        "current_strictness": trading_session.current_strictness,
        "auto_optimize": trading_session.auto_optimize,
        "strictness_tiers": {k: v.dict() for k, v in STRICTNESS_TIERS.items()}
    }


@app.post("/api/config")
async def update_config(payload: ConfigUpdateRequest):
    """Updates runtime AI engine and trading parameters."""
    if payload.api_key is not None or payload.base_url is not None or payload.model_name is not None:
        vision_engine.update_config(
            api_key=payload.api_key,
            base_url=payload.base_url,
            model_name=payload.model_name
        )

    with trading_session.lock:
        if payload.trade_amount is not None:
            trading_session.trade_amount = payload.trade_amount
        if payload.payout_rate is not None:
            trading_session.payout_rate = payload.payout_rate
        if payload.strictness_level is not None:
            trading_session.current_strictness = payload.strictness_level
        if payload.auto_optimize is not None:
            trading_session.auto_optimize = payload.auto_optimize

    return {"success": True, "message": "Configuration updated successfully."}


# ==============================================================================
# SERVER ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Quotex Autonomous Paper-Trading Agent at http://localhost:{port}")
    uvicorn.run("main.py:app", host=host, port=port, reload=True)
