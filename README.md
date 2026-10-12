# Quotex Autonomous Paper-Trading & Self-Optimizing Agent

An autonomous quantitative paper-trading system that uses **Vision AI** and **Computer Vision** to ingest Quotex demo charts, analyze price action directly from the visual canvas, execute virtual paper trades, verify real win/loss outcomes at candle close, and automatically self-tune its strictness parameters until it converges at a minimum **80% Win Rate (4 wins out of 5 trades in a rolling window)**.

---

## ⚡ Key Capabilities & Architecture

1. **Vision Ingestion & Auto-Cropping (`VisionProcessor`)**:
   - Ingests high-resolution video frames via the browser's WebRTC Screen Capture API (`navigator.mediaDevices.getDisplayMedia`).
   - Uses OpenCV and Pillow to isolate the active candlestick canvas (last 20–30 candles and the right-hand price axis) while filtering out navigation menus and non-chart UI elements.
2. **Self-Optimizing Strategy Engine (`StrategyOptimizer`)**:
   - Dynamic 5-Tier Strictness Matrix (Levels 1 to 5: *Lenient* → *Balanced* → *Selective* → *Defense* → *Sniper*).
   - If a loss occurs and the rolling accuracy drops below 80%, the agent automatically tightens strictness thresholds (demands higher confidence scores, stronger rejection wicks, and stricter EMA/trend alignment).
   - Once the rolling 5-trade accuracy achieves $\ge 80\%$ (4/5 or 5/5 wins), the agent transitions to **`TARGET ACHIEVED: 80% WIN RATE`**.
3. **Dual Vision Engine (AI Vision + Zero-Config CV Fallback)**:
   - **OpenAI-Compatible / xAI Grok Vision**: Connects to `gpt-4o`, `grok-vision-beta`, or OpenRouter to evaluate candlestick patterns, order blocks, and rejection wicks.
   - **Internal Heuristic Computer-Vision Engine**: Standalone OpenCV pipeline that analyzes color ratios, recent momentum, and price delta even without an API key or when offline.
4. **Virtual Execution & Verification Ledger (`TradingSessionManager`)**:
   - **T = 0 (Entry Time):** Evaluates candle setup, determines `CALL` or `PUT`, records entry price and expiry countdown.
   - **T + Expiry (Verification Time):** Inspects the closing frame at candle expiration, compares `Close Price` vs. `Entry Price`, resolves `WIN`, `LOSS`, or `DRAW`, and calculates virtual profit/loss.
5. **Cyberpunk Trading Dashboard**:
   - Dark-mode terminal with WebRTC Quotex tab capture.
   - Rolling 5-trade visualizer (`[W, W, L, W, W]`).
   - Zero-dependency Web Audio API sound synthesizer (win arpeggio, loss buzz, trade notification, and victory fanfare).

---

## 📁 File Structure

```
trade/
├── requirements.txt         # Core dependencies (FastAPI, OpenCV, Pillow, OpenAI, etc.)
├── .env.example             # Configuration template
├── agent.py                 # Core state manager, paper ledger, vision & self-optimizer
├── main.py                  # FastAPI REST API server & auto-cycle controller
├── static/
│   └── index.html           # Real-time web dashboard with WebRTC capture & audio
└── README.md                # Documentation and operation guide
```

---

## 🚀 Quickstart & Installation

### 1. Install Dependencies

Ensure Python 3.10+ (or Python 3.14) is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create your `.env` file from the provided template:

```bash
cp .env.example .env
```

To use an AI Vision model (OpenAI `gpt-4o` or xAI `grok-vision-beta`), enter your API key in `.env` or configure it directly in the dashboard UI settings:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

> **Note:** If `OPENAI_API_KEY` is not provided, the agent seamlessly operates in **Heuristic Computer-Vision Mode** using OpenCV candlestick pattern detection.

### 3. Launch the Server

Run the application:

```bash
python main.py
```

The application will start at:
👉 **`http://localhost:8000`**

---

## 🎯 How to Use with Quotex Demo

1. Open your browser and navigate to **`http://localhost:8000`**.
2. Open your **Quotex Demo Account** in another browser window or tab:
   - Select a standard currency pair (e.g., `EUR/USD`, `GBP/USD`).
   - Set the chart type to **Candlestick** (Japanese Candles).
   - Set the timeframe to **1M (60s)** or **30s**.
3. In the Agent Dashboard:
   - Click **`📡 Share Quotex Tab`** and select the Quotex window.
   - Match the **Candle Timeframe** dropdown (default: `60s (1 Minute)`).
   - Click **`▶ Start Autonomous Agent`**.
4. **Autonomous Operation:**
   - The agent counts down each candle cycle.
   - Just before the candle closes, it captures the chart frame, crops the candlestick canvas, and evaluates whether to issue a `CALL`, `PUT`, or `SKIP`.
   - When a trade is placed, the **Active Paper Trade HUD** displays the entry price and countdown.
   - At candle expiration, it verifies the outcome, logs the PnL, updates the rolling 5-trade accuracy, and auto-tightens strictness if needed.
   - When the rolling accuracy achieves at least 4 wins out of 5 trades, the status lights up green: **`TARGET ACHIEVED: 80% WIN RATE`**.

---

## ⚙️ Strategy Strictness Levels

| Level | Mode | Min Confidence | Description |
|---|---|---|---|
| **1** | Lenient | 62% | Trades on standard momentum, engulfing bodies, and basic S&R bounces. |
| **2** | Balanced | 70% | Requires clean trend alignment (EMA slope) or distinct candlestick rejection wicks. |
| **3** | Selective | 78% | Demands high confluence: Trend alignment + Key Support/Resistance rejection. |
| **4** | High Strictness | 84% | Activated after losses. Filters out 80% of candles; requires institutional pin-bars. |
| **5** | Sniper Mode | 90% | Highest conviction only. Requires clear momentum exhaustion at extreme S&R bounce. |

---

## 🔌 API Endpoints Summary

- `GET /api/status`: Live session stats, rolling window, strictness level, and target achieved flag.
- `GET /api/trades`: Full historical paper-trading ledger.
- `POST /api/analyze-and-trade`: Ingests a chart frame, runs vision analysis, and opens a virtual paper trade.
- `POST /api/verify-trade`: Verifies an active paper trade at expiration and settles PnL.
- `POST /api/auto-cycle`: Unified lifecycle cycle runner (handles pending verification + setup analysis).
- `POST /api/config`: Updates API keys, models, payout rates, and strictness parameters.
- `POST /api/reset`: Resets demo balance and clears trade history.
