# Maritime Chokepoint Transit Radar & VECM Forecasting Engine

[![Live Terminal](https://img.shields.io/badge/Live_Radar-Render_Cloud-00F2FE?style=for-the-badge&logo=render&logoColor=white)](https://maritime-chokepoint-radar.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly Dash](https://img.shields.io/badge/Frontend-Plotly_Dash-FF9900?style=for-the-badge&logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Statsmodels](https://img.shields.io/badge/Econometrics-Statsmodels_VECM-10B981?style=for-the-badge)](https://www.statsmodels.org/)

> **Live Production Terminal:** [https://maritime-chokepoint-radar.onrender.com](https://maritime-chokepoint-radar.onrender.com) *(Note: Hosted on Render's Free tier; allow ~30–45 seconds if waking from an idle cold start.)*

An end-to-end macroeconomic time series research and cloud inference system modeling daily cargo vessel transits across the **Suez Canal** and the **Cape of Good Hope**[cite: 6, 9]. Powered by high-frequency satellite AIS transit records from the **IMF PortWatch API** and **Brent Crude (`BZ=F`)** bunker fuel prices, this project pairs rigorous econometric hypothesis testing and multi-tier machine learning falsification with a live **Plotly Dash** interactive forecasting radar[cite: 6, 8, 9].

---

## 1. Macroeconomic Problem & Domain Motivation

Maritime trade between Asia and Europe operates on a physical, closed-loop corridor governed by a binary spatial routing choice[cite: 6, 9]:
* **The Suez Canal / Bab-el-Mandeb Corridor:** The fastest transit artery (~8,500 nm), subject to canal toll fees and acute regional security risks[cite: 6, 9].
* **The Cape of Good Hope Detour:** Circumnavigates southern Africa to avoid conflict zones, adding **10 to 14 days** of sailing time and substantial bunker fuel consumption[cite: 6, 9].

Spanning **2,876 sanitized daily observations** (`2019-01-01` to `2026-11-15`), the dataset captures four distinct logistical regimes: pre-pandemic equilibrium, COVID-19 supply-chain bottlenecks ("Slow Repositioning"), the March 2021 *Ever Given* canal blockage, and the late-2023+ Red Sea crisis ("The Great Rerouting")[cite: 1, 6, 9].

### Why a Strict Time Series Framework is Mathematically Mandatory
Applying standard cross-sectional machine learning violates the independent and identically distributed ($i.i.d.$) assumption because global vessel positions today are physically and contractually bound to fleet distributions yesterday[cite: 6, 9]:
* **Autoregressive Memory & Fleet Inertia:** Canal lock throughput ceilings carry day $t$ backlogs into day $t+1$, while a two-week Cape detour ties up vessel capacity and alters return schedules a month later[cite: 6, 9].
* **Dynamic Propagation Lags:** A disruption at Bab-el-Mandeb at time $t$ generates a delayed transit surge at the Cape of Good Hope at time $t + \tau$[cite: 6, 9].
* **Non-Stationarity & Structural Breaks:** Chokepoint volumes do not oscillate around a fixed historical mean; they exhibit unit roots, stochastic trends, and abrupt regime shifts[cite: 6, 9].

---

## 2. End-to-End Research & Engineering Pipeline

### Phase 1: Automated Data Mining & Temporal Alignment
* **Target Series ($Y$) — IMF PortWatch ArcGIS FeatureServer:** Built paginated Python extraction loops (`resultRecordCount=1000`, dynamic `resultOffset`) querying open ArcGIS REST endpoints for `Suez Canal` (`chokepoint1`) and `Cape of Good Hope` (`chokepoint7`), parsing nested JSON `n_total` vessel counts and converting Unix millisecond timestamps into normalized daily UTC indices[cite: 9, 10].
* **Exogenous Economic Context ($X$) — Brent Crude (`BZ=F`):** Extracted historical Brent Crude futures via `yfinance` (`auto_adjust=True`) as a continuous proxy for global maritime bunker fuel costs[cite: 9, 10].
* **Continuous Calendar Alignment:** Unified maritime and financial series via outer timestamp merges and applied forward-filling (`ffill`) across weekend market closures when cargo fleets remain at sea[cite: 6, 9, 10].

### Phase 2: Econometric Hypothesis Testing & EDA
* **Stationarity Testing (Augmented Dickey-Fuller):** Formal ADF testing with AIC lag selection returned $p = 0.5619$ for the Suez Canal and $p = 0.3132$ for the Cape of Good Hope ($p > 0.05$), proving both series contain unit roots and require differencing or cointegration modeling[cite: 9, 10].
* **Structural Break Detection (Bai-Perron / Pelt Algorithm):** Standardized signal segmentation via `ruptures` (`rpt.Pelt`, `min_size=30`) statistically isolated **5 Suez breaks** and **3 Cape breaks**, pinpointing the March 2021 *Ever Given* grounding and synchronized late-2023 / early-2024 Red Sea rerouting fractures without visual guesswork[cite: 1, 7, 9].
* **Cross-Correlation Function (CCF) Impulse Analysis:** Computing the CCF on first-differenced stationary series across $-30$ to $+30$ day lags revealed statistically significant negative correlation spikes at **+10 to +14 days**, mathematically verifying the exact physical sailing delay around Africa[cite: 1, 7].

### Phase 3: Spatial Cointegration Baseline (SARIMAX $\rightarrow$ VECM)
* **Univariate Baseline Limitation (`SARIMAX`):** An initial `SARIMAX(1,1,1)x(1,0,1,7)` model mapped weekly port schedules during calm windows but exhibited non-normal fat tails (`Jarque-Bera p = 0.00`) and exploding crisis variance (`Heteroskedasticity p = 0.00`) while remaining mathematically blind to cross-route substitution[cite: 7, 10].
* **Multivariate Upgrade (`VECM`):** Replaced SARIMAX with a simultaneous **Vector Error Correction Model** (`coint_rank=1`, AIC-selected lag order `k_ar_diff=15`), capturing both long-run closed-loop spatial equilibrium ($\text{ECT}_t = \beta' y_t + c_{\text{coint}}$) and short-run dynamic route adjustments[cite: 1, 7, 9].
* **Shock Isolation:** Extracted dual-route VECM residuals (`suez_residual_error`, `cape_residual_error`) across **2,860 aligned post-lag days** to isolate pure, shock-driven geopolitical volatility[cite: 1, 7].

### Phase 4: Non-Linear Residual Learning & The Principle of Parsimony
To test whether machine learning and deep sequence networks could predict uncaptured geopolitical shock residuals ($\hat{Y}_{\text{Hybrid}, t} = \hat{Y}_{\text{VECM}, t} + \hat{e}_{\text{ML}, t}$), two progressively complex architectures were evaluated under strict **5-Fold Expanding Window Cross-Validation (`TimeSeriesSplit`)** to prevent lookahead leakage[cite: 7, 8, 10]:

| Modeling Tier | Architecture & Specification | Out-of-Fold RMSE (Ships) | Out-of-Fold MAE (Ships) | Empirical Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Tier 1 (Winner)** | **Standalone Cointegrated VECM** (`r=1`, `k=15`)[cite: 1, 7] | **8.0236**[cite: 7] | **6.1634**[cite: 7] | **Deployed to Production** — Captures spatial equilibrium without overfitting noise[cite: 8, 9]. |
| **Tier 2** | **Hybrid VECM + XGBoost** (`14` lag/rolling features, `max_depth=4`)[cite: 7, 10] | `8.3987` *(-4.68%)*[cite: 7] | `6.4080` *(-3.97%)*[cite: 7] | **Rejected** — Squared-error variance shrinkage damped tails; trees failed to extrapolate unprecedented shocks[cite: 7, 8]. |
| **Tier 3** | **PyTorch TCN + Tail-Aware Loss** (`15d` causal dilated windows `[64, 15, 7]`, $\alpha=2.0$)[cite: 8] | `8.9699`[cite: 8] | `6.9178`[cite: 8] | **Rejected (Skipped Recombination)** — Chased crisis tails (`7.07` RMSE in Fold 5) but suffered data starvation & phantom volatility in calm periods[cite: 8]. |

> **Key Econometric Takeaway:** In ~2,800-row macroeconomic series governed by discrete, human-driven geopolitical interventions, gradient boosting dampens tail variance and deep convolutional networks starve for training sequences—confirming the standalone **VECM** as the mathematically superior production engine[cite: 8, 9].

---

## 3. Phase 5: Interactive "Modern Maritime Radar" Terminal

The validated VECM production engine is served through a custom dark-themed **Plotly Dash** web application designed around naval telemetry ergonomics:

* **Executive Telemetry Ribbon:** Six live monospace KPI scorecards tracking Suez and Cape 7-day moving averages, Cape diversion share (%), the live Cointegrating Error Correction Term ($\text{ECT}_t$), Brent Crude (`BZ=F`) bunker fuel prices, the composite Geopolitical Volatility Index ($\sigma$), and Phase 4 RMSE benchmarks[cite: 7, 8, 9].
* **Mission Control & Pesaran-Shin GIRF Shock Simulator:** Interactive scenario presets (*Acute Red Sea Rerouting*, *7-Day Canal Blockage*, *Red Sea De-Escalation*, *Bunker Fuel Spike*) plus custom sliders for forecast horizon ($14$–$90$ days), Suez vessel shocks ($\Delta$ ships/day), Brent Crude shocks ($\Delta\%$), and decay profiles (*Sustained*, *7d Transient*, *1d Impulse*).
* **Asia–Europe Maritime Loop Radar (`Scattergeo`):** Hardware-accelerated vector map from Malacca to Rotterdam with dynamic neon route widths and pulsing chokepoint markers scaled to real-time and projected corridor shares.
* **Dual-Trace Trajectories & Lütkepohl Confidence Cones:** Plots historical 7-day moving averages alongside out-of-sample VECM forecasts, analytical 80%/95% uncertainty cones, and toggleable Pelt structural break overlays[cite: 1, 7].
* **Empirical CCF & Analytical GIRF Verifier:** Two-tier impulse panel pairing the empirical $-30$ to $+30$ day CCF table with live VECM Generalized Impulse Response trajectories across the $+10$ to $+14$ day physical sailing window[cite: 1, 7].
* **Exogenous Bunker Fuel Tracker & Volatility Seismograph:** Dual-axis Brent Crude vs. Cape Share economics chart paired with a bottom-anchored EKG residual seismograph ($\pm 2.25\sigma$ shock thresholds)[cite: 1, 7, 9].

---

## 4. Repository Structure

```text
maritime-chokepoint-radar/
├── app.py                                  # Plotly Dash layout, Scattergeo radar & reactive callbacks
├── vecm_engine.py                          # Stateless VECM inference, Lütkepohl cones & GIRF shock engine
├── assets/
│   └── style.css                           # Auto-injected "Modern Maritime Radar" CSS theme
├── models/
│   └── vecm_fitted.pkl                     # Serialized full-history VECM artifact (k_ar_diff=15, r=1)
├── data/
│   ├── chokepoint_dashboard_data.parquet   # Primary fast-load aligned dataset & VECM residuals (2,860 rows)
│   ├── chokepoint_dashboard_data.csv       # CSV mirror of aligned PortWatch + Brent Crude metrics
│   ├── ccf_lags.csv                        # Pre-computed -30 to +30 day Cross-Correlation lag table
│   └── structural_breaks.json              # Pelt regime-shift timestamps & annotated crisis metadata
├── requirements.txt                        # Pinned cloud production dependencies (NumPy 2.x, Statsmodels 0.15)
├── render.yaml                             # Render cloud infrastructure-as-code blueprint
├── runtime_versions.json                   # Serialization environment manifest (Python 3.13.18)
└── .python-version                         # Enforces Python 3.13 binary pickle parity on Render
```

---

## 5. Local Installation & Execution

### Prerequisites
* **Python 3.12 or 3.13** (`3.13` recommended to match the `python 3.13.18` / `numpy 2.4.6` / `statsmodels 0.15.0` serialization manifest in `runtime_versions.json`)[cite: 5].

### 1. Clone the Repository
```bash
git clone [https://github.com/Gman-hu/maritime-chokepoint-radar.git](https://github.com/Gman-hu/maritime-chokepoint-radar.git)
cd maritime-chokepoint-radar
```

### 2. Create a Virtual Environment & Install Dependencies
**Windows (PowerShell):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### 3. Run the Econometric Parity Self-Test & Launch the Server
```bash
# Verify machine-precision VAR recursion parity (< 1e-13 ships) & live telemetry readout
python vecm_engine.py

# Launch local Plotly Dash server at [http://127.0.0.1:8050](http://127.0.0.1:8050)
python app.py
```

---

## 6. Cloud Production Architecture (Render CI/CD)

The application is deployed continuously from `main` to **Render** using a memory-optimized Linux WSGI configuration:
* **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
* **Start Command:** `gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --threads 2 --preload --timeout 120`
* **Copy-on-Write Memory Sharing (`--preload`):** Loads `models/vecm_fitted.pkl` and `data/chokepoint_dashboard_data.parquet` once in the master process before forking `gthread` workers, keeping total container RAM around ~200 MB—comfortably inside Render's 512 MB limit[cite: 2, 3].

---

## 7. Data Sources & Acknowledgments

* **Maritime Transit Volumes:** [IMF PortWatch](https://portwatch.imf.org/) (*Daily Chokepoint Transit Calls and Trade Volume Estimates*, open ArcGIS FeatureServer)[cite: 9, 10].
* **Energy & Bunker Fuel Proxy:** ICE Brent Crude Oil Futures (`BZ=F`) retrieved programmatically via [`yfinance`](https://pypi.org/project/yfinance/)[cite: 9, 10].
