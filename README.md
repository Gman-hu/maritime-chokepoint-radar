# Maritime Chokepoint Transit Radar & VECM Forecasting Engine

Interactive macroeconomic and logistical analytics terminal modeling daily vessel transits across the **Suez Canal** and the **Cape of Good Hope**, powered by a spatial Vector Error Correction Model (VECM).

* **Data Sources:** IMF PortWatch API (daily transit volume) & Yahoo Finance (`BZ=F` Brent Crude bunker fuel proxy).
* **Production Engine:** Cointegrated VECM (`coint_rank=1`, `k_ar_diff=15`) with live multi-step forecasting and residual Volatility Seismograph tracking.
* **Stack:** Python 3.13, Plotly Dash, Statsmodels, Gunicorn, Render.