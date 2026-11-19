"""
vecm_engine.py
Production Vector Error Correction Model (VECM) Inference & Scenario Shock Engine
Project: Maritime Chokepoint Transit Radar (Suez Canal vs. Cape of Good Hope)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ==============================================================================
# PATH CONFIGURATION & EMPIRICAL BENCHMARK CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Empirically validated 5-Fold Expanding Window metrics from Phases 3 & 4
MODEL_BENCHMARKS: dict[str, float] = {
    "vecm_standalone_rmse": 8.0236,
    "vecm_standalone_mae": 6.1634,
    "vecm_xgb_hybrid_rmse": 8.3987,
    "vecm_xgb_hybrid_mae": 6.4080,
    "tcn_deep_rmse": 8.9699,
    "tcn_deep_mae": 6.9178,
}

# Pre-configured geopolitical & macroeconomic scenario presets for dashboard UI
SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Unperturbed VECM Baseline",
        "suez_shock_ships": 0.0,
        "brent_shock_pct": 0.0,
        "shock_profile": "sustained",
    },
    "red_sea_escalation": {
        "label": "Acute Red Sea Rerouting (-30 Suez Ships, +15% Fuel)",
        "suez_shock_ships": -30.0,
        "brent_shock_pct": 15.0,
        "shock_profile": "sustained",
    },
    "ever_given_blockage": {
        "label": "7-Day Canal Blockage (-40 Suez Ships, Reopens Day 7)",
        "suez_shock_ships": -40.0,
        "brent_shock_pct": 8.0,
        "shock_profile": "transient_7d",
    },
    "corridor_normalization": {
        "label": "Red Sea De-Escalation (+25 Suez Ships, -10% Fuel)",
        "suez_shock_ships": 25.0,
        "brent_shock_pct": -10.0,
        "shock_profile": "sustained",
    },
    "bunker_fuel_spike": {
        "label": "Bunker Fuel Shock (+35% Brent Crude Surge)",
        "suez_shock_ships": 0.0,
        "brent_shock_pct": 35.0,
        "shock_profile": "sustained",
    },
}


# ==============================================================================
# CORE INFERENCE ENGINE CLASS
# ==============================================================================


class MaritimeVECMEngine:
    """
    Stateless, memory-cached inference engine wrapping the serialized VECM,
    aligned PortWatch + Brent Crude dataset, CCF lags, and Pelt structural breaks.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # 1. Load pre-computed datasets & structural break metadata
        parquet_path = self.data_dir / "chokepoint_dashboard_data.parquet"
        ccf_path = self.data_dir / "ccf_lags.csv"
        breaks_path = self.data_dir / "structural_breaks.json"
        model_path = self.models_dir / "vecm_fitted.pkl"

        for required_path in (parquet_path, ccf_path, breaks_path, model_path):
            if not required_path.exists():
                raise FileNotFoundError(f"Missing required production artifact: {required_path}")

        raw_df = pd.read_parquet(parquet_path)
        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.tz_localize(None).dt.normalize()
        self.df = raw_df.sort_values("date").reset_index(drop=True)

        self.ccf_df = pd.read_csv(ccf_path).sort_values("lag_days").reset_index(drop=True)
        with open(breaks_path, "r", encoding="utf-8") as f:
            self.structural_breaks: dict[str, Any] = json.load(f)

        # 2. Load serialized VECM dictionary artifact
        self.artifact: dict[str, Any] = joblib.load(model_path)
        self.vecm_results = self.artifact.get("vecm_results")
        self.endog_names: list[str] = list(
            self.artifact.get("endog_names", ["suez_transit_count", "cape_transit_count"])
        )
        self.k_ar: int = int(self.artifact["k_ar"])
        self.k_ar_diff: int = int(self.artifact["k_ar_diff"])
        self.coint_rank: int = int(self.artifact["coint_rank"])

        self.alpha = np.asarray(self.artifact["alpha"], dtype=float)
        self.beta = np.asarray(self.artifact["beta"], dtype=float)
        self.gamma = np.asarray(self.artifact["gamma"], dtype=float)
        self.sigma_u = np.asarray(self.artifact["sigma_u"], dtype=float)
        self.last_k_obs = np.asarray(self.artifact["last_k_observations"], dtype=float)
        self.last_date = pd.to_datetime(self.artifact["last_date"])

        # 3. Extract level-VAR representation (A_1 ... A_p) & exact constant intercept
        self.var_rep = self._extract_var_representation()
        self.level_intercept = self._calibrate_level_intercept()

        # 4. Enrich historical DataFrame with Volatility Seismograph & ECT series
        self._enrich_historical_metrics()

    # --------------------------------------------------------------------------
    # INTERNAL ECONOMETRIC INITIALIZATION HELPERS
    # --------------------------------------------------------------------------

    def _extract_var_representation(self) -> np.ndarray:
        """
        Extracts the (k_ar, 2, 2) level-VAR parameter matrices A_1 ... A_{k_ar}.
        Includes an analytical Lütkepohl (Eq. 6.2.3) fallback from raw matrices.
        """
        if self.vecm_results is not None and hasattr(self.vecm_results, "var_rep"):
            return np.asarray(self.vecm_results.var_rep, dtype=float)

        neqs = len(self.endog_names)
        pi_mat = self.alpha @ self.beta.T
        gamma_mats = [
            self.gamma[:, j * neqs : (j + 1) * neqs] for j in range(self.k_ar_diff)
        ]

        var_mats = np.zeros((self.k_ar, neqs, neqs), dtype=float)
        var_mats[0] = np.eye(neqs) + pi_mat + gamma_mats[0]
        for j in range(1, self.k_ar_diff):
            var_mats[j] = gamma_mats[j] - gamma_mats[j - 1]
        var_mats[self.k_ar - 1] = -gamma_mats[self.k_ar_diff - 1]
        return var_mats

    def _calibrate_level_intercept(self) -> np.ndarray:
        """
        Calibrates the invariant deterministic vector nu = alpha * c_coint in levels
        so recursive custom forecasts match statsmodels VECMResults.predict() to 1e-12.
        """
        ar_step1 = np.einsum("jik,jk->i", self.var_rep, self.last_k_obs[::-1])
        if self.vecm_results is not None:
            fc_step1 = np.asarray(self.vecm_results.predict(steps=1), dtype=float)[0]
            return fc_step1 - ar_step1
        return np.zeros(len(self.endog_names), dtype=float)

    def _compute_ma_matrices(self, max_steps: int) -> np.ndarray:
        """
        Computes Moving Average (Wold impulse) matrices Phi_0, Phi_1, ..., Phi_{max_steps}
        of shape (max_steps + 1, 2, 2) via Lütkepohl recursion (Eq. 2.1.22).
        """
        neqs = len(self.endog_names)
        phi = np.zeros((max_steps + 1, neqs, neqs), dtype=float)
        phi[0] = np.eye(neqs)

        for i in range(1, max_steps + 1):
            upper_j = min(i, self.k_ar)
            acc = np.zeros((neqs, neqs), dtype=float)
            for j in range(1, upper_j + 1):
                acc += phi[i - j] @ self.var_rep[j - 1]
            phi[i] = acc
        return phi

    def _enrich_historical_metrics(self) -> None:
        """
        Pre-computes standardized residual z-scores, the composite Geopolitical
        Volatility Index (GVI), and the Cointegrating Error Correction Term (ECT).
        """
        resid_stds = np.sqrt(np.diag(self.sigma_u))
        suez_std = max(float(resid_stds[0]), 1e-6)
        cape_std = max(float(resid_stds[1]), 1e-6)

        self.df["suez_resid_z"] = self.df["suez_residual_error"] / suez_std
        self.df["cape_resid_z"] = self.df["cape_residual_error"] / cape_std

        # Root-mean-square standardized residual shock across both chokepoints
        self.df["geopolitical_volatility_index"] = np.sqrt(
            0.5 * (self.df["suez_resid_z"] ** 2 + self.df["cape_resid_z"] ** 2)
        )
        self.df["volatility_7d_ma"] = (
            self.df["geopolitical_volatility_index"].rolling(window=7, min_periods=1).mean()
        )

        # Cointegrating Error Correction Term: ECT_t = beta' * y_t + c_coint
        denom = (self.alpha.T @ self.alpha).item()
        self.c_coint = (
            (self.alpha.T @ self.level_intercept).item() / denom if denom > 1e-12 else 0.0
        )
        endog_matrix = self.df[self.endog_names].to_numpy(dtype=float)
        self.df["coint_ect"] = (endog_matrix @ self.beta).ravel() + self.c_coint

    # --------------------------------------------------------------------------
    # PUBLIC DATA & TELEMETRY ACCESSORS
    # --------------------------------------------------------------------------

    def get_historical_slice(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Returns a chronologically filtered view of the aligned historical dataset."""
        mask = pd.Series(True, index=self.df.index)
        if start_date is not None:
            mask &= self.df["date"] >= pd.to_datetime(start_date)
        if end_date is not None:
            mask &= self.df["date"] <= pd.to_datetime(end_date)
        return self.df.loc[mask].copy()

    def get_structural_breaks(self) -> dict[str, Any]:
        """Returns Pelt regime-shift dates and annotated historical crises."""
        return self.structural_breaks

    def get_kpi_snapshot(
        self, as_of_date: str | pd.Timestamp | None = None
    ) -> dict[str, Any]:
        """
        Computes live executive telemetry for top-bar scorecards and map nodes.
        """
        sub = self.get_historical_slice(end_date=as_of_date)
        if sub.empty:
            sub = self.df

        latest = sub.iloc[-1]
        prev_7d = sub.iloc[-8] if len(sub) >= 8 else sub.iloc[0]

        def pct_delta(curr: float, prev: float) -> float:
            return ((curr - prev) / prev * 100.0) if abs(prev) > 1e-6 else 0.0

        gvi_val = float(latest["geopolitical_volatility_index"])
        gvi_7d = float(latest["volatility_7d_ma"])
        peak_val = max(gvi_val, gvi_7d)
        if peak_val >= 2.25:
            regime_status = "CRITICAL SHOCK"
        elif peak_val >= 1.35:
            regime_status = "ELEVATED STRESS"
        else:
            regime_status = "NOMINAL EQUILIBRIUM"

        return {
            "as_of_date": latest["date"].strftime("%Y-%m-%d"),
            "total_observations": int(len(sub)),
            # Suez telemetry
            "suez_latest": float(latest["suez_transit_count"]),
            "suez_7d_ma": float(latest["suez_7d_ma"]),
            "suez_7d_pct_change": pct_delta(
                float(latest["suez_7d_ma"]), float(prev_7d["suez_7d_ma"])
            ),
            "suez_latest_residual": float(latest["suez_residual_error"]),
            # Cape telemetry
            "cape_latest": float(latest["cape_transit_count"]),
            "cape_7d_ma": float(latest["cape_7d_ma"]),
            "cape_7d_pct_change": pct_delta(
                float(latest["cape_7d_ma"]), float(prev_7d["cape_7d_ma"])
            ),
            "cape_latest_residual": float(latest["cape_residual_error"]),
            # Route substitution & fuel context
            "cape_share_pct": float(latest["cape_share_pct"]),
            "brent_latest": float(latest["brent_fuel_price"]),
            "brent_7d_delta": float(latest["brent_fuel_price"] - prev_7d["brent_fuel_price"]),
            "brent_7d_pct_change": pct_delta(
                float(latest["brent_fuel_price"]), float(prev_7d["brent_fuel_price"])
            ),
            # Volatility & Cointegration diagnostics
            "volatility_index": gvi_val,
            "volatility_7d_ma": gvi_7d,
            "regime_status": regime_status,
            "coint_ect": float(latest["coint_ect"]),
            # Model parameters & Phase 4 empirical validation benchmarks
            "k_ar_diff": self.k_ar_diff,
            "coint_rank": self.coint_rank,
            "alpha_suez": float(self.alpha[0, 0]),
            "alpha_cape": float(self.alpha[1, 0]),
            "beta_cape": float(self.beta[1, 0]),
            "benchmarks": MODEL_BENCHMARKS,
        }

    # --------------------------------------------------------------------------
    # DYNAMIC FORECASTING & SCENARIO SHOCK SIMULATION
    # --------------------------------------------------------------------------

    def _forecast_from_window(self, init_window: np.ndarray, steps: int) -> np.ndarray:
        """
        Recursively projects multi-step levels from any (k_ar, 2) observation window.
        """
        if init_window.shape != (self.k_ar, len(self.endog_names)):
            raise ValueError(
                f"Expected initial window shape {(self.k_ar, len(self.endog_names))}, "
                f"received {init_window.shape}."
            )

        buffer = init_window.astype(float).copy()
        preds = np.zeros((steps, len(self.endog_names)), dtype=float)

        for step in range(steps):
            ar_term = np.einsum("jik,jk->i", self.var_rep, buffer[-self.k_ar :][::-1])
            next_val = self.level_intercept + ar_term
            preds[step] = next_val
            buffer = np.vstack([buffer[1:], next_val])

        return preds

    def generate_forecast(
        self,
        horizon: int = 30,
        suez_shock_ships: float = 0.0,
        brent_shock_pct: float = 0.0,
        shock_profile: str = "sustained",
    ) -> pd.DataFrame:
        """
        Computes out-of-sample VECM transit forecasts with 80% & 95% confidence cones
        and optional Pesaran-Shin Generalized Impulse Response (GIRF) shock injection.

        Parameters
        ----------
        horizon : int
            Number of days ahead to forecast (1 to 120).
        suez_shock_ships : float
            Simulated daily vessel shock at the Suez Canal (e.g., -35 for blockade).
        brent_shock_pct : float
            Simulated percentage change in Brent Crude bunker fuel cost (e.g., +25%).
        shock_profile : str
            'sustained' (Red Sea-type regime shift), 'transient_7d' (Ever Given-type
            7-day blockage & reopening), or 'impulse' (1-day port disruption).
        """
        horizon = int(np.clip(horizon, 1, 120))

        # 1. Baseline forecast & analytical Lütkepohl forecast error intervals
        if self.vecm_results is not None:
            base_fc, low_95, up_95 = self.vecm_results.predict(steps=horizon, alpha=0.05)
            _, low_80, up_80 = self.vecm_results.predict(steps=horizon, alpha=0.20)
        else:
            base_fc = self._forecast_from_window(self.last_k_obs, steps=horizon)
            phi_tmp = self._compute_ma_matrices(horizon)
            mse_diag = np.zeros((horizon, 2), dtype=float)
            cum_cov = np.zeros((2, 2), dtype=float)
            for h in range(horizon):
                cum_cov = cum_cov + phi_tmp[h] @ self.sigma_u @ phi_tmp[h].T
                mse_diag[h] = np.diag(cum_cov)
            se = np.sqrt(np.maximum(mse_diag, 0.0))
            low_95, up_95 = base_fc - 1.95996 * se, base_fc + 1.95996 * se
            low_80, up_80 = base_fc - 1.28155 * se, base_fc + 1.28155 * se

        half_w95 = up_95 - base_fc
        half_w80 = up_80 - base_fc

        # 2. Propagate exogenous / structural shocks via VECM Moving Average matrices
        delta_trajectory = np.zeros_like(base_fc)
        has_shock = (abs(suez_shock_ships) > 1e-6) or (abs(brent_shock_pct) > 1e-6)

        if has_shock:
            phi = self._compute_ma_matrices(horizon)
            innovations = np.zeros((horizon, 2), dtype=float)

            # Pesaran-Shin GIRF vector conditioned on Suez innovation size
            sigma_00 = max(float(self.sigma_u[0, 0]), 1e-6)
            girf_suez_vec = (self.sigma_u[:, 0] / sigma_00) * float(suez_shock_ships)

            # Economic bunker fuel cost-rebalancing vector (higher fuel penalizes longer Cape detour)
            resid_stds = np.sqrt(np.maximum(np.diag(self.sigma_u), 1e-6))
            fuel_intensity = (float(brent_shock_pct) / 10.0) * 0.18
            fuel_vec = np.array(
                [fuel_intensity * resid_stds[0], -fuel_intensity * resid_stds[1]],
                dtype=float,
            )

            combined_impulse = girf_suez_vec + fuel_vec
            innovations[0] = combined_impulse

            # Apply I(1) temporal innovation structure
            if shock_profile == "transient_7d" and horizon > 7:
                # Reopening counter-shock on Day 7 restores pre-blockage corridor flow
                innovations[7] = -0.88 * combined_impulse
            elif shock_profile == "impulse" and horizon > 1:
                # Next-day recovery leaves only short-run lag ripples
                innovations[1] = -0.92 * combined_impulse

            # Convolve innovations through Moving Average matrices Phi_0 ... Phi_{H-1}
            for h in range(horizon):
                acc = np.zeros(2, dtype=float)
                for j in range(h + 1):
                    if np.any(innovations[j] != 0.0):
                        acc += phi[h - j] @ innovations[j]
                delta_trajectory[h] = acc

        active_fc = np.maximum(base_fc + delta_trajectory, 0.0)
        base_fc_clipped = np.maximum(base_fc, 0.0)

        suez_low_95 = np.maximum(active_fc[:, 0] - half_w95[:, 0], 0.0)
        suez_up_95 = np.maximum(active_fc[:, 0] + half_w95[:, 0], 0.0)
        suez_low_80 = np.maximum(active_fc[:, 0] - half_w80[:, 0], 0.0)
        suez_up_80 = np.maximum(active_fc[:, 0] + half_w80[:, 0], 0.0)

        cape_low_95 = np.maximum(active_fc[:, 1] - half_w95[:, 1], 0.0)
        cape_up_95 = np.maximum(active_fc[:, 1] + half_w95[:, 1], 0.0)
        cape_low_80 = np.maximum(active_fc[:, 1] - half_w80[:, 1], 0.0)
        cape_up_80 = np.maximum(active_fc[:, 1] + half_w80[:, 1], 0.0)

        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        total_ships = np.maximum(active_fc[:, 0] + active_fc[:, 1], 1e-6)
        cape_share_fc = (active_fc[:, 1] / total_ships) * 100.0

        return pd.DataFrame(
            {
                "date": future_dates,
                "step_day": np.arange(1, horizon + 1),
                "suez_baseline_fc": base_fc_clipped[:, 0],
                "cape_baseline_fc": base_fc_clipped[:, 1],
                "suez_forecast": active_fc[:, 0],
                "cape_forecast": active_fc[:, 1],
                "suez_lower_80": suez_low_80,
                "suez_upper_80": suez_up_80,
                "suez_lower_95": suez_low_95,
                "suez_upper_95": suez_up_95,
                "cape_lower_80": cape_low_80,
                "cape_upper_80": cape_up_80,
                "cape_lower_95": cape_low_95,
                "cape_upper_95": cape_up_95,
                "forecast_cape_share_pct": cape_share_fc,
                "is_shock_active": has_shock,
            }
        )

    def get_impulse_response_and_ccf(
        self, periods: int = 30, shock_ships: float = -10.0
    ) -> dict[str, pd.DataFrame]:
        """
        Packages both the empirical Cross-Correlation Function (-30 to +30 days)
        and the analytical VECM Generalized Impulse Response curve (0 to `periods` days).
        """
        periods = int(np.clip(periods, 5, 60))
        phi = self._compute_ma_matrices(periods)
        sigma_00 = max(float(self.sigma_u[0, 0]), 1e-6)
        girf_init = (self.sigma_u[:, 0] / sigma_00) * float(shock_ships)

        responses = np.array([phi[h] @ girf_init for h in range(periods + 1)])
        girf_df = pd.DataFrame(
            {
                "horizon_days": np.arange(0, periods + 1),
                "suez_impulse_ships": responses[:, 0],
                "cape_response_ships": responses[:, 1],
                "in_physical_reroute_window": [10 <= h <= 14 for h in range(periods + 1)],
            }
        )

        return {
            "ccf_df": self.ccf_df.copy(),
            "girf_df": girf_df,
        }

    def forecast_from_historical_origin(
        self, origin_date: str | pd.Timestamp, horizon: int = 30
    ) -> pd.DataFrame:
        """
        Generates a rolling historical VECM replay starting from `origin_date`
        alongside realized observations for backtest inspection.
        """
        target_dt = pd.to_datetime(origin_date)
        eligible = self.df.index[self.df["date"] <= target_dt]
        if len(eligible) < self.k_ar:
            raise ValueError(
                f"Origin date {target_dt.date()} requires at least {self.k_ar} prior observations."
            )

        end_idx = int(eligible[-1])
        init_window = (
            self.df.loc[end_idx - self.k_ar + 1 : end_idx, self.endog_names]
            .to_numpy(dtype=float)
        )
        preds = np.maximum(self._forecast_from_window(init_window, steps=horizon), 0.0)

        fc_dates = pd.date_range(
            start=self.df.loc[end_idx, "date"] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        replay_df = pd.DataFrame(
            {
                "date": fc_dates,
                "suez_replay_fc": preds[:, 0],
                "cape_replay_fc": preds[:, 1],
            }
        )
        actuals = self.df[["date", "suez_transit_count", "cape_transit_count"]]
        return pd.merge(replay_df, actuals, on="date", how="left")


# ==============================================================================
# SINGLETON ACCESSOR FOR PLOTLY DASH WORKERS
# ==============================================================================


@lru_cache(maxsize=1)
def get_engine() -> MaritimeVECMEngine:
    """Returns the process-cached MaritimeVECMEngine singleton."""
    return MaritimeVECMEngine()


# ==============================================================================
# CLI DIAGNOSTIC SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    print("Initializing MaritimeVECMEngine diagnostic suite...")
    engine = get_engine()

    # 1. Verify exact numerical parity between custom VAR recursion and statsmodels.predict()
    sm_fc = engine.vecm_results.predict(steps=14)
    var_fc = engine._forecast_from_window(engine.last_k_obs, steps=14)
    max_diff = float(np.max(np.abs(sm_fc - var_fc)))
    assert np.allclose(sm_fc, var_fc, atol=1e-9), f"Recursion drift detected: {max_diff}"

    # 2. Verify KPI snapshot
    kpi = engine.get_kpi_snapshot()
    print(f"\n[OK] Loaded {kpi['total_observations']} aligned daily observations (through {kpi['as_of_date']})")
    print(f"[OK] Level-VAR Recursion Parity Max Error: {max_diff:.2e} ships")
    print(
        f"[OK] Live Telemetry -> Suez 7d MA: {kpi['suez_7d_ma']:.1f} ships | "
        f"Cape 7d MA: {kpi['cape_7d_ma']:.1f} ships | "
        f"Cape Share: {kpi['cape_share_pct']:.1f}% | "
        f"Brent Crude: ${kpi['brent_latest']:.2f}/bbl"
    )
    print(
        f"[OK] Volatility Seismograph -> GVI: {kpi['volatility_index']:.2f} sigma "
        f"({kpi['regime_status']}) | Equilibrium ECT: {kpi['coint_ect']:+.2f} ships"
    )
    print(
        f"[OK] Cointegration Parameters -> alpha_suez={kpi['alpha_suez']:.4f}, "
        f"alpha_cape={kpi['alpha_cape']:.4f}, beta_cape={kpi['beta_cape']:.4f}"
    )

    # 3. Verify Baseline vs. Red Sea Shock Scenario Forecast
    base_30 = engine.generate_forecast(horizon=30)
    shock_30 = engine.generate_forecast(
        horizon=30, suez_shock_ships=-30.0, brent_shock_pct=15.0, shock_profile="sustained"
    )
    day14_suez_delta = shock_30.loc[13, "suez_forecast"] - base_30.loc[13, "suez_forecast"]
    day14_cape_delta = shock_30.loc[13, "cape_forecast"] - base_30.loc[13, "cape_forecast"]

    print(
        f"[OK] 30-Day Scenario Check (-30 Suez Shock, +15% Brent) at Day +14 -> "
        f"Suez Shift: {day14_suez_delta:+.2f} ships | Cape Shift: {day14_cape_delta:+.2f} ships"
    )
    print("\nStep 4 Inference Engine verified and ready for Step 5 (Plotly Dash UI).")