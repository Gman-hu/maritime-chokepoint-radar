"""
app.py
Interactive Plotly Dash Frontend — Modern Maritime Radar & VECM Forecasting Terminal
Exposes `server = app.server` for Render cloud deployment via Gunicorn.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from vecm_engine import SCENARIO_PRESETS, get_engine

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# INITIALIZE CACHED VECM ENGINE & PLOTLY DASH SERVER
# ==============================================================================

engine = get_engine()
INITIAL_KPI = engine.get_kpi_snapshot()

app = Dash(
    __name__,
    title="Maritime Chokepoint Radar | VECM Terminal",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Aesthetic Color Constants
CYAN = "#00F2FE"
CYAN_GLOW = "rgba(0, 242, 254, 0.16)"
AMBER = "#FF9900"
AMBER_GLOW = "rgba(255, 153, 0, 0.16)"
ROSE = "#FB7185"
CRIMSON = "#FF3B5C"
EMERALD = "#10B981"
GRID_COLOR = "rgba(148, 163, 184, 0.09)"
ZERO_LINE = "rgba(230, 241, 255, 0.28)"
FONT_MONO = "JetBrains Mono, Consolas, monospace"
FONT_SANS = "Space Grotesk, sans-serif"

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "responsive": True,
}


def base_chart_layout(
    height: int = 420, margin: dict[str, int] | None = None
) -> dict[str, Any]:
    """Standardized dark-ocean radar layout with explicit height for Dash 4."""
    return {
        "height": height,
        "autosize": True,
        "margin": margin or {"l": 52, "r": 24, "t": 46, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(5, 9, 18, 0.58)",
        "font": {"family": FONT_SANS, "color": "#CBD5E1", "size": 11},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#070D19",
            "bordercolor": "rgba(0, 242, 254, 0.45)",
            "font": {"family": FONT_MONO, "size": 11, "color": "#F8FAFC"},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0.0,
            "font": {"family": FONT_MONO, "size": 10, "color": "#94A3B8"},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }


# ==============================================================================
# FIGURE BUILDERS
# ==============================================================================


def build_corridor_map(
    kpi: dict[str, Any], fc_df: pd.DataFrame, horizon: int
) -> go.Figure:
    """
    Vector-based Asia-to-Europe maritime loop radar map (Malacca -> Rotterdam)
    bounded cleanly to the Eastern Hemisphere trade corridor.
    """
    end_row = fc_df.iloc[-1]
    suez_fc_end = float(end_row["suez_forecast"])
    cape_fc_end = float(end_row["cape_forecast"])
    cape_share_fc = float(end_row["forecast_cape_share_pct"])
    suez_share_fc = 100.0 - cape_share_fc

    suez_route_lats = [1.29, 6.0, 12.6, 20.2, 29.9, 31.3, 35.9, 36.0, 43.0, 49.5, 51.92]
    suez_route_lons = [103.85, 80.2, 43.3, 38.5, 32.5, 32.3, 14.5, -5.4, -9.3, -2.0, 4.48]

    cape_route_lats = [1.29, -8.0, -20.5, -31.0, -34.35, -23.0, -5.0, 14.7, 35.0, 48.5, 51.92]
    cape_route_lons = [103.85, 88.0, 62.0, 35.0, 18.47, 8.0, -6.0, -17.8, -11.0, -5.0, 4.48]

    suez_width = max(1.8, (suez_share_fc / 100.0) * 5.4)
    cape_width = max(1.8, (cape_share_fc / 100.0) * 5.4)

    fig = go.Figure()

    # Outer neon halos
    fig.add_trace(
        go.Scattergeo(
            lat=suez_route_lats,
            lon=suez_route_lons,
            mode="lines",
            line={"width": suez_width * 2.8, "color": "rgba(0, 242, 254, 0.18)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=cape_route_lats,
            lon=cape_route_lons,
            mode="lines",
            line={"width": cape_width * 2.8, "color": "rgba(255, 153, 0, 0.18)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Core shipping lanes
    fig.add_trace(
        go.Scattergeo(
            lat=suez_route_lats,
            lon=suez_route_lons,
            mode="lines",
            name=f"Suez Corridor ({suez_share_fc:.1f}%)",
            line={"width": suez_width, "color": CYAN},
            hovertemplate="<b>Suez — Red Sea Corridor</b><br>Projected Share: "
            f"{suez_share_fc:.1f}%<br>Baseline Distance: ~8,500 nm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=cape_route_lats,
            lon=cape_route_lons,
            mode="lines",
            name=f"Cape Detour (+10–14d | {cape_share_fc:.1f}%)",
            line={"width": cape_width, "color": AMBER},
            hovertemplate="<b>Cape of Good Hope Detour</b><br>Projected Share: "
            f"{cape_share_fc:.1f}%<br>Sailing Penalty: +10 to +14 Days (+3,500 nm)<extra></extra>",
        )
    )

    # Concentric radar pulse rings at Suez & Cape
    suez_size = float(np.clip(kpi["suez_7d_ma"] * 0.26, 12, 30))
    cape_size = float(np.clip(kpi["cape_7d_ma"] * 0.26, 12, 30))

    fig.add_trace(
        go.Scattergeo(
            lat=[30.58, -34.35],
            lon=[32.27, 18.47],
            mode="markers",
            marker={
                "size": [suez_size * 2.1, cape_size * 2.1],
                "color": ["rgba(0, 242, 254, 0.18)", "rgba(255, 153, 0, 0.18)"],
                "line": {"width": 1.2, "color": [CYAN, AMBER]},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Chokepoint & Terminal Anchor Nodes
    fig.add_trace(
        go.Scattergeo(
            lat=[30.58, -34.35, 12.58, 1.29, 51.92],
            lon=[32.27, 18.47, 43.33, 103.85, 4.48],
            mode="markers+text",
            text=[
                f"SUEZ ({kpi['suez_7d_ma']:.0f}/d)",
                f"CAPE ({kpi['cape_7d_ma']:.0f}/d)",
                "BAB-EL-MANDEB",
                "MALACCA",
                "ROTTERDAM",
            ],
            textposition=[
                "top right",
                "top right",
                "middle right",
                "top left",
                "bottom right",
            ],
            textfont={"family": FONT_MONO, "size": 9.5, "color": "#F1F5F9"},
            marker={
                "size": [suez_size, cape_size, 8, 7, 7],
                "color": [CYAN, AMBER, CRIMSON, "#CBD5E1", "#CBD5E1"],
                "line": {"width": 1.5, "color": "#05080F"},
            },
            customdata=[
                f"Live 7d MA: {kpi['suez_7d_ma']:.1f} ships/d<br>Day +{horizon} Forecast: {suez_fc_end:.1f} ships/d",
                f"Live 7d MA: {kpi['cape_7d_ma']:.1f} ships/d<br>Day +{horizon} Forecast: {cape_fc_end:.1f} ships/d",
                "Primary Geopolitical Risk Chokepoint (Red Sea Southern Gate)",
                "Asia Origin Hub — Singapore / Malacca Strait",
                "Europe Destination Hub — Port of Rotterdam",
            ],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_geos(
        projection_type="equirectangular",
        showland=True,
        landcolor="#0D1729",
        showocean=True,
        oceancolor="#050913",
        showcoastlines=True,
        coastlinecolor="#23436E",
        coastlinewidth=0.9,
        showcountries=True,
        countrycolor="#162944",
        countrywidth=0.5,
        showlakes=False,
        showframe=True,
        framecolor="rgba(0, 242, 254, 0.18)",
        framewidth=1,
        bgcolor="rgba(0,0,0,0)",
        lataxis={
            "range": [-42, 57],
            "showgrid": True,
            "gridcolor": "rgba(0, 242, 254, 0.05)",
            "dtick": 15,
        },
        lonaxis={
            "range": [-16, 112],
            "showgrid": True,
            "gridcolor": "rgba(0, 242, 254, 0.05)",
            "dtick": 20,
        },
    )

    layout = base_chart_layout(height=440, margin={"l": 6, "r": 6, "t": 38, "b": 8})
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def build_trajectory_figure(
    hist_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    breaks: dict[str, Any],
    show_breaks: bool,
    show_cones: bool,
    show_raw: bool,
) -> go.Figure:
    """
    Main Dual-Trace Time Series & Dynamic Multi-Step VECM Forecast Trajectory
    with staggered break badges and single-row legend spacing.
    """
    fig = go.Figure()
    has_shock = bool(fc_df["is_shock_active"].iloc[0])

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["suez_transit_count"],
                mode="lines",
                line={"color": "rgba(0, 242, 254, 0.20)", "width": 0.9},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["cape_transit_count"],
                mode="lines",
                line={"color": "rgba(255, 153, 0, 0.20)", "width": 0.9},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Neon glow underlays
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["suez_7d_ma"],
            mode="lines",
            line={"color": CYAN_GLOW, "width": 6},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["cape_7d_ma"],
            mode="lines",
            line={"color": AMBER_GLOW, "width": 6},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Core historical 7d MA traces
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["suez_7d_ma"],
            mode="lines",
            name="Suez (7d MA)",
            line={"color": CYAN, "width": 2.2},
            hovertemplate="Suez 7d MA: <b>%{y:.1f}</b> ships<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["cape_7d_ma"],
            mode="lines",
            name="Cape (7d MA)",
            line={"color": AMBER, "width": 2.2},
            hovertemplate="Cape 7d MA: <b>%{y:.1f}</b> ships<extra></extra>",
        )
    )

    # 95% Lütkepohl confidence cones (kept out of legend to prevent 2-line wrapping)
    if show_cones:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fc_df["date"], fc_df["date"].iloc[::-1]]),
                y=pd.concat([fc_df["suez_upper_95"], fc_df["suez_lower_95"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(0, 242, 254, 0.12)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fc_df["date"], fc_df["date"].iloc[::-1]]),
                y=pd.concat([fc_df["cape_upper_95"], fc_df["cape_lower_95"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(255, 153, 0, 0.12)",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    last_dt = hist_df["date"].iloc[-1]
    fc_x = pd.concat([pd.Series([last_dt]), fc_df["date"]], ignore_index=True)
    suez_fc_y = np.concatenate(
        [[hist_df["suez_transit_count"].iloc[-1]], fc_df["suez_forecast"].to_numpy()]
    )
    cape_fc_y = np.concatenate(
        [[hist_df["cape_transit_count"].iloc[-1]], fc_df["cape_forecast"].to_numpy()]
    )

    if has_shock:
        fig.add_trace(
            go.Scatter(
                x=fc_df["date"],
                y=fc_df["suez_baseline_fc"],
                mode="lines",
                name="Suez Baseline",
                line={"color": "rgba(0, 242, 254, 0.42)", "width": 1.3, "dash": "dot"},
                hovertemplate="Suez Baseline: %{y:.1f} ships<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["date"],
                y=fc_df["cape_baseline_fc"],
                mode="lines",
                name="Cape Baseline",
                line={"color": "rgba(255, 153, 0, 0.42)", "width": 1.3, "dash": "dot"},
                hovertemplate="Cape Baseline: %{y:.1f} ships<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=suez_fc_y,
            mode="lines",
            name="Suez Forecast",
            line={"color": "#7CF9FF", "width": 2.6, "dash": "dash"},
            hovertemplate="Suez Forecast: <b>%{y:.1f}</b> ships<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=cape_fc_y,
            mode="lines",
            name="Cape Forecast",
            line={"color": "#FFC266", "width": 2.6, "dash": "dash"},
            hovertemplate="Cape Forecast: <b>%{y:.1f}</b> ships<extra></extra>",
        )
    )

    shapes: list[dict[str, Any]] = [
        {
            "type": "line",
            "x0": last_dt.strftime("%Y-%m-%d"),
            "x1": last_dt.strftime("%Y-%m-%d"),
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "rgba(255,255,255,0.48)", "width": 1.3, "dash": "dot"},
        }
    ]
    annotations: list[dict[str, Any]] = [
        {
            "x": last_dt.strftime("%Y-%m-%d"),
            "y": 0.96,
            "yref": "paper",
            "text": "FORECAST ORIGIN",
            "showarrow": False,
            "xanchor": "right",
            "xshift": -6,
            "font": {"family": FONT_MONO, "size": 9, "color": "#E2E8F0"},
            "bgcolor": "rgba(5, 8, 15, 0.82)",
            "borderpad": 2,
        }
    ]

    if show_breaks:
        min_dt = hist_df["date"].min()
        for bp_str in breaks.get("suez_structural_breaks", []):
            if pd.to_datetime(bp_str) >= min_dt:
                shapes.append(
                    {
                        "type": "line",
                        "x0": bp_str,
                        "x1": bp_str,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {
                            "color": "rgba(255, 59, 92, 0.30)",
                            "width": 1.0,
                            "dash": "dash",
                        },
                    }
                )
        # Stagger crisis labels vertically so they never collide with FORECAST ORIGIN
        y_slots = [0.95, 0.84]
        visible_events = [
            ev
            for ev in breaks.get("annotated_events", [])
            if pd.to_datetime(ev["date"]) >= min_dt
        ]
        for idx, ev in enumerate(visible_events):
            shapes.append(
                {
                    "type": "line",
                    "x0": ev["date"],
                    "x1": ev["date"],
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": CRIMSON, "width": 1.5, "dash": "dashdot"},
                }
            )
            annotations.append(
                {
                    "x": ev["date"],
                    "y": y_slots[idx % len(y_slots)],
                    "yref": "paper",
                    "text": ev["label"].upper(),
                    "showarrow": False,
                    "xanchor": "left",
                    "xshift": 5,
                    "font": {"family": FONT_MONO, "size": 8.5, "color": CRIMSON},
                    "bgcolor": "rgba(5, 8, 15, 0.84)",
                    "borderpad": 2,
                }
            )

    layout = base_chart_layout(height=440, margin={"l": 52, "r": 20, "t": 44, "b": 38})
    layout.update(
        {
            "shapes": shapes,
            "annotations": annotations,
            "xaxis": {
                "gridcolor": GRID_COLOR,
                "tickfont": {"family": FONT_MONO, "size": 10},
                "zeroline": False,
            },
            "yaxis": {
                "title": {
                    "text": "Daily Vessels (Ships)",
                    "font": {"family": FONT_MONO, "size": 10},
                },
                "gridcolor": GRID_COLOR,
                "tickfont": {"family": FONT_MONO, "size": 10},
                "zeroline": False,
            },
        }
    )
    fig.update_layout(**layout)
    return fig


def build_ccf_girf_figure(
    ccf_df: pd.DataFrame, girf_df: pd.DataFrame, shock_ships: float
) -> go.Figure:
    """
    Two-tier Impulse Physics Panel with separated legend and subplot headers.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.25,
        subplot_titles=(
            "EMPIRICAL CCF (DIFFERENCED SERIES — +10 TO +14D SAILING LAG WINDOW)",
            f"ANALYTICAL VECM GIRF ({shock_ships:+.0f} SUEZ VESSEL SHOCK PROPAGATION)",
        ),
    )

    bar_colors = [
        AMBER if bool(in_win) else "rgba(0, 242, 254, 0.58)"
        for in_win in ccf_df["in_physical_reroute_window"]
    ]

    fig.add_trace(
        go.Bar(
            x=ccf_df["lag_days"],
            y=ccf_df["ccf_weekly_diff"],
            marker={"color": bar_colors, "line": {"width": 0}},
            name="7d Diff CCF",
            hovertemplate="Lag %{x:+d} Days<br>CCF Correlation: <b>%{y:.4f}</b><extra></extra>",
        ),
        row=1,
        col=1,
    )

    ci_up = float(ccf_df["conf_upper_95"].iloc[0])
    fig.add_hline(
        y=ci_up,
        line={"color": "rgba(251,113,133,0.5)", "width": 1, "dash": "dot"},
        row=1,
        col=1,
    )
    fig.add_hline(
        y=-ci_up,
        line={"color": "rgba(251,113,133,0.5)", "width": 1, "dash": "dot"},
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=9.5,
        x1=14.5,
        fillcolor="rgba(255, 153, 0, 0.14)",
        line={"color": "rgba(255, 153, 0, 0.48)", "width": 1},
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=girf_df["horizon_days"],
            y=girf_df["suez_impulse_ships"],
            mode="lines",
            name="Suez GIRF (Ships)",
            line={"color": CYAN, "width": 2.2},
            hovertemplate="Day +%{x}: Suez <b>%{y:+.2f}</b> ships<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=girf_df["horizon_days"],
            y=girf_df["cape_response_ships"],
            mode="lines",
            name="Cape GIRF Response (Ships)",
            line={"color": AMBER, "width": 2.2},
            hovertemplate="Day +%{x}: Cape <b>%{y:+.2f}</b> ships<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_vrect(
        x0=10,
        x1=14,
        fillcolor="rgba(255, 153, 0, 0.12)",
        line={"color": "rgba(255, 153, 0, 0.38)", "width": 1},
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": ZERO_LINE, "width": 1}, row=2, col=1)

    layout = base_chart_layout(height=410, margin={"l": 52, "r": 20, "t": 68, "b": 40})
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.14,
        "xanchor": "left",
        "x": 0.0,
        "font": {"family": FONT_MONO, "size": 10, "color": "#94A3B8"},
        "bgcolor": "rgba(0,0,0,0)",
    }
    fig.update_layout(**layout)

    for annotation in fig.layout.annotations:
        annotation.font = {"family": FONT_MONO, "size": 9.5, "color": "#CBD5E1"}

    fig.update_xaxes(
        title={
            "text": "Lag k (Days | +k = Suez Leads Cape)",
            "font": {"family": FONT_MONO, "size": 9.5},
        },
        gridcolor=GRID_COLOR,
        tickfont={"family": FONT_MONO, "size": 9.5},
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title={
            "text": "Horizon Days After Shock",
            "font": {"family": FONT_MONO, "size": 9.5},
        },
        gridcolor=GRID_COLOR,
        tickfont={"family": FONT_MONO, "size": 9.5},
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title={"text": "Corr (r)", "font": {"family": FONT_MONO, "size": 9.5}},
        gridcolor=GRID_COLOR,
        tickfont={"family": FONT_MONO, "size": 9.5},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title={"text": "Δ Ships", "font": {"family": FONT_MONO, "size": 9.5}},
        gridcolor=GRID_COLOR,
        tickfont={"family": FONT_MONO, "size": 9.5},
        row=2,
        col=1,
    )
    return fig


def build_fuel_substitution_figure(hist_df: pd.DataFrame) -> go.Figure:
    """
    Dual-Axis Macroeconomic Context Tracker:
    Forward-filled Brent Crude bunker fuel proxy ($/bbl) vs. Cape Diversion Share (%).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    cape_share_7d = hist_df["cape_share_pct"].rolling(window=7, min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=cape_share_7d,
            mode="lines",
            name="Cape Corridor Share (7d %)",
            fill="tozeroy",
            fillcolor="rgba(255, 153, 0, 0.10)",
            line={"color": AMBER, "width": 2.0},
            hovertemplate="Cape Share: <b>%{y:.1f}%</b><extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["brent_fuel_price"],
            mode="lines",
            name="Brent Crude Proxy (BZ=F $/bbl)",
            line={"color": ROSE, "width": 1.8},
            hovertemplate="Brent Crude: <b>$%{y:.2f}/bbl</b><extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=50.0,
        line={"color": "rgba(0, 242, 254, 0.35)", "width": 1.1, "dash": "dot"},
        secondary_y=False,
    )

    layout = base_chart_layout(height=410, margin={"l": 52, "r": 56, "t": 44, "b": 40})
    fig.update_layout(**layout)

    fig.update_xaxes(gridcolor=GRID_COLOR, tickfont={"family": FONT_MONO, "size": 10})
    fig.update_yaxes(
        title={
            "text": "Cape Transit Share (%)",
            "font": {"family": FONT_MONO, "size": 10, "color": AMBER},
        },
        range=[0, 100],
        gridcolor=GRID_COLOR,
        tickfont={"family": FONT_MONO, "size": 10, "color": AMBER},
        secondary_y=False,
    )
    fig.update_yaxes(
        title={
            "text": "Brent Crude ($ / bbl)",
            "font": {"family": FONT_MONO, "size": 10, "color": ROSE},
        },
        gridcolor="rgba(0,0,0,0)",
        tickfont={"family": FONT_MONO, "size": 10, "color": ROSE},
        secondary_y=True,
    )
    return fig


def build_seismograph_figure(
    hist_df: pd.DataFrame, breaks: dict[str, Any], show_breaks: bool
) -> go.Figure:
    """
    Bottom-anchored EKG Volatility Seismograph isolating pure shock-driven VECM residuals.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["suez_residual_error"],
            mode="lines",
            name="Suez VECM Residual (Ships)",
            line={"color": CYAN, "width": 1.35},
            hovertemplate="Suez Shock Error: <b>%{y:+.2f}</b> ships<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["cape_residual_error"],
            mode="lines",
            name="Cape VECM Residual (Ships)",
            line={"color": AMBER, "width": 1.35},
            opacity=0.88,
            hovertemplate="Cape Shock Error: <b>%{y:+.2f}</b> ships<extra></extra>",
        )
    )

    suez_2sigma = 2.25 * float(np.sqrt(engine.sigma_u[0, 0]))
    fig.add_hline(y=0.0, line={"color": ZERO_LINE, "width": 1.2})
    fig.add_hline(
        y=suez_2sigma,
        line={"color": "rgba(255, 59, 92, 0.45)", "width": 1.0, "dash": "dot"},
    )
    fig.add_hline(
        y=-suez_2sigma,
        line={"color": "rgba(255, 59, 92, 0.45)", "width": 1.0, "dash": "dot"},
    )

    shapes: list[dict[str, Any]] = []
    if show_breaks:
        min_dt = hist_df["date"].min()
        for ev in breaks.get("annotated_events", []):
            if pd.to_datetime(ev["date"]) >= min_dt:
                shapes.append(
                    {
                        "type": "line",
                        "x0": ev["date"],
                        "x1": ev["date"],
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": CRIMSON, "width": 1.3, "dash": "dashdot"},
                    }
                )

    layout = base_chart_layout(height=305, margin={"l": 52, "r": 20, "t": 38, "b": 36})
    layout.update(
        {
            "shapes": shapes,
            "xaxis": {
                "gridcolor": GRID_COLOR,
                "tickfont": {"family": FONT_MONO, "size": 10},
            },
            "yaxis": {
                "title": {
                    "text": "Residual Error (Ships)",
                    "font": {"family": FONT_MONO, "size": 10},
                },
                "gridcolor": GRID_COLOR,
                "tickfont": {"family": FONT_MONO, "size": 10},
            },
        }
    )
    fig.update_layout(**layout)
    return fig


def build_benchmark_figure(kpi: dict[str, Any]) -> go.Figure:
    """
    Phase 4 5-Fold Expanding Window RMSE Comparison with locked axis & external value labels.
    """
    bm = kpi["benchmarks"]
    models = [
        "Tier 3: VECM + TCN<br>(Data Starvation)",
        "Tier 2: VECM + XGB<br>(Tail Damping)",
        "Tier 1: Standalone VECM<br>(Production Winner)",
    ]
    rmse_vals = [
        bm["tcn_deep_rmse"],
        bm["vecm_xgb_hybrid_rmse"],
        bm["vecm_standalone_rmse"],
    ]
    mae_vals = [
        bm["tcn_deep_mae"],
        bm["vecm_xgb_hybrid_mae"],
        bm["vecm_standalone_mae"],
    ]
    colors = ["rgba(251, 113, 133, 0.78)", "rgba(255, 153, 0, 0.82)", CYAN]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=models,
            x=rmse_vals,
            orientation="h",
            marker={"color": colors, "line": {"width": 1, "color": "#05080F"}},
            text=[f"{v:.4f}" for v in rmse_vals],
            textposition="outside",
            cliponaxis=False,
            textfont={"family": FONT_MONO, "size": 10.5, "color": "#F8FAFC"},
            customdata=mae_vals,
            hovertemplate="<b>%{y}</b><br>Out-of-Fold RMSE: <b>%{x:.4f}</b> ships"
            "<br>Out-of-Fold MAE: <b>%{customdata:.4f}</b> ships<extra></extra>",
            showlegend=False,
        )
    )

    layout = base_chart_layout(height=215, margin={"l": 152, "r": 54, "t": 12, "b": 34})
    layout["hovermode"] = "closest"
    layout.update(
        {
            "xaxis": {
                "title": {
                    "text": "5-Fold Expanding Window RMSE (Ships — Lower is Better)",
                    "font": {"family": FONT_MONO, "size": 9.5},
                },
                "range": [0, 10.8],
                "fixedrange": True,
                "gridcolor": GRID_COLOR,
                "tickfont": {"family": FONT_MONO, "size": 9.5},
            },
            "yaxis": {
                "fixedrange": True,
                "tickfont": {"family": FONT_MONO, "size": 9.5, "color": "#E2E8F0"},
            },
        }
    )
    fig.update_layout(**layout)
    return fig


# ==============================================================================
# COMPUTE INITIAL FIGURES FOR ZERO-FLASH FIRST RENDER
# ==============================================================================


def _build_initial_bundle() -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str]:
    init_start = engine.last_date - pd.Timedelta(days=365 * 3)
    hist_df = engine.get_historical_slice(start_date=init_start)
    breaks = engine.get_structural_breaks()
    fc_df = engine.generate_forecast(horizon=30)
    imp = engine.get_impulse_response_and_ccf(periods=30, shock_ships=-20.0)
    end_fc = fc_df.iloc[-1]
    banner = (
        f"UNPERTURBED VECM // DAY +30 EQUILIBRIUM: "
        f"SUEZ {end_fc['suez_forecast']:.1f} SHIPS/D | "
        f"CAPE {end_fc['cape_forecast']:.1f} SHIPS/D | "
        f"CAPE SHARE {end_fc['forecast_cape_share_pct']:.1f}%"
    )
    return (
        build_corridor_map(INITIAL_KPI, fc_df, 30),
        build_trajectory_figure(hist_df, fc_df, breaks, True, True, False),
        build_ccf_girf_figure(imp["ccf_df"], imp["girf_df"], -20.0),
        build_fuel_substitution_figure(hist_df),
        build_seismograph_figure(hist_df, breaks, True),
        banner,
    )


INIT_MAP, INIT_TRAJ, INIT_CCF, INIT_FUEL, INIT_SEIS, INIT_BANNER = _build_initial_bundle()


# ==============================================================================
# APPLICATION DOM LAYOUT
# ==============================================================================


def build_kpi_ribbon(kpi: dict[str, Any]) -> list[html.Div]:
    gvi_color = "emerald" if kpi["regime_status"] == "NOMINAL EQUILIBRIUM" else "amber"
    return [
        html.Div(
            className="kpi-card",
            children=[
                html.Div("01 // SUEZ CANAL (7D MA)", className="kpi-label"),
                html.Div(f"{kpi['suez_7d_ma']:.1f} ships", className="kpi-value cyan"),
                html.Div(
                    [
                        html.Span(f"Last: {kpi['suez_latest']:.0f}/d"),
                        html.Span(f"{kpi['suez_7d_pct_change']:+.1f}% (7d)"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
        html.Div(
            className="kpi-card amber",
            children=[
                html.Div("02 // CAPE OF GOOD HOPE (7D MA)", className="kpi-label"),
                html.Div(f"{kpi['cape_7d_ma']:.1f} ships", className="kpi-value amber"),
                html.Div(
                    [
                        html.Span(f"Last: {kpi['cape_latest']:.0f}/d"),
                        html.Span(f"{kpi['cape_7d_pct_change']:+.1f}% (7d)"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
        html.Div(
            className="kpi-card amber",
            children=[
                html.Div("03 // CAPE DIVERSION SHARE", className="kpi-label"),
                html.Div(f"{kpi['cape_share_pct']:.1f}%", className="kpi-value"),
                html.Div(
                    [
                        html.Span("Coint. ECT:"),
                        html.Span(f"{kpi['coint_ect']:+.2f} ships"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
        html.Div(
            className="kpi-card rose",
            children=[
                html.Div("04 // BRENT BUNKER PROXY (BZ=F)", className="kpi-label"),
                html.Div(f"${kpi['brent_latest']:.2f}/bbl", className="kpi-value rose"),
                html.Div(
                    [
                        html.Span(f"Δ7d: {kpi['brent_7d_delta']:+.2f}$"),
                        html.Span(f"{kpi['brent_7d_pct_change']:+.1f}%"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
        html.Div(
            className=f"kpi-card {gvi_color}",
            children=[
                html.Div("05 // VOLATILITY INDEX (GVI)", className="kpi-label"),
                html.Div(
                    f"{kpi['volatility_index']:.2f} σ",
                    className=f"kpi-value {gvi_color}",
                ),
                html.Div(
                    [
                        html.Span(kpi["regime_status"]),
                        html.Span(f"7d: {kpi['volatility_7d_ma']:.2f}σ"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
        html.Div(
            className="kpi-card emerald",
            children=[
                html.Div("06 // PRODUCTION VECM RMSE", className="kpi-label"),
                html.Div(
                    f"{kpi['benchmarks']['vecm_standalone_rmse']:.4f}",
                    className="kpi-value emerald",
                ),
                html.Div(
                    [
                        html.Span(f"XGB: {kpi['benchmarks']['vecm_xgb_hybrid_rmse']:.2f}"),
                        html.Span(f"TCN: {kpi['benchmarks']['tcn_deep_rmse']:.2f}"),
                    ],
                    className="kpi-sub",
                ),
            ],
        ),
    ]


pulse_cls = (
    "pulse-dot"
    if INITIAL_KPI["regime_status"] == "NOMINAL EQUILIBRIUM"
    else (
        "pulse-dot warn"
        if INITIAL_KPI["regime_status"] == "ELEVATED STRESS"
        else "pulse-dot critical"
    )
)

beta_sign = "+" if INITIAL_KPI["beta_cape"] >= 0 else "-"
coint_sign = "+" if engine.c_coint >= 0 else "-"

app.layout = html.Div(
    className="radar-shell",
    children=[
        # 1. TOP COMMAND HEADER
        html.Header(
            className="command-header",
            children=[
                html.Div(
                    [
                        html.Div(
                            "PHASE 5 PRODUCTION TERMINAL // SPATIAL COINTEGRATION & IMPULSE RADAR",
                            className="header-eyebrow",
                        ),
                        html.H1(
                            "MARITIME CHOKEPOINT TRANSIT RADAR",
                            className="header-title",
                        ),
                        html.P(
                            "Real-time econometric monitoring & Generalized Impulse Response (GIRF) forecasting "
                            "across the Suez Canal and Cape of Good Hope, powered by IMF PortWatch daily vessel "
                            "transits and Brent Crude (BZ=F) bunker fuel economics.",
                            className="header-subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="header-telemetry-badges",
                    children=[
                        html.Div(
                            [
                                html.Span(className=pulse_cls),
                                html.Span(f"STATE: {INITIAL_KPI['regime_status']}"),
                            ],
                            className="telemetry-pill",
                        ),
                        html.Div(
                            f"ENGINE: VECM(k={INITIAL_KPI['k_ar_diff']}, r={INITIAL_KPI['coint_rank']})",
                            className="telemetry-pill",
                        ),
                        html.Div(
                            f"OBS: {INITIAL_KPI['total_observations']:,} DAYS (THRU {INITIAL_KPI['as_of_date']})",
                            className="telemetry-pill",
                        ),
                    ],
                ),
            ],
        ),
        # 2. KPI SCORECARD RIBBON
        html.Section(build_kpi_ribbon(INITIAL_KPI), className="kpi-grid"),
        # 3. INTERACTIVE MISSION CONTROL & SCENARIO SHOCK DECK
        html.Section(
            className="control-deck",
            children=[
                html.Div(
                    className="control-top-row",
                    children=[
                        html.Div(
                            [
                                html.Span(
                                    "GEOPOLITICAL & BUNKER SCENARIO PRESETS",
                                    className="control-group-label",
                                ),
                                dcc.RadioItems(
                                    id="scenario-preset",
                                    options=[
                                        {"label": v["label"], "value": k}
                                        for k, v in SCENARIO_PRESETS.items()
                                    ],
                                    value="baseline",
                                    className="tactical-radio",
                                ),
                            ]
                        ),
                        html.Div(
                            INIT_BANNER,
                            id="live-shock-readout",
                            className="shock-readout-banner",
                        ),
                    ],
                ),
                html.Div(
                    className="control-sliders-grid",
                    children=[
                        html.Div(
                            className="slider-box",
                            children=[
                                html.Span(
                                    "FORECAST HORIZON (DAYS)",
                                    className="control-group-label",
                                ),
                                dcc.Slider(
                                    id="horizon-slider",
                                    min=14,
                                    max=90,
                                    step=1,
                                    value=30,
                                    marks={14: "14d", 30: "30d", 60: "60d", 90: "90d"},
                                    tooltip={"placement": "top", "always_visible": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="slider-box",
                            children=[
                                html.Span(
                                    "SUEZ VESSEL SHOCK (Δ SHIPS/DAY)",
                                    className="control-group-label",
                                ),
                                dcc.Slider(
                                    id="suez-shock-slider",
                                    min=-50,
                                    max=40,
                                    step=5,
                                    value=0,
                                    marks={
                                        -50: "-50",
                                        -25: "-25",
                                        0: "0",
                                        25: "+25",
                                        40: "+40",
                                    },
                                    tooltip={"placement": "top", "always_visible": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="slider-box",
                            children=[
                                html.Span(
                                    "BRENT BUNKER SHOCK (Δ %)",
                                    className="control-group-label",
                                ),
                                dcc.Slider(
                                    id="brent-shock-slider",
                                    min=-30,
                                    max=50,
                                    step=5,
                                    value=0,
                                    marks={-30: "-30%", 0: "0%", 25: "+25%", 50: "+50%"},
                                    tooltip={"placement": "top", "always_visible": False},
                                ),
                            ],
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "SHOCK DECAY PROFILE",
                                    className="control-group-label",
                                ),
                                dcc.RadioItems(
                                    id="shock-profile",
                                    options=[
                                        {"label": "Sustained", "value": "sustained"},
                                        {"label": "7d Blockage", "value": "transient_7d"},
                                        {"label": "1d Impulse", "value": "impulse"},
                                    ],
                                    value="sustained",
                                    className="tactical-radio",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "HISTORICAL WINDOW & OVERLAYS",
                                    className="control-group-label",
                                ),
                                dcc.RadioItems(
                                    id="lookback-window",
                                    options=[
                                        {"label": "1Y", "value": "1Y"},
                                        {"label": "3Y", "value": "3Y"},
                                        {"label": "2023+", "value": "REDSEA"},
                                        {"label": "All (2019+)", "value": "ALL"},
                                    ],
                                    value="3Y",
                                    className="tactical-radio",
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Checklist(
                                    id="overlay-toggles",
                                    options=[
                                        {"label": "Pelt Breaks", "value": "breaks"},
                                        {"label": "95% CI Cones", "value": "cones"},
                                        {"label": "Raw Daily", "value": "raw"},
                                    ],
                                    value=["breaks", "cones"],
                                    className="tactical-radio",
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        # 4. HERO ANALYTICAL ROW: VECTOR RADAR MAP + COINTEGRATED FORECAST TRAJECTORIES
        html.Section(
            className="hero-grid",
            children=[
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "ASIA–EUROPE MARITIME LOOP RADAR",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "MALACCA → SUEZ / CAPE → ROTTERDAM",
                                    className="panel-tag",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="corridor-map-graph",
                            figure=INIT_MAP,
                            config=PLOT_CONFIG,
                            style={"height": "440px", "width": "100%"},
                        ),
                    ],
                ),
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "DUAL-ROUTE TRANSIT FLOWS & VECM FORECAST HORIZON",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "COINTEGRATED I(1) TRAJECTORIES + 95% CONES",
                                    className="panel-tag amber",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="trajectory-graph",
                            figure=INIT_TRAJ,
                            config=PLOT_CONFIG,
                            style={"height": "440px", "width": "100%"},
                        ),
                    ],
                ),
            ],
        ),
        # 5. SECONDARY ANALYTICAL ROW: CCF / GIRF IMPULSE PHYSICS + BUNKER ECONOMICS
        html.Section(
            className="mid-grid",
            children=[
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "IMPULSE RESPONSE & CROSS-CORRELATION (CCF) LAG VERIFIER",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "+10 TO +14 DAY PHYSICAL SAILING DELAY",
                                    className="panel-tag",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="ccf-girf-graph",
                            figure=INIT_CCF,
                            config=PLOT_CONFIG,
                            style={"height": "410px", "width": "100%"},
                        ),
                    ],
                ),
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "EXOGENOUS BUNKER FUEL (BZ=F) VS. CAPE REROUTING SHARE",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "ECONOMIC DIVERSION THRESHOLD",
                                    className="panel-tag rose",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="fuel-substitution-graph",
                            figure=INIT_FUEL,
                            config=PLOT_CONFIG,
                            style={"height": "410px", "width": "100%"},
                        ),
                    ],
                ),
            ],
        ),
        # 6. BOTTOM ROW: VOLATILITY SEISMOGRAPH + PHASE 4 BENCHMARK FALSIFICATION
        html.Section(
            className="bottom-grid",
            children=[
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "THE VOLATILITY SEISMOGRAPH (LIVE VECM RESIDUAL ERRORS)",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "DETRENDED GEOPOLITICAL SHOCK ISOLATION (±2.25σ)",
                                    className="panel-tag",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="seismograph-graph",
                            figure=INIT_SEIS,
                            config=PLOT_CONFIG,
                            style={"height": "305px", "width": "100%"},
                        ),
                    ],
                ),
                html.Div(
                    className="radar-panel",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "MODEL ARCHITECTURE VERDICT (PHASE 4)",
                                    className="panel-title",
                                ),
                                html.Span(
                                    "PRINCIPLE OF PARSIMONY",
                                    className="panel-tag amber",
                                ),
                            ],
                            className="panel-header",
                        ),
                        dcc.Graph(
                            id="benchmark-graph",
                            figure=build_benchmark_figure(INITIAL_KPI),
                            config=PLOT_CONFIG,
                            style={"height": "215px", "width": "100%"},
                        ),
                        html.Div(
                            className="coint-equation-box",
                            children=[
                                html.Div(
                                    f"Cointegrating Vector: ECT_t = Suez_t {beta_sign} "
                                    f"{abs(INITIAL_KPI['beta_cape']):.4f}·Cape_t {coint_sign} "
                                    f"{abs(engine.c_coint):.2f}"
                                ),
                                html.Div(
                                    f"Adjustment Speeds: α_Suez = {INITIAL_KPI['alpha_suez']:+.4f} | "
                                    f"α_Cape = {INITIAL_KPI['alpha_cape']:+.4f} | AIC Lag k = {INITIAL_KPI['k_ar_diff']}d"
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # 7. FOOTER
        html.Footer(
            className="terminal-footer",
            children=[
                html.Span(
                    "DATA SOURCES: IMF PORTWATCH ARCGIS API (DAILY TRANSITS) & YAHOO FINANCE (BZ=F BRENT CRUDE)"
                ),
                html.Span(
                    "PRODUCTION ENGINE: STATSMODELS VECM (COINT_RANK=1, K_AR_DIFF=15) // PLOTLY DASH"
                ),
            ],
        ),
    ],
)


# ==============================================================================
# STATELESS REACTIVE CALLBACKS
# ==============================================================================


@app.callback(
    Output("suez-shock-slider", "value"),
    Output("brent-shock-slider", "value"),
    Output("shock-profile", "value"),
    Input("scenario-preset", "value"),
    prevent_initial_call=True,
)
def sync_sliders_to_preset(preset_key: str) -> tuple[float, float, str]:
    preset = SCENARIO_PRESETS.get(preset_key, SCENARIO_PRESETS["baseline"])
    return (
        float(preset["suez_shock_ships"]),
        float(preset["brent_shock_pct"]),
        str(preset["shock_profile"]),
    )


@app.callback(
    Output("corridor-map-graph", "figure"),
    Output("trajectory-graph", "figure"),
    Output("ccf-girf-graph", "figure"),
    Output("fuel-substitution-graph", "figure"),
    Output("seismograph-graph", "figure"),
    Output("live-shock-readout", "children"),
    Output("live-shock-readout", "className"),
    Input("horizon-slider", "value"),
    Input("suez-shock-slider", "value"),
    Input("brent-shock-slider", "value"),
    Input("shock-profile", "value"),
    Input("lookback-window", "value"),
    Input("overlay-toggles", "value"),
    prevent_initial_call=True,
)
def update_radar_terminal(
    horizon: int,
    suez_shock: float,
    brent_shock: float,
    shock_profile: str,
    lookback: str,
    overlays: list[str],
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str, str]:
    horizon = int(horizon or 30)
    suez_shock = float(suez_shock or 0.0)
    brent_shock = float(brent_shock or 0.0)
    overlays = overlays or []

    end_dt = engine.last_date
    if lookback == "1Y":
        start_dt = end_dt - pd.Timedelta(days=365)
    elif lookback == "3Y":
        start_dt = end_dt - pd.Timedelta(days=365 * 3)
    elif lookback == "REDSEA":
        start_dt = pd.Timestamp("2023-09-01")
    else:
        start_dt = None

    hist_df = engine.get_historical_slice(start_date=start_dt)
    breaks = engine.get_structural_breaks()
    kpi = engine.get_kpi_snapshot()

    fc_df = engine.generate_forecast(
        horizon=horizon,
        suez_shock_ships=suez_shock,
        brent_shock_pct=brent_shock,
        shock_profile=shock_profile,
    )
    girf_reference = suez_shock if abs(suez_shock) > 1e-6 else -20.0
    impulse_bundle = engine.get_impulse_response_and_ccf(
        periods=max(horizon, 30), shock_ships=girf_reference
    )

    show_breaks = "breaks" in overlays
    show_cones = "cones" in overlays
    show_raw = "raw" in overlays

    map_fig = build_corridor_map(kpi, fc_df, horizon)
    traj_fig = build_trajectory_figure(
        hist_df, fc_df, breaks, show_breaks, show_cones, show_raw
    )
    ccf_fig = build_ccf_girf_figure(
        impulse_bundle["ccf_df"], impulse_bundle["girf_df"], girf_reference
    )
    fuel_fig = build_fuel_substitution_figure(hist_df)
    seis_fig = build_seismograph_figure(hist_df, breaks, show_breaks)

    end_fc = fc_df.iloc[-1]
    suez_delta = float(end_fc["suez_forecast"] - end_fc["suez_baseline_fc"])
    cape_delta = float(end_fc["cape_forecast"] - end_fc["cape_baseline_fc"])

    if bool(end_fc["is_shock_active"]):
        banner_text = (
            f"SHOCK ACTIVE // DAY +{horizon} PROJECTION: "
            f"SUEZ {end_fc['suez_forecast']:.1f}/d ({suez_delta:+.1f}) | "
            f"CAPE {end_fc['cape_forecast']:.1f}/d ({cape_delta:+.1f}) | "
            f"CAPE SHARE {end_fc['forecast_cape_share_pct']:.1f}%"
        )
        banner_cls = "shock-readout-banner active"
    else:
        banner_text = (
            f"UNPERTURBED VECM // DAY +{horizon} EQUILIBRIUM: "
            f"SUEZ {end_fc['suez_forecast']:.1f} SHIPS/D | "
            f"CAPE {end_fc['cape_forecast']:.1f} SHIPS/D | "
            f"CAPE SHARE {end_fc['forecast_cape_share_pct']:.1f}%"
        )
        banner_cls = "shock-readout-banner"

    return map_fig, traj_fig, ccf_fig, fuel_fig, seis_fig, banner_text, banner_cls


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)