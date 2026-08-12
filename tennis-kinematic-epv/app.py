"""
app.py
------
Streamlit Dashboard for Tennis Kinematic Expected Point Value (EPV) Engine.
Integrates custom differential pursuit kinematics, XGBoost inference,
SHAP explainability, and Plotly surface optimization heatmaps.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from xgboost import XGBClassifier

# --- STREAMLIT PAGE CONFIGURATION & DYNAMIC STYLING ---
st.set_page_config(
    page_title="Tennis Kinematic EPV Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for dense metrics display and theme responsiveness
st.markdown(
    """
    <style>
        .block-container { 
            padding-top: 1rem !important; 
            padding-bottom: 0rem !important; 
            padding-left: 1.5rem !important; 
            padding-right: 1.5rem !important; 
        }
        section[data-testid="stSidebar"] > div { 
            padding-top: 1rem !important; 
            padding-bottom: 0.5rem !important; 
        }
        section[data-testid="stSidebar"] div[data-baseweb="input"] { 
            min-height: 28px !important; 
            height: 28px !important; 
        }
        section[data-testid="stSidebar"] input { 
            font-size: 0.8rem !important; 
            padding: 2px 6px !important; 
        }
        @media (prefers-color-scheme: light) {
            div[data-testid="stMetric"] {
                background-color: rgba(0, 0, 0, 0.04) !important;
                border: 1px solid rgba(0, 0, 0, 0.08) !important;
                padding: 4px 8px !important;
                border-radius: 6px;
            }
        }
        @media (prefers-color-scheme: dark) {
            div[data-testid="stMetric"] {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                padding: 4px 8px !important;
                border-radius: 6px;
            }
        }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    </style>
""",
    unsafe_allow_html=True,
)

# --- COURT BOUNDARY CONSTANTS (Meters) ---
HALF_WIDTH = 4.115     # Standard singles court half-width
HALF_LENGTH = 11.885   # Standard court half-length
OUTER_BOUND_X = 7.5    # Extended out-of-bounds boundary X
OUTER_BOUND_Y = 15.0   # Extended out-of-bounds boundary Y

@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost model artifact."""
    # Anchor the directory path directly to where app.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "epv_xgboost_model.pkl")
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(
            "Model file 'epv_xgboost_model.pkl' not found. "
            "Please run 'python train_model.py' first to generate the model artifact."
        )
        st.stop()

model = load_model()

class PhysicsTennisEngine:
    """
    Handles kinematic trajectory physics, differential interception vector calculations,
    and point validity checks.
    """

    @staticmethod
    def calculate_positions(
        defender_pos: np.ndarray,
        defender_vel: float,
        ball_pos: np.ndarray,
        ball_avg_vel: float,
        ball_target: np.ndarray,
        surface_friction: float = 0.65,
        spin_deflection_deg: float = 0.0,
    ):
        """Calculates ball flight time, defender interception path, and post-bounce trajectories."""
        flight_vec = ball_target - ball_pos
        dist_flight = np.linalg.norm(flight_vec)
        t_flight = dist_flight / max(ball_avg_vel, 0.1)
        v_ball_vec = flight_vec / max(t_flight, 1e-5)

        # Quadratic differential pursuit solution
        D = ball_pos - defender_pos
        A = np.dot(v_ball_vec, v_ball_vec) - (defender_vel**2)
        B = 2.0 * np.dot(D, v_ball_vec)
        C = np.dot(D, D)

        discriminant = B**2 - 4 * A * C
        t_intercept = None

        if discriminant >= 0:
            t1 = (-B - np.sqrt(discriminant)) / (2 * A)
            t2 = (-B + np.sqrt(discriminant)) / (2 * A)
            valid_times = [t for t in [t1, t2] if 0 <= t <= t_flight + 0.1]
            if valid_times:
                t_intercept = min(valid_times)

        if t_intercept is not None:
            ideal_target = ball_pos + v_ball_vec * t_intercept
        else:
            incoming_angle_rad = np.arctan2(flight_vec[1], flight_vec[0])
            post_angle_rad = incoming_angle_rad + np.radians(spin_deflection_deg)
            bounce_dir = np.array([np.cos(post_angle_rad), np.sin(post_angle_rad)])
            bounce_dir[1] = -abs(bounce_dir[1])
            bounce_unit = bounce_dir / (np.linalg.norm(bounce_dir) + 1e-6)
            ideal_target = ball_target + bounce_unit * 2.0

        # Constrain target to defender side of court
        ideal_target[0] = np.clip(ideal_target[0], -OUTER_BOUND_X, OUTER_BOUND_X)
        ideal_target[1] = np.clip(ideal_target[1], -OUTER_BOUND_Y, -0.1)

        # Calculate max distance defender can travel during ball flight time
        max_dist_run = defender_vel * t_flight
        vec_to_ideal = ideal_target - defender_pos
        dist_to_ideal = np.linalg.norm(vec_to_ideal)

        if dist_to_ideal <= max_dist_run:
            best_reachable_pos = ideal_target.copy()
            can_reach = True
        else:
            unit_dir = vec_to_ideal / (dist_to_ideal + 1e-6)
            best_reachable_pos = defender_pos + unit_dir * max_dist_run
            can_reach = False

        # Compute post-bounce ray bounds
        incoming_angle_rad = np.arctan2(flight_vec[1], flight_vec[0])
        post_angle_rad = incoming_angle_rad + np.radians(spin_deflection_deg)
        bounce_dir = np.array([np.cos(post_angle_rad), np.sin(post_angle_rad)])
        bounce_dir[1] = -abs(bounce_dir[1])
        bounce_unit = bounce_dir / (np.linalg.norm(bounce_dir) + 1e-6)

        t_x = (
            (OUTER_BOUND_X - ball_target[0]) / bounce_unit[0]
            if bounce_unit[0] > 0
            else (-OUTER_BOUND_X - ball_target[0]) / bounce_unit[0]
        )
        t_y = (-OUTER_BOUND_Y - ball_target[1]) / bounce_unit[1]
        t_exit = min(abs(t_x), abs(t_y))
        bounce_end = ball_target + bounce_unit * t_exit

        return (
            ideal_target,
            best_reachable_pos,
            can_reach,
            t_flight,
            bounce_end,
            max(ball_avg_vel * surface_friction, 0.1),
        )

    @classmethod
    def evaluate_point_state(
        cls,
        defender_pos,
        defender_vel,
        ball_pos,
        ball_avg_vel,
        ball_target,
        surface_friction,
        spin_deflection_deg,
    ):
        """Evaluates whether the shot is valid and calculates court deficit metrics."""
        is_own_court = ball_target[1] >= 0
        is_out = (abs(ball_target[0]) > HALF_WIDTH) or (
            ball_target[1] < -HALF_LENGTH
        )
        is_valid_shot = (not is_own_court) and (not is_out)

        (
            ideal_pos,
            best_reachable_pos,
            can_reach,
            t_flight,
            bounce_end,
            v_post,
        ) = cls.calculate_positions(
            defender_pos,
            defender_vel,
            ball_pos,
            ball_avg_vel,
            ball_target,
            surface_friction,
            spin_deflection_deg,
        )

        dist_to_ideal = np.linalg.norm(ideal_pos - defender_pos)
        t_reach = dist_to_ideal / max(defender_vel, 0.1)
        time_margin = t_reach - t_flight

        total_deficit = np.linalg.norm(ideal_pos - best_reachable_pos)
        exposed_area = total_deficit * 4.0

        return {
            "ideal_position": ideal_pos,
            "best_reachable_pos": best_reachable_pos,
            "bounce_end": bounce_end,
            "v_post_bounce": round(v_post, 1),
            "t_flight": round(t_flight, 2),
            "t_reach": round(t_reach, 2),
            "time_margin": time_margin,
            "total_deficit": round(total_deficit, 2),
            "exposed_area": round(exposed_area, 2),
            "can_reach": can_reach,
            "is_valid_shot": is_valid_shot,
            "is_own_court": is_own_court,
            "is_out": is_out,
        }

    @staticmethod
    def compute_smooth_probability(results, raw_model_prob):
        """Combines raw XGBoost output with kinematic time margin adjustments."""
        if not results["is_valid_shot"]:
            return 0.0
        time_diff = results["time_margin"]
        reach_factor = 1.0 / (1.0 + np.exp(-4.0 * time_diff))
        return float(
            np.clip(
                (1.0 - reach_factor) * raw_model_prob + reach_factor * 0.98,
                0.01,
                0.99,
            )
        )


def compute_epv_grid(
    defender_pos, v_def, ball_pos, v_ball_avg, surface_mu, spin_alpha, model, grid_res=20
):
    """Computes a 2D surface matrix evaluating EPV across the entire opponent court."""
    x_range = np.linspace(-HALF_WIDTH, HALF_WIDTH, grid_res)
    y_range = np.linspace(-HALF_LENGTH, HALF_LENGTH, grid_res)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            target = np.array([X[i, j], Y[i, j]])
            if Y[i, j] >= 0:
                Z[i, j] = 0.0
                continue

            res = PhysicsTennisEngine.evaluate_point_state(
                defender_pos, v_def, ball_pos, v_ball_avg, target, surface_mu, spin_alpha
            )

            sample = pd.DataFrame(
                [
                    {
                        "recovery_deficit_m": res["total_deficit"],
                        "exposed_area_m2": res["exposed_area"],
                        "deficit_x_exposed": res["total_deficit"] * res["exposed_area"],
                    }
                ]
            )

            raw_p = model.predict_proba(sample)[0][1]
            Z[i, j] = PhysicsTennisEngine.compute_smooth_probability(res, raw_p)

    return X, Y, Z


def draw_epv_heatmap(X, Y, Z, defender_pos, ball_pos, ball_target, results):
    """Generates Plotly Contour map for Tab 2 (Showing starting ball point ONLY, no trajectory)."""
    max_idx = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    best_x, best_y, best_epv = X[max_idx], Y[max_idx], Z[max_idx] * 100

    fig = go.Figure(
        data=go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=Z * 100,
            colorscale="Viridis",
            colorbar=dict(title="EPV %"),
            contours=dict(coloring="heatmap", showlabels=True),
        )
    )

    # Add court boundary geometry
    hw, hl = HALF_WIDTH, HALF_LENGTH
    fig.add_shape(type="rect", x0=-hw, y0=-hl, x1=hw, y1=hl, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-hw, y0=0, x1=hw, y1=0, line=dict(color="white", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-hw, y0=-6.40, x1=hw, y1=-6.40, line=dict(color="white", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-hw, y0=6.40, x1=hw, y1=6.40, line=dict(color="white", width=1, dash="dash"))

    # Render ONLY Ball Origin point (No trajectory line)
    fig.add_trace(
        go.Scatter(
            x=[ball_pos[0]],
            y=[ball_pos[1]],
            mode="markers+text",
            marker=dict(size=12, color="#FF1E27", symbol="circle", line=dict(color="white", width=2)),
            text=["Ball Origin"],
            textposition="top center",
            name="Ball Origin",
        )
    )

    # Render Defender Location
    fig.add_trace(
        go.Scatter(
            x=[defender_pos[0]],
            y=[defender_pos[1]],
            mode="markers+text",
            marker=dict(size=14, color="#FF007F", symbol="x", line=dict(color="white", width=1.5)),
            text=["Defender"],
            textposition="top center",
            name="Defender",
        )
    )

    # Render Optimal Surface Landing Target Spot
    if best_epv > 0:
        fig.add_trace(
            go.Scatter(
                x=[best_x],
                y=[best_y],
                mode="markers+text",
                marker=dict(size=16, color="#00FF66", symbol="star", line=dict(color="white", width=2)),
                text=[f"Max Target ({best_epv:.1f}%)"],
                textposition="bottom center",
                name="Max Target Spot",
            )
        )

    fig.update_layout(
        title="Surface EPV Heatmap",
        xaxis=dict(range=[-5.0, 5.0], visible=False),
        yaxis=dict(range=[-12.5, 12.5], visible=False, scaleanchor="x", scaleratio=1),
        width=480,
        height=580,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig, best_x, best_y, best_epv


def generate_llm_summary(results, epv_prob, inputs_dict):
    """Generates natural language tactical summary based on game state."""
    if results["is_own_court"]:
        return "**Tactical Assessment:** Shot terminated in hitter's own court (Net Fault). EPV: **0.0%**."
    if results["is_out"]:
        return "**Tactical Assessment:** Shot landed out of bounds. EPV: **0.0%**."

    summary = f"**Situation Overview:** Ball played at **{inputs_dict['v_ball_avg']} m/s**.\n\n"
    if results["can_reach"]:
        summary += (
            f"**Defender Coverage:** Defender speed (**{inputs_dict['v_def']} m/s**) allows full recovery "
            f"within **{results['t_flight']}s** flight time. Point recovery chance is high (**EPV: {epv_prob:.1%}**)."
        )
    else:
        summary += (
            f"**Defender Coverage:** Defender caught out of position with a **{results['total_deficit']:.2f} m deficit**, "
            f"exposing **{results['exposed_area']:.1f} m² of open court**. Offensive advantage is elevated (**EPV: {epv_prob:.1%}**)."
        )
    return summary


def render_shap_waterfall(model, feature_df):
    """Generates SHAP waterfall diagnostic chart."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(feature_df)

        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        shap.plots.waterfall(shap_values[0], show=False)

        for child in ax.get_children():
            if isinstance(child, plt.Text):
                child.set_color("#E0E0E0")

        ax.tick_params(colors="#E0E0E0", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = base_val[0]

        return base_val, shap_values[0].values, feature_df.columns.tolist(), base_val + np.sum(shap_values[0].values)
    except Exception as e:
        st.error(f"SHAP explanation unavailable: {e}")
        return None, None, None, None


def draw_physics_court(defender_pos, ball_pos, ball_target, results):
    """Generates Plotly Dynamic Pursuit Map with full trajectory vectors for Tab 1."""
    hw, hl = HALF_WIDTH, HALF_LENGTH
    fig = go.Figure()

    # Draw court lines
    fig.add_shape(type="rect", x0=-hw, y0=-hl, x1=hw, y1=hl, line=dict(color="gray", width=2.5))
    fig.add_shape(type="line", x0=-hw, y0=0, x1=hw, y1=0, line=dict(color="gray", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-hw, y0=-6.40, x1=hw, y1=-6.40, line=dict(color="gray", width=1.2))
    fig.add_shape(type="line", x0=-hw, y0=6.40, x1=hw, y1=6.40, line=dict(color="gray", width=1.2))

    if results["is_valid_shot"]:
        bounce_end = results["bounce_end"]
        fig.add_trace(
            go.Scatter(
                x=[ball_target[0], bounce_end[0]],
                y=[ball_target[1], bounce_end[1]],
                mode="lines",
                line=dict(color="#FF5733", dash="dot", width=2),
                name="Post-Bounce Path",
            )
        )

    # Ball trajectory
    fig.add_trace(
        go.Scatter(
            x=[ball_pos[0], ball_target[0]],
            y=[ball_pos[1], ball_target[1]],
            mode="lines+markers",
            line=dict(color="#FF1E27", width=2.5),
            marker=dict(size=[6, 12], color="#FF1E27"),
            name="In-Flight Path",
        )
    )

    # Defender path
    reachable_pos = results["best_reachable_pos"]
    def_color = "#00FF66" if results["can_reach"] else "#FF4B4B"
    fig.add_trace(
        go.Scatter(
            x=[defender_pos[0], reachable_pos[0]],
            y=[defender_pos[1], reachable_pos[1]],
            mode="lines",
            line=dict(color=def_color, width=2.5, dash="dash"),
            name="Defender Path",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[defender_pos[0]],
            y=[defender_pos[1]],
            mode="markers",
            marker=dict(size=16, color="#AB63FA" if results["can_reach"] else "#FF4B4B"),
            name="Defender",
        )
    )

    fig.update_layout(
        title="Dynamic Pursuit Intercept Map",
        xaxis=dict(range=[-7.5, 7.5], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-15.0, 13.5], visible=False),
        width=420,
        height=580,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


# --- SIDEBAR INPUT CONTROL SYNCING ---
st.title("🎾 Kinematic EPV Dashboard")


def slider_num_input(label, key, min_val, max_val, default_val, step_val):
    """Synchronizes Streamlit slider and numeric input widgets."""
    st.sidebar.markdown(f"**{label}**")
    col1, col2 = st.sidebar.columns([2, 1])

    is_int = isinstance(default_val, int) and isinstance(step_val, int)
    cast = int if is_int else float

    key_slider, key_num = f"{key}_slider", f"{key}_num"

    if key_slider not in st.session_state:
        st.session_state[key_slider] = cast(default_val)
    if key_num not in st.session_state:
        st.session_state[key_num] = cast(default_val)

    def sync_from_slider():
        st.session_state[key_num] = st.session_state[key_slider]

    def sync_from_num():
        st.session_state[key_slider] = st.session_state[key_num]

    col1.slider(
        label, min_value=cast(min_val), max_value=cast(max_val), step=cast(step_val),
        key=key_slider, on_change=sync_from_slider, label_visibility="collapsed"
    )
    col2.number_input(
        label, min_value=cast(min_val), max_value=cast(max_val), step=cast(step_val),
        key=key_num, on_change=sync_from_num, label_visibility="collapsed"
    )
    return st.session_state[key_slider]


# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Global Heatmap Drivers")
b_x = slider_num_input("Ball Origin X (m)", "b_x", -4.115, 4.115, 2.0, 0.1)
b_y = slider_num_input("Ball Origin Y (m)", "b_y", 0.1, 11.885, 8.0, 0.1)
v_ball_avg = slider_num_input("Ball Speed (m/s)", "v_ball_avg", 10.0, 50.0, 25.0, 0.5)
surface_mu = slider_num_input("Friction (μ)", "surface_mu", 0.30, 0.90, 0.65, 0.05)
spin_alpha = slider_num_input("Spin Angle (°)", "spin_alpha", -30.0, 30.0, 5.0, 0.5)

def_x = slider_num_input("Defender Pos X (m)", "def_x", -4.115, 4.115, 0.0, 0.1)
def_y = slider_num_input("Defender Pos Y (m)", "def_y", -11.885, 0.0, -11.0, 0.1)
v_def = slider_num_input("Defender Speed (m/s)", "v_def", 1.0, 10.0, 4.5, 0.1)

st.sidebar.divider()
st.sidebar.header("2. Single-Shot Target Line")
b_target_x = slider_num_input("Ball Landing X (m)", "b_target_x", -6.0, 6.0, -3.5, 0.1)
b_target_y = slider_num_input("Ball Landing Y (m)", "b_target_y", -13.0, 11.885, -9.0, 0.1)

# Execution
defender_pos = np.array([def_x, def_y])
ball_pos = np.array([b_x, b_y])
ball_target = np.array([b_target_x, b_target_y])

results = PhysicsTennisEngine.evaluate_point_state(
    defender_pos, v_def, ball_pos, v_ball_avg, ball_target, surface_mu, spin_alpha
)

sample_input = pd.DataFrame(
    [
        {
            "recovery_deficit_m": results["total_deficit"],
            "exposed_area_m2": results["exposed_area"],
            "deficit_x_exposed": results["total_deficit"] * results["exposed_area"],
        }
    ]
)

raw_prob = model.predict_proba(sample_input)[0][1]
final_prob = PhysicsTennisEngine.compute_smooth_probability(results, raw_prob)

# Top Bar Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shot EPV", f"{final_prob:.1%}")
c2.metric("Ideal Spot (⭐)", f"({results['ideal_position'][0]:.2f}m, {results['ideal_position'][1]:.2f}m)" if results["is_valid_shot"] else "N/A")
c3.metric("Max Reachable (📍)", f"({results['best_reachable_pos'][0]:.2f}m, {results['best_reachable_pos'][1]:.2f}m)" if results["is_valid_shot"] else "N/A")
c4.metric("Flight Time", f"{results['t_flight']} s")

# Layout Tabs
tab1, tab2 = st.tabs(["🎾 Dynamic Pursuit Map", "🔥 Target EPV Heatmap"])

with tab1:
    left_col, right_col = st.columns([1.1, 1])
    with left_col:
        fig_court = draw_physics_court(defender_pos, ball_pos, ball_target, results)
        st.plotly_chart(fig_court, width="stretch")
    with right_col:
        st.markdown("### 🤖 Language Model Situation Summary")
        st.info(generate_llm_summary(results, final_prob, {"v_ball_avg": v_ball_avg, "v_def": v_def}))

        if results["is_valid_shot"]:
            base_val, shap_vals, feature_names, fx_val = render_shap_waterfall(model, sample_input)
            with st.expander("ℹ️ Complete Mathematical Derivation & SHAP Log-Odds", expanded=True):
                if base_val is not None:
                    st.latex(r"f(x) = E[f(X)] + \sum_{i=1}^{n} \text{SHAP}_i")
                    st.latex(f"f(x) = \\mathbf{{{base_val:+.3f}}} \\implies \\mathbf{{{fx_val:.3f}}}")
                    st.latex(f"\\text{{Probability}} = \\frac{{1}}{{1 + e^{{-({fx_val:.3f})}}}} \\approx \\mathbf{{{raw_prob * 100:.1f}\\%}}")

with tab2:
    st.markdown("### 🎯 Optimal Shot Landing Surface Optimization")
    with st.spinner("Calculating surface EPV matrix..."):
        X_grid, Y_grid, Z_grid = compute_epv_grid(defender_pos, v_def, ball_pos, v_ball_avg, surface_mu, spin_alpha, model)
        fig_heatmap, best_x, best_y, best_epv = draw_epv_heatmap(X_grid, Y_grid, Z_grid, defender_pos, ball_pos, ball_target, results)

        h_col1, h_col2 = st.columns([1.1, 1])
        with h_col1:
            st.plotly_chart(fig_heatmap, width="stretch")
        with h_col2:
            st.markdown("### 📊 Target Optimization Insights")
            st.metric("Max Surface EPV", f"{best_epv:.1f}%")
            st.write(f"**Optimal Landing Spot:** `({best_x:.2f}m, {best_y:.2f}m)`")
