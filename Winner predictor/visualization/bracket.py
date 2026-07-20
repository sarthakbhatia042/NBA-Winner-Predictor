"""
NBA Playoff Predictor — Bracket Visualization
Renders the full playoff bracket using matplotlib with team names,
series scores, and win probabilities.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# =============================================================================
# Color Scheme
# =============================================================================
COLORS = {
    "bg": "#0a0e27",
    "card_bg": "#141937",
    "card_border": "#2a3166",
    "winner_bg": "#0d3320",
    "winner_border": "#1db954",
    "loser_text": "#6b7280",
    "winner_text": "#ffffff",
    "prob_high": "#1db954",
    "prob_mid": "#f59e0b",
    "prob_low": "#ef4444",
    "title_text": "#ffffff",
    "subtitle_text": "#9ca3af",
    "round_label": "#6366f1",
    "connector": "#374151",
    "finals_gold": "#fbbf24",
    "champion_glow": "#fbbf24",
    # Status colors
    "actual_border": "#3b82f6",     # Blue for actual/completed results
    "in_progress_border": "#f59e0b",  # Orange for in-progress series
    "actual_badge_bg": "#1e3a5f",
    "predicted_badge_bg": "#1a1a2e",
}


def _prob_color(prob: float) -> str:
    """Get color based on win probability."""
    if prob >= 0.65:
        return COLORS["prob_high"]
    elif prob >= 0.5:
        return COLORS["prob_mid"]
    else:
        return COLORS["prob_low"]


def _truncate_name(name: str, max_len: int = 18) -> str:
    """Truncate team name if too long."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "…"


def _draw_matchup_box(ax, x: float, y: float, width: float, height: float,
                       series_result: dict, is_finals: bool = False):
    """
    Draw a single matchup box showing two teams, score, and probability.
    Differentiates actual results (blue), in-progress (orange), and predicted.
    """
    higher_name = series_result.get("higher_seed_name", "TBD")
    lower_name = series_result.get("lower_seed_name", "TBD")
    higher_seed = series_result.get("higher_seed_num", "")
    lower_seed = series_result.get("lower_seed_num", "")
    winner_id = series_result.get("winner")
    higher_id = series_result.get("higher_seed_id")
    score = series_result.get("series_score", (0, 0))
    prob = series_result.get("win_probability", 0.5)
    is_actual = series_result.get("is_actual", False)
    is_in_progress = series_result.get("is_in_progress", False)
    
    higher_won = winner_id == higher_id
    
    # Box border color based on status
    if is_finals:
        border_color = COLORS["winner_border"]
        border_width = 2.5
    elif is_actual:
        border_color = COLORS["actual_border"]
        border_width = 2.0
    elif is_in_progress:
        border_color = COLORS["in_progress_border"]
        border_width = 2.0
    else:
        border_color = COLORS["card_border"]
        border_width = 1.5
    
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["card_bg"],
        edgecolor=border_color,
        linewidth=border_width,
    )
    ax.add_patch(box)
    
    # Split box into top (higher seed) and bottom (lower seed)
    mid_y = y + height / 2
    
    # Highlight winner row
    if higher_won:
        winner_highlight = FancyBboxPatch(
            (x + 0.003, mid_y + 0.003), width - 0.006, height / 2 - 0.006,
            boxstyle="round,pad=0.01",
            facecolor=COLORS["winner_bg"],
            edgecolor="none",
        )
        ax.add_patch(winner_highlight)
    else:
        winner_highlight = FancyBboxPatch(
            (x + 0.003, y + 0.003), width - 0.006, height / 2 - 0.006,
            boxstyle="round,pad=0.01",
            facecolor=COLORS["winner_bg"],
            edgecolor="none",
        )
        ax.add_patch(winner_highlight)
    
    # Divider line
    ax.plot([x + 0.01, x + width - 0.01], [mid_y, mid_y],
            color=COLORS["card_border"], linewidth=0.5, alpha=0.5)
    
    # Higher seed text
    seed_str = f"({higher_seed})" if higher_seed else ""
    h_text_color = COLORS["winner_text"] if higher_won else COLORS["loser_text"]
    h_font_weight = "bold" if higher_won else "normal"
    
    ax.text(x + 0.015, mid_y + height / 4, f"{seed_str} {_truncate_name(higher_name)}",
            fontsize=7, color=h_text_color, fontweight=h_font_weight,
            va="center", ha="left", fontfamily="monospace")
    
    # Higher seed score
    h_score = score[0] if higher_won else score[1]
    ax.text(x + width - 0.015, mid_y + height / 4, str(h_score),
            fontsize=8, color=h_text_color, fontweight="bold",
            va="center", ha="right", fontfamily="monospace")
    
    # Lower seed text
    seed_str = f"({lower_seed})" if lower_seed else ""
    l_text_color = COLORS["winner_text"] if not higher_won else COLORS["loser_text"]
    l_font_weight = "bold" if not higher_won else "normal"
    
    ax.text(x + 0.015, y + height / 4, f"{seed_str} {_truncate_name(lower_name)}",
            fontsize=7, color=l_text_color, fontweight=l_font_weight,
            va="center", ha="left", fontfamily="monospace")
    
    # Lower seed score
    l_score = score[1] if higher_won else score[0]
    ax.text(x + width - 0.015, y + height / 4, str(l_score),
            fontsize=8, color=l_text_color, fontweight="bold",
            va="center", ha="right", fontfamily="monospace")
    
    # Status badge below the box
    if is_actual:
        badge_text = "ACTUAL"
        badge_color = COLORS["actual_border"]
        badge_bg = COLORS["actual_badge_bg"]
    elif is_in_progress:
        badge_text = "LIVE+PRED"
        badge_color = COLORS["in_progress_border"]
        badge_bg = COLORS["predicted_badge_bg"]
    else:
        prob_display = prob if higher_won else (1 - prob)
        badge_text = f"PRED {prob_display:.0%}"
        badge_color = _prob_color(prob_display)
        badge_bg = COLORS["predicted_badge_bg"]
    
    ax.text(x + width / 2, y - 0.015, badge_text,
            fontsize=5.5, color=badge_color, fontweight="bold",
            va="center", ha="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=badge_bg,
                     edgecolor=badge_color, linewidth=0.8, alpha=0.9))


def _draw_connector(ax, x1: float, y1: float, x2: float, y2: float):
    """Draw a connecting line between matchup boxes."""
    mid_x = (x1 + x2) / 2
    ax.plot([x1, mid_x], [y1, y1], color=COLORS["connector"], linewidth=1, alpha=0.4)
    ax.plot([mid_x, mid_x], [min(y1, y2), max(y1, y2)],
            color=COLORS["connector"], linewidth=1, alpha=0.4)
    ax.plot([mid_x, x2], [y2, y2], color=COLORS["connector"], linewidth=1, alpha=0.4)


def render_bracket(bracket: dict, season: str, model_type: str = "XGBoost",
                   save_path: str = None):
    """
    Render the full NBA playoff bracket.
    
    Parameters
    ----------
    bracket : dict
        Output from simulate_bracket()
    season : str
        Season string (e.g., "2025-26")
    model_type : str
        Name of model used for display
    save_path : str
        Path to save the PNG. If None, saves to output dir.
    """
    fig, ax = plt.subplots(1, 1, figsize=(22, 14), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    # ── Title ──
    ax.text(0.5, 0.96, f"{season} NBA PLAYOFF PREDICTIONS",
            fontsize=20, color=COLORS["title_text"], fontweight="bold",
            va="center", ha="center", fontfamily="sans-serif")
    ax.text(0.5, 0.935, f"Model: {model_type}  |  Actual Results + Predictions  |  Best-of-7 Series",
            fontsize=9, color=COLORS["subtitle_text"],
            va="center", ha="center", fontfamily="sans-serif")
    
    # ── Legend ──
    legend_y = 0.91
    legend_items = [
        ("ACTUAL", COLORS["actual_border"], COLORS["actual_badge_bg"]),
        ("LIVE+PRED", COLORS["in_progress_border"], COLORS["predicted_badge_bg"]),
        ("PREDICTED", COLORS["prob_mid"], COLORS["predicted_badge_bg"]),
    ]
    legend_x_start = 0.38
    for i, (label, color, bg) in enumerate(legend_items):
        lx = legend_x_start + i * 0.09
        ax.plot([lx, lx + 0.015], [legend_y, legend_y], color=color, linewidth=3)
        ax.text(lx + 0.02, legend_y, label, fontsize=6.5, color=color,
                va="center", ha="left", fontfamily="monospace", fontweight="bold")
    
    # ── Layout Configuration ──
    box_w = 0.145
    box_h = 0.065
    
    # Column x-positions (left to right): East R1, East Semis, East Finals, NBA Finals, West Finals, West Semis, West R1
    col_x = {
        "east_r1": 0.02,
        "east_semi": 0.195,
        "east_conf": 0.37,
        "finals": 0.425,
        "west_conf": 0.49,
        "west_semi": 0.66,
        "west_r1": 0.835,
    }
    
    # Y positions for each round
    r1_ys = [0.78, 0.67, 0.52, 0.41]           # 4 series per conference
    semi_ys = [0.725, 0.465]                     # 2 series per conference
    conf_y = [0.595]                             # 1 series per conference
    finals_y = 0.595                             # NBA Finals
    
    # ── Round Labels ──
    label_y = 0.88
    for label, x in [("FIRST ROUND", col_x["east_r1"] + box_w / 2),
                      ("CONF SEMIS", col_x["east_semi"] + box_w / 2),
                      ("CONF FINALS", col_x["east_conf"] + box_w / 2),
                      ("NBA FINALS", col_x["finals"] + box_w / 2 + 0.02),
                      ("CONF FINALS", col_x["west_conf"] + box_w / 2),
                      ("CONF SEMIS", col_x["west_semi"] + box_w / 2),
                      ("FIRST ROUND", col_x["west_r1"] + box_w / 2)]:
        ax.text(x, label_y, label, fontsize=7, color=COLORS["round_label"],
                fontweight="bold", va="center", ha="center", fontfamily="sans-serif",
                alpha=0.8)
    
    # Conference labels
    ax.text(col_x["east_r1"] + 0.17, 0.85, "EASTERN CONFERENCE",
            fontsize=10, color=COLORS["subtitle_text"], fontweight="bold",
            va="center", ha="center", fontfamily="sans-serif", alpha=0.6)
    ax.text(col_x["west_r1"] - 0.02, 0.85, "WESTERN CONFERENCE",
            fontsize=10, color=COLORS["subtitle_text"], fontweight="bold",
            va="center", ha="center", fontfamily="sans-serif", alpha=0.6)
    
    # ── Draw East Bracket ──
    east = bracket.get("East", {})
    
    # First Round
    east_r1 = east.get("First Round", [])
    for i, series in enumerate(east_r1[:4]):
        _draw_matchup_box(ax, col_x["east_r1"], r1_ys[i], box_w, box_h, series)
    
    # Semis
    east_semi = east.get("Conference Semifinals", [])
    for i, series in enumerate(east_semi[:2]):
        _draw_matchup_box(ax, col_x["east_semi"], semi_ys[i], box_w, box_h, series)
    
    # Connectors: R1 → Semis
    for i in range(0, min(4, len(east_r1)), 2):
        if i // 2 < len(semi_ys):
            r1_center_1 = r1_ys[i] + box_h / 2
            r1_center_2 = r1_ys[i + 1] + box_h / 2
            semi_center = semi_ys[i // 2] + box_h / 2
            _draw_connector(ax, col_x["east_r1"] + box_w, r1_center_1,
                          col_x["east_semi"], semi_center)
            _draw_connector(ax, col_x["east_r1"] + box_w, r1_center_2,
                          col_x["east_semi"], semi_center)
    
    # Conference Finals
    east_conf = east.get("Conference Finals", [])
    for i, series in enumerate(east_conf[:1]):
        _draw_matchup_box(ax, col_x["east_conf"], conf_y[0], box_w, box_h, series)
    
    # Connectors: Semis → Conf Finals
    for i in range(min(2, len(east_semi))):
        semi_center = semi_ys[i] + box_h / 2
        conf_center = conf_y[0] + box_h / 2
        _draw_connector(ax, col_x["east_semi"] + box_w, semi_center,
                      col_x["east_conf"], conf_center)
    
    # ── Draw West Bracket (mirrored) ──
    west = bracket.get("West", {})
    
    # First Round
    west_r1 = west.get("First Round", [])
    for i, series in enumerate(west_r1[:4]):
        _draw_matchup_box(ax, col_x["west_r1"], r1_ys[i], box_w, box_h, series)
    
    # Semis
    west_semi = west.get("Conference Semifinals", [])
    for i, series in enumerate(west_semi[:2]):
        _draw_matchup_box(ax, col_x["west_semi"], semi_ys[i], box_w, box_h, series)
    
    # Connectors: R1 → Semis
    for i in range(0, min(4, len(west_r1)), 2):
        if i // 2 < len(semi_ys):
            r1_center_1 = r1_ys[i] + box_h / 2
            r1_center_2 = r1_ys[i + 1] + box_h / 2
            semi_center = semi_ys[i // 2] + box_h / 2
            _draw_connector(ax, col_x["west_r1"], r1_center_1,
                          col_x["west_semi"] + box_w, semi_center)
            _draw_connector(ax, col_x["west_r1"], r1_center_2,
                          col_x["west_semi"] + box_w, semi_center)
    
    # Conference Finals
    west_conf = west.get("Conference Finals", [])
    for i, series in enumerate(west_conf[:1]):
        _draw_matchup_box(ax, col_x["west_conf"], conf_y[0], box_w, box_h, series)
    
    # Connectors: Semis → Conf Finals
    for i in range(min(2, len(west_semi))):
        semi_center = semi_ys[i] + box_h / 2
        conf_center = conf_y[0] + box_h / 2
        _draw_connector(ax, col_x["west_semi"], semi_center,
                      col_x["west_conf"] + box_w, conf_center)
    
    # ── NBA Finals ──
    finals = bracket.get("Finals")
    if finals:
        finals_box_w = 0.155
        finals_box_h = 0.075
        finals_x = 0.423
        _draw_matchup_box(ax, finals_x, finals_y - 0.13, finals_box_w,
                          finals_box_h, finals, is_finals=True)
        
        # Connector: Conf Finals → Finals
        east_conf_center = conf_y[0] + box_h / 2
        west_conf_center = conf_y[0] + box_h / 2
        finals_center = (finals_y - 0.13) + finals_box_h / 2
        
        _draw_connector(ax, col_x["east_conf"] + box_w, east_conf_center,
                      finals_x, finals_center)
        _draw_connector(ax, col_x["west_conf"], west_conf_center,
                      finals_x + finals_box_w, finals_center)
    
    # ── Champion Display ──
    champion = bracket.get("champion", "TBD")
    if champion and champion != "TBD":
        champ_y = 0.12
        
        # Champion box
        champ_box = FancyBboxPatch(
            (0.32, champ_y - 0.03), 0.36, 0.07,
            boxstyle="round,pad=0.02",
            facecolor="#1a1a2e",
            edgecolor=COLORS["champion_glow"],
            linewidth=3,
        )
        ax.add_patch(champ_box)
        
        ax.text(0.5, champ_y + 0.02, ">>> NBA CHAMPION <<<",
                fontsize=10, color=COLORS["champion_glow"], fontweight="bold",
                va="center", ha="center", fontfamily="sans-serif")
        ax.text(0.5, champ_y - 0.005, champion,
                fontsize=16, color=COLORS["title_text"], fontweight="bold",
                va="center", ha="center", fontfamily="sans-serif")
    
    # ── Footer ──
    ax.text(0.5, 0.03, "Built with nba_api | Logistic Regression + XGBoost | Leave-One-Season-Out CV",
            fontsize=7, color=COLORS["subtitle_text"],
            va="center", ha="center", fontfamily="sans-serif", alpha=0.5)
    
    plt.tight_layout(pad=0.5)
    
    # Save
    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, f"bracket_{season.replace('-', '_')}.png")
    
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"\n  Bracket saved to {save_path}")
    
    plt.close(fig)
    return save_path
