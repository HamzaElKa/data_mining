# src/visualization.py
# Session 1 — Map visualization (Folium) from cleaned Flickr data
# Input: df_clean with columns: lat, long, text, taken_dt/upload_dt (optional)
# Output: an interactive HTML map you can open in your browser

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class MapConfig:
    output_html: str = "outputs/map_session1.html"
    sample_n: int = 15000          # keep map smooth
    random_state: int = 42
    center: Tuple[float, float] = (45.7640, 4.8357)  # Lyon center
    zoom_start: int = 12
    max_markers: int = 15000       # safety


def _safe_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for visualization: {missing}")


def create_map(
    df_clean: pd.DataFrame,
    *,
    cfg: MapConfig = MapConfig(),
    lat_col: str = "lat",
    lon_col: str = "long",
    text_col: str = "text",
) -> str:
    """
    Create an interactive Folium map with clustered markers.
    Returns the output HTML path.
    """
    _require_columns(df_clean, [lat_col, lon_col])

    # Lazy import: folium only needed here
    import folium
    from folium.plugins import MarkerCluster

    df = df_clean.copy()

    # Keep only valid numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    # Sample to keep performance
    df = _safe_sample(df, cfg.sample_n, cfg.random_state)
    if len(df) > cfg.max_markers:
        df = df.head(cfg.max_markers)

    # Build map
    m = folium.Map(location=list(cfg.center), zoom_start=cfg.zoom_start, control_scale=True)
    cluster = MarkerCluster(name="Photos").add_to(m)

    # Add markers
    has_text = text_col in df.columns
    has_taken = "taken_dt" in df.columns
    has_upload = "upload_dt" in df.columns

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        # Popup content (keep it short to stay fast)
        parts = []
        if "id" in df.columns:
            parts.append(f"<b>ID</b>: {row.get('id','')}")
        if "user" in df.columns:
            parts.append(f"<b>User</b>: {row.get('user','')}")
        if has_taken:
            td = row.get("taken_dt", None)
            if pd.notna(td):
                parts.append(f"<b>Taken</b>: {str(td)[:19]}")
        if has_upload:
            ud = row.get("upload_dt", None)
            if pd.notna(ud):
                parts.append(f"<b>Upload</b>: {str(ud)[:19]}")
        if has_text:
            txt = str(row.get(text_col, "") or "")
            if len(txt) > 160:
                txt = txt[:160] + "..."
            if txt.strip():
                parts.append(f"<b>Text</b>: {txt}")

        popup_html = "<br/>".join(parts) if parts else None

        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            fill=True,
            popup=folium.Popup(popup_html, max_width=350) if popup_html else None,
        ).add_to(cluster)

    # Make sure output folder exists
    import os
    os.makedirs(os.path.dirname(cfg.output_html), exist_ok=True)

    m.save(cfg.output_html)
    return cfg.output_html


if __name__ == "__main__":
    # Full pipeline test:
    # Run from project root:
    #   python src/visualization.py
    try:
        from load_data import load_data
        from cleaning import clean_data

        df_raw, _ = load_data("../data/flickr_data2.csv")
        df_clean, rep = clean_data(df_raw)

        out = create_map(df_clean)
        print(f"[OK] Map saved to: {out}")
        print("Open it in your browser (double click the .html).")

    except Exception as e:
        print(f"[ERROR] {e}")
