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


def create_cluster_map_with_names(
    df_clustered: pd.DataFrame,
    descriptions: dict = None,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    cluster_col: str = "cluster",
    text_col: str = "text",
    output_html: str = "outputs/map_clusters_named.html",
    sample_n: int = 25000,
    random_state: int = 42,
    center: Tuple[float, float] = (45.7640, 4.8357),
    zoom_start: int = 12,
) -> str:
    """
    Create an interactive Folium map with clusters colored by ID and labeled with TF-IDF descriptions.
    
    Parameters:
    -----------
    descriptions : dict or list
        Either a dict mapping cluster_id -> description string,
        or a list of ClusterDescription objects
    
    Returns:
    --------
    output_html : str
    """
    import os
    import folium
    
    _require_columns(df_clustered, [lat_col, lon_col, cluster_col])
    
    df = df_clustered.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    # Exclude noise
    df = df[df[cluster_col] != -1].copy()
    
    if sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)
    
    # Build cluster centroid map for cluster labels
    cluster_centers = {}
    for cid in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cid]
        center_lat = cluster_data[lat_col].mean()
        center_lon = cluster_data[lon_col].mean()
        cluster_centers[int(cid)] = (center_lat, center_lon)
    
    # Convert descriptions if needed
    descriptions_dict = {}
    if descriptions:
        if isinstance(descriptions, dict):
            descriptions_dict = descriptions
        else:
            # Assume list of ClusterDescription objects
            for desc in descriptions:
                if hasattr(desc, 'cluster_id') and hasattr(desc, 'description'):
                    descriptions_dict[int(desc.cluster_id)] = desc.description
    
    # Build map
    m = folium.Map(location=list(center), zoom_start=zoom_start, control_scale=True)
    
    # Color function
    def color_for_cluster(cid: int) -> str:
        if cid == -1:
            return "#666666"
        import colorsys
        hue = (cid * 47) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    
    # Add markers for each point
    for _, row in df.iterrows():
        cid = int(row[cluster_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            fill=True,
            color=color_for_cluster(cid),
            fill_color=color_for_cluster(cid),
            fill_opacity=0.7,
            opacity=0.7,
            popup=f"Cluster {cid}",
        ).add_to(m)
    
    # Add cluster center markers with labels
    for cid, (center_lat, center_lon) in cluster_centers.items():
        cluster_name = descriptions_dict.get(cid, f"Cluster {cid}")
        
        # Add a larger circle for cluster center
        folium.CircleMarker(
            location=(center_lat, center_lon),
            radius=8,
            fill=True,
            color=color_for_cluster(cid),
            fill_color=color_for_cluster(cid),
            fill_opacity=0.9,
            opacity=0.9,
            popup=folium.Popup(
                f"<b>{cluster_name}</b><br/>Cluster ID: {cid}",
                max_width=300
            ),
            weight=3,
        ).add_to(m)
        
        # Add text label near cluster center
        folium.Marker(
            location=(center_lat, center_lon),
            popup=cluster_name,
            icon=folium.Icon(color='gray', icon='info-sign', prefix='glyphicon'),
        ).add_to(m)
    
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    return output_html


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
