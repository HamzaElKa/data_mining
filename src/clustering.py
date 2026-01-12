# src/clustering.py
# Session 1 — Spatial clustering (DBSCAN / HDBSCAN-like baseline) on Flickr GPS points
# Input: df_clean from cleaning.py with columns: lat, long, (optional) text, taken_dt, etc.
#
# Why DBSCAN for Session 1:
# - No need to choose K
# - Finds dense areas (hotspots) + labels noise
# - Works well for geo points when using haversine distance
#
# Output:
# - df_clustered: dataframe with a new column 'cluster'
# - ClusterReport: summary (n_clusters, noise %, top clusters sizes)
# - Optionally: save clusters to CSV + generate a folium map per clusters

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterReport:
    n_rows_in: int
    n_rows_used: int
    eps_meters: float
    min_samples: int
    n_clusters: int
    noise_points: int
    noise_ratio: float
    cluster_sizes_top10: List[Tuple[int, int]]  # (cluster_id, size)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for clustering: {missing}")


def _to_radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # shape: (n, 2)
    return np.radians(np.column_stack([lat, lon]))


def _eps_meters_to_radians(eps_m: float) -> float:
    # Earth radius ~ 6371 km
    earth_radius_m = 6_371_000.0
    return eps_m / earth_radius_m


def run_dbscan_geo(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    eps_meters: float = 120.0,
    min_samples: int = 30,
    cluster_col: str = "cluster",
) -> Tuple[pd.DataFrame, ClusterReport]:
    """
    Run DBSCAN with haversine distance on (lat,lon).

    eps_meters:
      radius (in meters) defining neighborhood density.
      Typical values for city photo hotspots: 50m to 250m.

    min_samples:
      min number of points within eps to form a cluster.
      If too high -> fewer clusters and more noise.
      If too low -> too many tiny clusters.
    """
    _require_columns(df_clean, [lat_col, lon_col])

    from sklearn.cluster import DBSCAN

    df = df_clean.copy()

    # Ensure numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    n_rows_in = int(len(df_clean))
    n_rows_used = int(len(df))

    coords_rad = _to_radians(df[lat_col].to_numpy(), df[lon_col].to_numpy())

    eps_rad = _eps_meters_to_radians(eps_meters)

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    labels = model.fit_predict(coords_rad)
    df[cluster_col] = labels.astype(int)

    # Report
    noise_points = int((labels == -1).sum())
    noise_ratio = float(noise_points / len(labels)) if len(labels) else 0.0

    # Cluster count excludes noise (-1)
    cluster_ids = sorted([cid for cid in set(labels.tolist()) if cid != -1])
    n_clusters = int(len(cluster_ids))

    sizes = (
        pd.Series(labels)
        .value_counts()
        .drop(labels=[-1], errors="ignore")
        .sort_values(ascending=False)
    )
    top10 = [(int(idx), int(val)) for idx, val in sizes.head(10).items()]

    rep = ClusterReport(
        n_rows_in=n_rows_in,
        n_rows_used=n_rows_used,
        eps_meters=float(eps_meters),
        min_samples=int(min_samples),
        n_clusters=n_clusters,
        noise_points=noise_points,
        noise_ratio=noise_ratio,
        cluster_sizes_top10=top10,
    )

    return df, rep


def print_cluster_report(rep: ClusterReport) -> None:
    print("\n" + "=" * 92)
    print("SESSION 1 — CLUSTERING REPORT (DBSCAN haversine)")
    print("=" * 92)
    print(f"Rows in:    {rep.n_rows_in:,}")
    print(f"Rows used:  {rep.n_rows_used:,}")
    print(f"eps:        {rep.eps_meters:.1f} meters")
    print(f"min_samples:{rep.min_samples}")
    print(f"Clusters:   {rep.n_clusters:,}")
    print(f"Noise:      {rep.noise_points:,} ({rep.noise_ratio * 100:.2f}%)")
    print("\nTop clusters (id -> size):")
    if rep.cluster_sizes_top10:
        for cid, size in rep.cluster_sizes_top10:
            print(f" - {cid:>4} -> {size:,}")
    else:
        print(" - No clusters found (all noise).")
    print("=" * 92 + "\n")


def save_clustered_csv(df_clustered: pd.DataFrame, path: str = "outputs/clustered.csv") -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_clustered.to_csv(path, index=False)
    return path


def make_cluster_map(
    df_clustered: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    cluster_col: str = "cluster",
    output_html: str = "outputs/map_clusters.html",
    sample_n: int = 25000,
    random_state: int = 42,
    center: Tuple[float, float] = (45.7640, 4.8357),
    zoom_start: int = 12,
) -> str:
    """
    Optional: build a Folium map where each point is colored by cluster id.
    (Folium uses Leaflet; we keep it lightweight with sampling.)
    """
    import os
    import folium

    _require_columns(df_clustered, [lat_col, lon_col, cluster_col])

    df = df_clustered.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    if sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)

    m = folium.Map(location=list(center), zoom_start=zoom_start, control_scale=True)

    # Simple palette (no manual colors required, we let folium pick via HSV mapping)
    # We'll map cluster id -> hue.
    # Noise (-1) will be gray-ish by using a fixed color.
    def color_for_cluster(cid: int) -> str:
        if cid == -1:
            return "#666666"
        # deterministic hue from cluster id
        hue = (cid * 47) % 360
        # convert hue to hex via a simple HSV -> RGB conversion using python's colorsys
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    for _, row in df.iterrows():
        cid = int(row[cluster_col])
        folium.CircleMarker(
            location=(float(row[lat_col]), float(row[lon_col])),
            radius=2.5,
            fill=True,
            color=color_for_cluster(cid),
            fill_color=color_for_cluster(cid),
            fill_opacity=0.7,
            opacity=0.7,
            popup=f"cluster={cid}",
        ).add_to(m)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    return output_html


if __name__ == "__main__":
    # Pipeline test:
    # Run from project root: python src/clustering.py
    try:
        from load_data import load_data
        from cleaning import clean_data

        df_raw, _ = load_data("../data/flickr_data2.csv")
        df_clean, _ = clean_data(df_raw)

        # You can tune eps/min_samples during Session 1 exploration
        df_clustered, rep = run_dbscan_geo(
            df_clean,
            eps_meters=120.0,
            min_samples=30,
        )
        print_cluster_report(rep)

        out_csv = save_clustered_csv(df_clustered, "outputs/clustered.csv")
        print(f"[OK] Clustered CSV saved to: {out_csv}")

        # Optional cluster map (requires folium installed)
        try:
            out_map = make_cluster_map(df_clustered, output_html="outputs/map_clusters.html")
            print(f"[OK] Cluster map saved to: {out_map}")
        except Exception as e:
            print(f"[WARN] Could not create cluster map (folium missing?): {e}")

    except Exception as e:
        print(f"[ERROR] {e}")
