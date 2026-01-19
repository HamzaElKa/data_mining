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
    deduplicate_coords: bool = True,
    coord_precision: int = 4,
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
      
    deduplicate_coords:
      If True, keep only 1 photo per unique GPS coordinate (rounded to coord_precision).
      This prevents over-representation of heavily photographed exact spots.
      Recommended: True for balanced clustering.
      
    coord_precision:
      Number of decimal places for coordinate rounding (4 = ~11m precision).
    """
    _require_columns(df_clean, [lat_col, lon_col])

    from sklearn.cluster import DBSCAN

    df = df_clean.copy()

    # Ensure numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    n_rows_in = int(len(df_clean))
    
    # Optional: deduplicate coordinates to prevent over-representation
    if deduplicate_coords:
        # Round coords and keep first occurrence per unique coordinate
        df['_lat_round'] = df[lat_col].round(coord_precision)
        df['_lon_round'] = df[lon_col].round(coord_precision)
        df_sample = df.drop_duplicates(subset=['_lat_round', '_lon_round'], keep='first').copy()
        df_sample = df_sample.drop(columns=['_lat_round', '_lon_round'])
    else:
        df_sample = df.copy()
    
    n_rows_used = int(len(df_sample))

    coords_rad = _to_radians(df_sample[lat_col].to_numpy(), df_sample[lon_col].to_numpy())

    eps_rad = _eps_meters_to_radians(eps_meters)

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    labels = model.fit_predict(coords_rad)
    
    # Assign cluster labels back to df_sample
    df_sample[cluster_col] = labels.astype(int)
    
    # If we deduplicated, we need to propagate labels back to original df
    if deduplicate_coords:
        # Create mapping from rounded coords to cluster labels
        df_sample['_lat_round'] = df_sample[lat_col].round(coord_precision)
        df_sample['_lon_round'] = df_sample[lon_col].round(coord_precision)
        
        # Map original df
        df['_lat_round'] = df[lat_col].round(coord_precision)
        df['_lon_round'] = df[lon_col].round(coord_precision)
        
        # Merge labels
        coord_to_cluster = df_sample[['_lat_round', '_lon_round', cluster_col]].drop_duplicates()
        df = df.drop(columns=[cluster_col], errors='ignore')
        df = df.merge(coord_to_cluster, on=['_lat_round', '_lon_round'], how='left')
        df[cluster_col] = df[cluster_col].fillna(-1).astype(int)
        df = df.drop(columns=['_lat_round', '_lon_round'])
    else:
        df = df_sample

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

    # Exclure le bruit (cluster = -1) de la visualisation
    df = df[df[cluster_col] != -1].copy()

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


# ============================================================================
# SESSION 2: K-MEANS CLUSTERING
# ============================================================================

def _gps_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Convert GPS (lat, lon) to approximate Cartesian (X, Y) in km.
    Uses simple equirectangular projection centered on Lyon.
    Good enough for local area (~50km radius).
    """
    # Lyon center as reference
    lat_center = 45.7640
    lon_center = 4.8357
    
    # Earth radius in km
    R = 6371.0
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_center_rad = np.radians(lat_center)
    lon_center_rad = np.radians(lon_center)
    
    # Approximate cartesian coordinates in km
    X = R * (lon_rad - lon_center_rad) * np.cos(lat_center_rad)
    Y = R * (lat_rad - lat_center_rad)
    
    return np.column_stack([X, Y])


def run_kmeans(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    n_clusters: int = 50,
    cluster_col: str = "cluster",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run K-Means clustering on GPS coordinates.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters (must be chosen a priori).
        Typical values for Lyon: 30-70.
    
    Returns:
    --------
    df_clustered : DataFrame with new column 'cluster'
    report : dict with metrics (inertia, silhouette, etc.)
    """
    _require_columns(df_clean, [lat_col, lon_col])
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    df = df_clean.copy()
    
    # Ensure numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    n_rows = int(len(df))
    
    # Convert GPS to Cartesian (km)
    X = _gps_to_cartesian(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    # Run K-Means
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    
    labels = model.fit_predict(X)
    df[cluster_col] = labels.astype(int)
    
    # Calculate metrics
    inertia = float(model.inertia_)
    silhouette = float(silhouette_score(X, labels))
    davies_bouldin = float(davies_bouldin_score(X, labels))
    
    # Cluster sizes
    cluster_sizes = df[cluster_col].value_counts().sort_values(ascending=False)
    cluster_sizes_top10 = [(int(cid), int(size)) for cid, size in cluster_sizes.head(10).items()]
    
    report = {
        "algorithm": "K-Means",
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "inertia": inertia,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "cluster_sizes_top10": cluster_sizes_top10,
    }
    
    return df, report


def find_optimal_k(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    k_range: range = range(20, 81, 10),
    random_state: int = 42,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Test multiple K values and return metrics for elbow/silhouette analysis.
    
    Returns:
    --------
    k_values : list of int
    inertias : list of float (for elbow plot)
    silhouettes : list of float (for silhouette plot)
    """
    _require_columns(df_clean, [lat_col, lon_col])
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    df = df_clean.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    X = _gps_to_cartesian(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    k_values = []
    inertias = []
    silhouettes = []
    
    for k in k_range:
        print(f"Testing K-Means with K={k}...")
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = model.fit_predict(X)
        
        k_values.append(k)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))
    
    return k_values, inertias, silhouettes


# ============================================================================
# SESSION 2: HDBSCAN CLUSTERING
# ============================================================================

def run_hdbscan(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    min_cluster_size: int = 50,
    min_samples: int = 50,
    cluster_col: str = "cluster",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run HDBSCAN clustering on GPS coordinates.
    
    HDBSCAN advantages over DBSCAN:
    - Handles varying density (Bellecour dense vs Terreaux less dense)
    - Hierarchical structure
    - More robust to parameter choice
    
    Parameters:
    -----------
    min_cluster_size : int
        Minimum number of points to form a cluster.
        Similar to min_samples in DBSCAN but more stable.
        
    min_samples : int
        How conservative the clustering is.
        Higher = more points labeled as noise.
    
    Returns:
    --------
    df_clustered : DataFrame with new column 'cluster'
    report : dict with metrics
    """
    _require_columns(df_clean, [lat_col, lon_col])
    
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    df = df_clean.copy()
    
    # Ensure numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    n_rows = int(len(df))
    
    # Convert to radians for haversine
    coords_rad = _to_radians(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    # Run HDBSCAN
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        core_dist_n_jobs=-1,
    )
    
    labels = model.fit_predict(coords_rad)
    df[cluster_col] = labels.astype(int)
    
    # Calculate metrics
    noise_points = int((labels == -1).sum())
    noise_ratio = float(noise_points / len(labels)) if len(labels) else 0.0
    
    cluster_ids = [cid for cid in set(labels.tolist()) if cid != -1]
    n_clusters = len(cluster_ids)
    
    # Cluster sizes
    cluster_sizes = df[df[cluster_col] != -1][cluster_col].value_counts().sort_values(ascending=False)
    cluster_sizes_top10 = [(int(cid), int(size)) for cid, size in cluster_sizes.head(10).items()]
    
    # Metrics (exclude noise for silhouette/DB)
    mask_no_noise = labels != -1
    if mask_no_noise.sum() > 1 and n_clusters > 1:
        X_no_noise = coords_rad[mask_no_noise]
        labels_no_noise = labels[mask_no_noise]
        silhouette = float(silhouette_score(X_no_noise, labels_no_noise, metric='haversine'))
        davies_bouldin = float(davies_bouldin_score(X_no_noise, labels_no_noise))
    else:
        silhouette = None
        davies_bouldin = None
    
    report = {
        "algorithm": "HDBSCAN",
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "noise_points": noise_points,
        "noise_ratio": noise_ratio,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "cluster_sizes_top10": cluster_sizes_top10,
    }
    
    return df, report
