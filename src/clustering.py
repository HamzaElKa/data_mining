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
    cluster_names: Optional[Dict[int, str]] = None,  # NEW: {cluster_id: poi_name}
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
        
        # Popup with POI name if available
        if cluster_names and cid in cluster_names:
            popup_text = f"<b>{cluster_names[cid]}</b><br>Cluster {cid}"
        else:
            popup_text = f"Cluster {cid}"
        
        folium.CircleMarker(
            location=(float(row[lat_col]), float(row[lon_col])),
            radius=2.5,
            fill=True,
            color=color_for_cluster(cid),
            fill_color=color_for_cluster(cid),
            fill_opacity=0.7,
            opacity=0.7,
            popup=popup_text,
        ).add_to(m)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    return output_html


def make_cluster_map_named(
    df_clustered: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    cluster_col: str = "cluster",
    name_col: str = "cluster_name",
    output_html: str = "outputs/map_clusters_named.html",
    sample_n: int = 25000,
    random_state: int = 42,
    center: Tuple[float, float] = (45.7640, 4.8357),
    zoom_start: int = 12,
) -> str:
    """
    Build a Folium map with cluster names displayed.
    
    Enhanced version that shows cluster names in popups and map controls.
    
    Parameters:
    -----------
    name_col : str
        Column containing cluster names (from text mining)
    
    Returns:
    --------
    output_html : str
        Path to saved HTML map
    """
    import os
    import folium
    from folium import plugins
    
    _require_columns(df_clustered, [lat_col, lon_col, cluster_col])
    
    # Check if name column exists
    if name_col not in df_clustered.columns:
        print(f"[WARN] Column '{name_col}' not found. Using cluster IDs instead.")
        name_col = cluster_col
    
    df = df_clustered.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    # Exclude noise
    df = df[df[cluster_col] != -1].copy()
    
    if sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)
    
    m = folium.Map(location=list(center), zoom_start=zoom_start, control_scale=True)
    
    def color_for_cluster(cid: int) -> str:
        if cid == -1:
            return "#666666"
        hue = (cid * 47) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    
    # Add markers with cluster names
    for _, row in df.iterrows():
        cid = int(row[cluster_col])
        name = str(row[name_col]) if name_col in row and pd.notna(row[name_col]) else f"Cluster {cid}"
        
        # Create popup with cluster info
        popup_text = f"""
        <b>{name}</b><br>
        ID: {cid}<br>
        Lat: {row[lat_col]:.4f}<br>
        Lon: {row[lon_col]:.4f}
        """
        
        folium.CircleMarker(
            location=(float(row[lat_col]), float(row[lon_col])),
            radius=2.5,
            fill=True,
            color=color_for_cluster(cid),
            fill_color=color_for_cluster(cid),
            fill_opacity=0.7,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=name,
        ).add_to(m)
    
    # Add a legend showing cluster names
    try:
        # Get unique clusters and their names
        cluster_legend = df[[cluster_col, name_col]].drop_duplicates().sort_values(cluster_col)
        
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: 400px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; overflow-y: scroll; border-radius: 5px; padding: 10px;">
        <b>Cluster Names</b><br><hr>
        """
        
        for _, row in cluster_legend.head(20).iterrows():
            cid = int(row[cluster_col])
            name = str(row[name_col])
            color = color_for_cluster(cid)
            legend_html += f'<i style="background:{color}; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></i> {name}<br>'
        
        legend_html += "</div>"
        
        m.get_root().html.add_child(folium.Element(legend_html))
    except Exception as e:
        print(f"[WARN] Could not add legend: {e}")
    
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    return output_html



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


# ============================================================================
# SESSION 2: HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
# ============================================================================

def run_hierarchical(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    n_clusters: int = 50,
    linkage: str = "complete",
    cluster_col: str = "cluster",
    random_state: int = 42,
    max_samples: int = 10000,  # Limit for memory management
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run Agglomerative Hierarchical Clustering on GPS coordinates.
    
    ⚠️ WARNING: Hierarchical clustering requires O(n²) memory.
    For large datasets (>10k points), we sample first, then predict labels for all points.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters (must be chosen a priori).
        Typical values for Lyon: 30-70.
    linkage : str
        Linkage method: 'complete', 'average', 'single', 'ward'
        - complete: maximum distance between clusters (good for well-separated clusters)
        - average: average distance (compromise)
        - single: minimum distance (can create chains)
        - ward: minimizes within-cluster variance (requires euclidean)
    max_samples : int
        Maximum number of samples to use for clustering (memory limit).
        If df has more rows, we sample first, cluster, then predict for all.
    
    Returns:
    --------
    df_clustered : DataFrame with new column 'cluster'
    report : dict with metrics (silhouette, etc.)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
    
    _require_columns(df_clean, [lat_col, lon_col])
    
    df = df_clean.copy()
    
    # Ensure numeric coords
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    n_rows = int(len(df))
    print(f"[Hierarchical] Dataset: {n_rows:,} points")
    
    # Convert to Cartesian coordinates (km)
    X_all = _gps_to_cartesian(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    # If dataset too large, sample first
    if n_rows > max_samples:
        print(f"[Hierarchical] Sampling {max_samples:,} points for clustering (memory constraint)...")
        df_sample = df.sample(n=max_samples, random_state=random_state)
        X_sample = _gps_to_cartesian(df_sample[lat_col].to_numpy(), df_sample[lon_col].to_numpy())
        
        # Cluster on sample
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage=linkage,
        )
        labels_sample = model.fit_predict(X_sample)
        
        # Assign all points to nearest cluster center
        # Compute cluster centers from sample
        cluster_centers = np.array([
            X_sample[labels_sample == i].mean(axis=0)
            for i in range(n_clusters)
        ])
        
        # Assign all points to nearest center
        closest, _ = pairwise_distances_argmin_min(X_all, cluster_centers)
        labels = closest
        
        print(f"[Hierarchical] Assigned all {n_rows:,} points to {n_clusters} clusters (based on sample)")
    else:
        # Small dataset: cluster directly
        print(f"[Hierarchical] Clustering all {n_rows:,} points...")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage=linkage,
        )
        labels = model.fit_predict(X_all)
    
    df[cluster_col] = labels.astype(int)
    
    # Calculate metrics
    n_clusters_found = len(set(labels))
    
    # Cluster sizes
    cluster_sizes = df[cluster_col].value_counts().sort_values(ascending=False)
    cluster_sizes_top10 = [(int(cid), int(size)) for cid, size in cluster_sizes.head(10).items()]
    
    # Metrics (on sample if too large)
    if n_rows > max_samples:
        X_metric = X_sample
        labels_metric = labels_sample
    else:
        X_metric = X_all
        labels_metric = labels
    
    silhouette = float(silhouette_score(X_metric, labels_metric))
    davies_bouldin = float(davies_bouldin_score(X_metric, labels_metric))
    
    report = {
        "algorithm": f"Hierarchical ({linkage})",
        "n_rows": n_rows,
        "n_clusters": n_clusters_found,
        "noise_points": 0,  # Hierarchical doesn't have noise
        "noise_ratio": 0.0,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "cluster_sizes_top10": cluster_sizes_top10,
    }
    
    print(f"[Hierarchical] Clusters: {n_clusters_found}, Silhouette: {silhouette:.3f}, DB: {davies_bouldin:.3f}")
    
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


# ============================================================================
# PARAMETER OPTIMIZATION FUNCTIONS
# ============================================================================

def optimize_dbscan_parameters(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    eps_range: list = None,
    min_samples_range: list = None,
    metric: str = "silhouette",
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize DBSCAN parameters using grid search.
    
    Parameters:
    -----------
    eps_range : list
        Range of eps values in meters to test (default: [40, 50, 60, 75])
    min_samples_range : list
        Range of min_samples values to test (default: [30, 50, 70])
    metric : str
        Metric to optimize: "silhouette", "davies_bouldin", or "balance"
    
    Returns:
    --------
    best_params : dict with optimal parameters
    results_df : DataFrame with all tested parameter combinations
    """
    if eps_range is None:
        eps_range = [40, 50, 60, 75]
    if min_samples_range is None:
        min_samples_range = [30, 50, 70]
    
    _require_columns(df_clean, [lat_col, lon_col])
    
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    df = df_clean.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    coords_rad = _to_radians(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    results = []
    
    print(f"\nOptimizing DBSCAN parameters (metric={metric})...")
    print(f"Testing {len(eps_range)} × {len(min_samples_range)} = {len(eps_range)*len(min_samples_range)} combinations")
    
    for i, eps_m in enumerate(eps_range):
        for j, min_samp in enumerate(min_samples_range):
            print(f"  [{i*len(min_samples_range)+j+1}/{len(eps_range)*len(min_samples_range)}] eps={eps_m}m, min_samples={min_samp}...", end=' ', flush=True)
            
            eps_rad = _eps_meters_to_radians(eps_m)
            
            model = DBSCAN(
                eps=eps_rad,
                min_samples=min_samp,
                metric="haversine",
                algorithm="ball_tree",
                n_jobs=-1,
            )
            
            labels = model.fit_predict(coords_rad)
            
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            noise_points = int((labels == -1).sum())
            noise_ratio = float(noise_points / len(labels)) if len(labels) else 0.0
            
            # Calculate metrics
            silhouette_val = None
            davies_bouldin_val = None
            
            mask_no_noise = labels != -1
            if mask_no_noise.sum() > 1 and n_clusters > 1:
                try:
                    silhouette_val = float(silhouette_score(coords_rad[mask_no_noise], labels[mask_no_noise], metric='haversine'))
                    davies_bouldin_val = float(davies_bouldin_score(coords_rad[mask_no_noise], labels[mask_no_noise]))
                except:
                    pass
            
            print(f"clusters={n_clusters}, noise={noise_ratio:.1%}")
            
            results.append({
                "eps_meters": eps_m,
                "min_samples": min_samp,
                "n_clusters": n_clusters,
                "noise_points": noise_points,
                "noise_ratio": noise_ratio,
                "silhouette_score": silhouette_val,
                "davies_bouldin_index": davies_bouldin_val,
            })
    
    results_df = pd.DataFrame(results)
    
    # Select best parameters based on metric
    if metric == "silhouette":
        # Higher silhouette is better
        valid = results_df[results_df["silhouette_score"].notna()]
        if len(valid) > 0:
            best_idx = valid["silhouette_score"].idxmax()
        else:
            # Fallback: balance between clusters and noise
            best_idx = results_df[(results_df['n_clusters'] > 20) & (results_df['n_clusters'] < 100)]['noise_ratio'].idxmin()
    elif metric == "davies_bouldin":
        # Lower DB is better
        valid = results_df[results_df["davies_bouldin_index"].notna()]
        if len(valid) > 0:
            best_idx = valid["davies_bouldin_index"].idxmin()
        else:
            best_idx = results_df[(results_df['n_clusters'] > 20) & (results_df['n_clusters'] < 100)]['noise_ratio'].idxmin()
    else:  # "balance"
        # Find sweet spot: 30-60 clusters, <70% noise
        valid = results_df[(results_df['n_clusters'] >= 30) & (results_df['n_clusters'] <= 60) & (results_df['noise_ratio'] < 0.7)]
        if len(valid) > 0:
            best_idx = valid['silhouette_score'].idxmax() if valid['silhouette_score'].notna().any() else valid.index[0]
        else:
            best_idx = results_df[(results_df['n_clusters'] > 20) & (results_df['n_clusters'] < 100)]['noise_ratio'].idxmin()
    
    best_row = results_df.loc[best_idx]
    best_params = {
        "eps_meters": float(best_row["eps_meters"]),
        "min_samples": int(best_row["min_samples"]),
    }
    
    print(f"✅ Best parameters found: eps={best_params['eps_meters']}m, min_samples={best_params['min_samples']}")
    
    return best_params, results_df


def optimize_kmeans_parameters(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    k_range: list = None,
) -> Tuple[int, pd.DataFrame]:
    """
    Optimize K-Means number of clusters using elbow + silhouette methods.
    
    Parameters:
    -----------
    k_range : list
        Range of K values to test (default: [20, 30, 40, 50, 60, 70, 80])
    
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters
    results_df : DataFrame with metrics for each K
    """
    if k_range is None:
        k_range = [20, 30, 40, 50, 60, 70, 80]
    
    _require_columns(df_clean, [lat_col, lon_col])
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    df = df_clean.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    X = _gps_to_cartesian(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    results = []
    
    print(f"\nOptimizing K-Means (testing K={k_range[0]}-{k_range[-1]})...")
    
    for i, k in enumerate(k_range):
        print(f"  [{i+1}/{len(k_range)}] K={k}...", end=' ', flush=True)
        
        model = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300)
        labels = model.fit_predict(X)
        
        inertia = float(model.inertia_)
        silhouette = float(silhouette_score(X, labels))
        
        print(f"silhouette={silhouette:.4f}")
        
        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette,
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal K using silhouette (higher is better)
    optimal_k = int(results_df.loc[results_df["silhouette_score"].idxmax(), "k"])
    
    print(f"✅ Optimal K (max silhouette): {optimal_k}")
    
    return optimal_k, results_df


def optimize_hdbscan_parameters(
    df_clean: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    min_cluster_size_range: list = None,
    min_samples_range: list = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize HDBSCAN parameters using grid search.
    
    Parameters:
    -----------
    min_cluster_size_range : list
        Range of min_cluster_size values (default: [40, 50, 60])
    min_samples_range : list
        Range of min_samples values (default: [20, 50])
    
    Returns:
    --------
    best_params : dict with optimal parameters
    results_df : DataFrame with all tested combinations
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan not installed")
    
    if min_cluster_size_range is None:
        min_cluster_size_range = [40, 50, 60]
    if min_samples_range is None:
        min_samples_range = [20, 50]
    
    _require_columns(df_clean, [lat_col, lon_col])
    
    from sklearn.metrics import silhouette_score
    
    df = df_clean.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    
    coords_rad = _to_radians(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    
    results = []
    
    print(f"\nOptimizing HDBSCAN parameters...")
    print(f"Testing {len(min_cluster_size_range)} × {len(min_samples_range)} = {len(min_cluster_size_range)*len(min_samples_range)} combinations")
    
    for i, min_csize in enumerate(min_cluster_size_range):
        for j, min_samp in enumerate(min_samples_range):
            print(f"  [{i*len(min_samples_range)+j+1}/{len(min_cluster_size_range)*len(min_samples_range)}] min_cluster_size={min_csize}, min_samples={min_samp}...", end=' ', flush=True)
            
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_csize,
                min_samples=min_samp,
                metric="haversine",
                core_dist_n_jobs=-1,
            )
            
            labels = model.fit_predict(coords_rad)
            
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            noise_points = int((labels == -1).sum())
            noise_ratio = float(noise_points / len(labels)) if len(labels) else 0.0
            
            # Calculate metrics
            silhouette_val = None
            
            mask_no_noise = labels != -1
            if mask_no_noise.sum() > 1 and n_clusters > 1:
                try:
                    silhouette_val = float(silhouette_score(coords_rad[mask_no_noise], labels[mask_no_noise], metric='haversine'))
                except:
                    pass
            
            print(f"clusters={n_clusters}, noise={noise_ratio:.1%}")
            
            results.append({
                "min_cluster_size": min_csize,
                "min_samples": min_samp,
                "n_clusters": n_clusters,
                "noise_points": noise_points,
                "noise_ratio": noise_ratio,
                "silhouette_score": silhouette_val,
            })
    
    results_df = pd.DataFrame(results)
    
    # Select best: balance clusters and noise
    valid = results_df[(results_df['n_clusters'] >= 30) & (results_df['n_clusters'] <= 60) & (results_df['noise_ratio'] < 0.75)]
    if len(valid) > 0:
        best_idx = valid['silhouette_score'].idxmax() if valid['silhouette_score'].notna().any() else valid.index[0]
    else:
        best_idx = results_df[(results_df['n_clusters'] > 20) & (results_df['n_clusters'] < 100)].index[0]
    
    best_row = results_df.loc[best_idx]
    best_params = {
        "min_cluster_size": int(best_row["min_cluster_size"]),
        "min_samples": int(best_row["min_samples"]),
    }
    
    print(f"✅ Best parameters found: min_cluster_size={best_params['min_cluster_size']}, min_samples={best_params['min_samples']}")
    
    return best_params, results_df
