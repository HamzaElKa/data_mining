# src/main.py
# SESSION 2 â€” End-to-end pipeline:
# load_data -> cleaning -> 3 clustering algos (DBSCAN/KMeans/HDBSCAN) + tuning ->
# visualization on map -> text mining (TF-IDF keywords) to describe clusters

from __future__ import annotations

import os
import traceback
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from load_data import load_data, print_report
from cleaning import clean_data, print_cleaning_report
from visualization import create_map, MapConfig

from clustering import (
    run_dbscan_geo,
    run_kmeans,
    run_hdbscan,
    run_hierarchical,
    make_cluster_map,
    save_clustered_csv,
)

from text_mining import (
    preprocess_text,
    extract_cluster_descriptions,
    print_cluster_descriptions,
    save_descriptions_csv,
)

# We still reuse comparison.py but in a SAFE way (avoid silhouette crash)
from comparison import save_comparison_csv


def _ensure_outputs() -> None:
    os.makedirs("outputs", exist_ok=True)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x.upper() in {"N/A", ""}:
                return None
        return float(x)
    except Exception:
        return None


def _count_effective_clusters(labels: pd.Series) -> int:
    # excludes noise (-1)
    s = set(int(v) for v in labels.dropna().astype(int).tolist())
    s.discard(-1)
    return len(s)


def _safe_internal_metrics_cartesian(df: pd.DataFrame, cluster_col: str = "cluster") -> Tuple[Optional[float], Optional[float]]:
    """
    Compute silhouette + Davies-Bouldin safely on non-noise points.
    Returns (silhouette, davies_bouldin) or (None, None) if not computable.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from clustering import _gps_to_cartesian

    if cluster_col not in df.columns:
        return None, None

    mask = df[cluster_col] != -1
    if mask.sum() < 3:
        return None, None

    n_clusters = _count_effective_clusters(df.loc[mask, cluster_col])
    if n_clusters < 2:
        return None, None

    X = _gps_to_cartesian(df.loc[mask, "lat"].to_numpy(), df.loc[mask, "long"].to_numpy())
    labels = df.loc[mask, cluster_col].to_numpy()

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None

    try:
        dbi = float(davies_bouldin_score(X, labels))
    except Exception:
        dbi = None

    return sil, dbi


def _summarize_report(name: str, df: pd.DataFrame, cluster_col: str = "cluster") -> Dict[str, Any]:
    n_rows = int(len(df))
    noise = int((df[cluster_col] == -1).sum()) if cluster_col in df.columns else None
    noise_ratio = (noise / n_rows) if (noise is not None and n_rows > 0) else None
    n_clusters = _count_effective_clusters(df[cluster_col]) if cluster_col in df.columns else None

    sil, dbi = _safe_internal_metrics_cartesian(df, cluster_col=cluster_col)

    # top cluster size (excluding noise)
    top_cluster_size = 0
    if cluster_col in df.columns:
        vc = df[df[cluster_col] != -1][cluster_col].value_counts()
        if len(vc) > 0:
            top_cluster_size = int(vc.iloc[0])

    return {
        "algorithm": name,
        "n_rows": n_rows,
        "n_clusters": n_clusters if n_clusters is not None else "N/A",
        "noise_points": noise if noise is not None else "N/A",
        "noise_ratio": f"{(noise_ratio*100):.1f}%" if noise_ratio is not None else "N/A",
        "top_cluster_size": top_cluster_size if top_cluster_size is not None else "N/A",
        "silhouette_score": f"{sil:.4f}" if sil is not None else "N/A",
        "davies_bouldin_index": f"{dbi:.4f}" if dbi is not None else "N/A",
    }


def _write_algo_outputs(
    df_algo: pd.DataFrame,
    algo_name: str,
    tag: str,
    *,
    make_map: bool = True,
    cluster_names: Optional[Dict[int, str]] = None,  # NEW: POI titles
) -> Dict[str, str]:
    """
    Save clustered CSV + (optional) map HTML for a given algorithm.
    Returns dict of produced file paths.
    """
    paths: Dict[str, str] = {}

    csv_path = f"outputs/clustered_{tag}.csv"
    save_clustered_csv(df_algo, csv_path)
    paths["csv"] = csv_path

    if make_map:
        html_path = f"outputs/map_clusters_{tag}.html"
        try:
            out = make_cluster_map(
                df_algo,
                output_html=html_path,
                sample_n=25000,
                random_state=42,
                center=(45.7640, 4.8357),
                zoom_start=12,
                cluster_names=cluster_names,  # NEW: pass POI titles
            )
            paths["map"] = out
        except Exception as e:
            print(f"[WARN] Could not generate cluster map for {algo_name}: {e}")

    return paths


def main() -> None:
    _ensure_outputs()

    # ---- Dataset path (same style as Session 1 main) ----
    # Adjust if needed: in your repo, you used "../flickr_data2.csv"
    csv_path = "../flickr_data2.csv"  # Fichier Ã  la racine du projet (comme ton main Session 1)
    print(f"[INFO] Using CSV: {csv_path}")

    # ---- 1) Load + explore ----
    df_raw, rep_raw = load_data(csv_path)
    print_report(rep_raw)

    # ---- 2) Clean (Session 2: keep it strict & consistent) ----
    # You can enable drop_duplicate_coords if you want less over-representation.
    df_clean, rep_clean = clean_data(
        df_raw,
        drop_duplicate_coords=False,   # keep False unless you want stronger balancing
        strict_time_validation=False,
    )
    print_cleaning_report(rep_clean)

    # ---- 3) Base map (cleaned points) ----
    map_cfg = MapConfig(
        output_html="outputs/map_cleaned_session2.html",
        sample_n=20000,
        random_state=42,
        center=(45.7640, 4.8357),
        zoom_start=12,
        max_markers=20000,
    )
    out_map = create_map(df_clean, cfg=map_cfg)
    print(f"[OK] Cleaned points map saved to: {out_map}")

    # =========================================================================
    # 4) Try 3 clustering algorithms + "optimized" parameters (session 2 goal)
    # =========================================================================
    # IMPORTANT:
    # - DBSCAN eps/min_samples must be tuned for "meaningful" POIs.
    # - KMeans K must be chosen (use your prior elbow/silhouette later if needed).
    # - HDBSCAN requires `pip install hdbscan` (optional).
    #
    # These defaults are good "session 2 demo" values. Adapt fast if needed.
    dbscan_params = dict(
        eps_meters=50.0,          # try 50, 80, 120 depending on density
        min_samples=50,           # try 20..80
        deduplicate_coords=True,  # recommended for better POI balance
        coord_precision=4,
        cluster_col="cluster",
    )
    kmeans_params = dict(
        n_clusters=50,            # typical 30..70
        cluster_col="cluster",
        random_state=42,
    )
    hdbscan_params = dict(
        min_cluster_size=250,     # Augmenté pour éviter sur-fragmentation (try 200..350)
        min_samples=50,           # Aligné avec DBSCAN
        cluster_col="cluster",
    )
    hierarchical_params = dict(
        n_clusters=50,            # Same as K-Means for comparison
        linkage="complete",       # Try: 'complete', 'average', 'single'
        cluster_col="cluster",
        random_state=42,
    )

    results_rows = []
    produced_files: Dict[str, Dict[str, str]] = {}

    # ---- 4.1 DBSCAN ---- (SKIPPED)
    # print("\n" + "=" * 90)
    # print("[1/3] DBSCAN")
    # print("=" * 90)
    # df_dbscan, rep_db = run_dbscan_geo(df_clean, **dbscan_params)
    # row_db = _summarize_report("DBSCAN", df_dbscan)
    # row_db["parameters"] = f"eps={dbscan_params['eps_meters']}m, min_samples={dbscan_params['min_samples']}"
    # results_rows.append(row_db)
    # produced_files["dbscan"] = _write_algo_outputs(df_dbscan, "DBSCAN", "dbscan")
    df_dbscan = None

    # ---- 4.2 K-Means ---- (SKIPPED)
    # print("\n" + "=" * 90)
    # print("[2/3] K-Means")
    # print("=" * 90)
    # df_kmeans, rep_km = run_kmeans(df_clean, **kmeans_params)
    # row_km = _summarize_report("K-Means", df_kmeans)
    # row_km["parameters"] = f"n_clusters={kmeans_params['n_clusters']}"
    # results_rows.append(row_km)
    # produced_files["kmeans"] = _write_algo_outputs(df_kmeans, "K-Means", "kmeans")
    df_kmeans = None

    # ---- 4.3 HDBSCAN ---- (SKIPPED)
    # print("\n" + "=" * 90)
    # print("[1/2] HDBSCAN (with min_cluster_size=250)")
    # print("=" * 90)
    # df_hdb = None
    # try:
    #     df_hdb, rep_hdb = run_hdbscan(df_clean, **hdbscan_params)
    #     row_hdb = _summarize_report("HDBSCAN", df_hdb)
    #     row_hdb["parameters"] = f"min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params['min_samples']}"
    #     results_rows.append(row_hdb)
    #     produced_files["hdbscan"] = _write_algo_outputs(df_hdb, "HDBSCAN", "hdbscan")
    # except Exception as e:
    #     print(f"[WARN] HDBSCAN skipped (not installed or error): {e}")
    #     results_rows.append({
    #         "algorithm": "HDBSCAN",
    #         "n_rows": int(len(df_clean)),
    #         "n_clusters": "N/A",
    #         "noise_points": "N/A",
    #         "noise_ratio": "N/A",
    #         "top_cluster_size": "N/A",
    #         "silhouette_score": "N/A",
    #         "davies_bouldin_index": "N/A",
    #         "parameters": "Not installed / error",
    #     })
    df_hdb = None

    # ---- 4.4 Hierarchical (Agglomerative) ----
    print("\n" + "=" * 90)
    print("[1/1] Hierarchical Clustering (Agglomerative)")
    print("=" * 90)
    df_hierarchical = None
    try:
        df_hierarchical, rep_hier = run_hierarchical(df_clean, **hierarchical_params)
        row_hier = _summarize_report("Hierarchical", df_hierarchical)
        row_hier["parameters"] = f"n_clusters={hierarchical_params['n_clusters']}, linkage={hierarchical_params['linkage']}"
        results_rows.append(row_hier)
        produced_files["hierarchical"] = _write_algo_outputs(df_hierarchical, "Hierarchical", "hierarchical")
    except Exception as e:
        print(f"[WARN] Hierarchical skipped (error): {e}")
        results_rows.append({
            "algorithm": "Hierarchical",
            "n_rows": int(len(df_clean)),
            "n_clusters": "N/A",
            "noise_points": "N/A",
            "noise_ratio": "N/A",
            "top_cluster_size": "N/A",
            "silhouette_score": "N/A",
            "davies_bouldin_index": "N/A",
            "parameters": "Error",
        })

    # ---- 4.4 Save comparison metrics ----
    comparison_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 90)
    print("SESSION 2 â€” METRICS COMPARISON (SAFE)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    metrics_csv = "outputs/comparison_metrics_session2.csv"
    save_comparison_csv(comparison_df, metrics_csv)
    print(f"\n[OK] Metrics table saved to: {metrics_csv}")

    # =========================================================================
    # 5) Text pattern mining (TF-IDF) to describe clusters
    # =========================================================================
    # Choose the algorithm for text mining:
    # Priority: hierarchical > hdbscan > kmeans > dbscan
    if df_hierarchical is not None:
        df_for_text = df_hierarchical
        algo_for_text = "hierarchical"
    elif df_hdb is not None:
        df_for_text = df_hdb
        algo_for_text = "hdbscan"
    elif df_kmeans is not None:
        df_for_text = df_kmeans
        algo_for_text = "kmeans"
    else:
        df_for_text = df_dbscan
        algo_for_text = "dbscan"

    print("\n" + "=" * 90)
    print(f"SESSION 2 — TEXT MINING (TF-IDF) on {algo_for_text.upper()}")
    print("=" * 90)

    # Force text preprocessing (unidecode, spam removal, etc.)
    # Even if 'text' column already exists, reprocess it for TF-IDF
    print("[Text Mining] Preprocessing text (unidecode, spam removal)...")
    df_for_text = preprocess_text(
        df_for_text, 
        tags_col="tags",
        title_col="title",
        text_col="text"
    )

    descriptions = extract_cluster_descriptions(
        df_for_text,
        cluster_col="cluster",
        text_col="text",
        top_n_keywords=10,
        min_df=2,
        max_df=0.8,
    )

    if not descriptions:
        print("[WARN] No cluster descriptions produced (maybe all noise or empty text).")
    else:
        print_cluster_descriptions(descriptions, top_n=12)
        desc_csv = "outputs/cluster_descriptions_tfidf.csv"
        save_descriptions_csv(descriptions, desc_csv)
        print(f"\n[OK] TF-IDF cluster descriptions saved to: {desc_csv}")
        
        # Generate wordclouds for top 5 clusters (visual validation)
        print("\n[Text Mining] Generating wordclouds for top 5 clusters...")
        from text_mining import create_wordcloud_for_cluster
        for i, desc in enumerate(descriptions[:5]):
            try:
                out_img = create_wordcloud_for_cluster(
                    df_for_text,
                    cluster_id=desc.cluster_id,
                    text_col="text"
                )
                if out_img:
                    print(f"  ✓ Cluster {desc.cluster_id}: {out_img}")
            except Exception as e:
                print(f"  ✗ Cluster {desc.cluster_id}: {e}")
        
        # =====================================================================
        # NEW: Re-generate map with POI titles in popups
        # =====================================================================
        print("\n[Text Mining] Regenerating map with POI titles...")
        cluster_names_dict = {desc.cluster_id: desc.cluster_title for desc in descriptions}
        
        try:
            map_with_titles = make_cluster_map(
                df_for_text,
                output_html=f"outputs/map_clusters_{algo_for_text}_with_titles.html",
                sample_n=25000,
                random_state=42,
                center=(45.7640, 4.8357),
                zoom_start=12,
                cluster_names=cluster_names_dict,
            )
            print(f"  ✓ Map with POI titles: {map_with_titles}")
        except Exception as e:
            print(f"  ✗ Failed to generate map with titles: {e}")

    # =========================================================================
    # DONE â€” recap
    # =========================================================================
    print("\nâœ… SESSION 2 PIPELINE FINISHED.")
    print("\nOpen these outputs:")
    print("- outputs/map_cleaned_session2.html")
    print("- outputs/map_clusters_dbscan.html")
    print("- outputs/map_clusters_kmeans.html")
    print("- outputs/map_clusters_hdbscan.html (if generated)")
    print("- outputs/comparison_metrics_session2.csv")
    print("- outputs/cluster_descriptions_tfidf.csv")
    print("- outputs/clustered_dbscan.csv / clustered_kmeans.csv / clustered_hdbscan.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()