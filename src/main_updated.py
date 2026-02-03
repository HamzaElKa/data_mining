# src/main.py
# End-to-end pipeline: load_data -> cleaning -> visualization -> multi-algorithm clustering
# Tests: DBSCAN (spatial), K-Means (spatial), Temporal K-Means

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
    run_temporal_kmeans,
    run_temporal_hdbscan,
    analyze_temporal_clusters,
    make_cluster_map,
    save_clustered_csv,
    print_cluster_report,
    ClusterReport,
)


def print_temporal_report(report: Dict) -> None:
    """Print temporal clustering report."""
    print("\n" + "=" * 92)
    print("TEMPORAL CLUSTERING REPORT")
    print("=" * 92)
    print(f"Algorithm:     {report['algorithm']}")
    print(f"Rows:          {report['n_rows']:,}")
    print(f"Clusters:      {report['n_clusters']}")
    print(f"Features:      {', '.join(report['features_used'])}")
    print(f"Silhouette:    {report['silhouette_score']:.4f}" if report.get('silhouette_score') else "Silhouette:    N/A")
    if report.get('noise_points') is not None:
        print(f"Noise:         {report['noise_points']:,} ({report['noise_ratio']*100:.1f}%)")
    print("\nTop clusters (id -> size):")
    if report.get('cluster_sizes'):
        for cid, size in report['cluster_sizes'][:10]:
            print(f" - {cid:>4} -> {size:,}")
    print("=" * 92 + "\n")


def main() -> None:
    # ---- Paths (run from project root or src/) ----
    csv_path = "../flickr_data2.csv"

    # ========== STEP 1: Load + Explore ==========
    print("\n" + "="*80)
    print("[STEP 1/8] Loading data...")
    print("="*80)
    df_raw, rep_raw = load_data(csv_path)
    print_report(rep_raw)

    # ---- STEP 2: Clean ----
    print("\n" + "="*80)
    print("[STEP 2/8] Cleaning data...")
    print("="*80)
    df_clean, rep_clean = clean_data(df_raw)
    print_cleaning_report(rep_clean)

    # ---- STEP 3: Map (sampled) ----
    print("\n" + "="*80)
    print("[STEP 3/8] Creating cleaned data map...")
    print("="*80)
    os.makedirs("outputs", exist_ok=True)
    map_cfg = MapConfig(
        output_html="outputs/map_cleaned_session2.html",
        sample_n=20000,
        random_state=42,
        center=(45.7640, 4.8357),
        zoom_start=12,
        max_markers=20000,
    )
    out_map = create_map(df_clean, cfg=map_cfg)
    print(f"✅ Map saved to: {out_map}")

    # ---- STEP 4: SPATIAL CLUSTERING - DBSCAN ----

    print("\n" + "="*80)
    print("[STEP 4/8] SPATIAL CLUSTERING - Algorithm 1: DBSCAN")
    print("="*80)
    
    df_dbscan, rep_dbscan = run_dbscan_geo(
        df_clean,
        eps_meters=120.0,
        min_samples=50,
        cluster_col="cluster",
        deduplicate_coords=True,
        coord_precision=4,
    )
    print_cluster_report(rep_dbscan)
    
    # Save DBSCAN results
    out_csv_dbscan = save_clustered_csv(df_dbscan, "outputs/clustered_dbscan_main.csv")
    print(f"✅ DBSCAN CSV saved to: {out_csv_dbscan}")
    
    try:
        out_map_dbscan = make_cluster_map(
            df_dbscan,
            output_html="outputs/map_dbscan_main.html",
            sample_n=25000,
            random_state=42,
        )
        print(f"✅ DBSCAN map saved to: {out_map_dbscan}")
    except Exception as e:
        print(f"⚠️  DBSCAN map failed: {e}")

    # ---- STEP 5: SPATIAL CLUSTERING - K-MEANS ----

    print("\n" + "="*80)
    print("[STEP 5/8] SPATIAL CLUSTERING - Algorithm 2: K-Means")
    print("="*80)
    
    df_kmeans, rep_kmeans = run_kmeans(
        df_clean,
        n_clusters=45,
        cluster_col="cluster",
        random_state=42,
    )
    
    print("\n" + "=" * 92)
    print("K-MEANS CLUSTERING REPORT")
    print("=" * 92)
    print(f"Algorithm:  {rep_kmeans['algorithm']}")
    print(f"Rows:       {rep_kmeans['n_rows']:,}")
    print(f"Clusters:   {rep_kmeans['n_clusters']}")
    print(f"Inertia:    {rep_kmeans['inertia']:.2f}")
    print(f"Silhouette: {rep_kmeans['silhouette_score']:.4f}")
    print(f"Davies-Bouldin: {rep_kmeans['davies_bouldin_index']:.4f}")
    print("\nTop clusters (id -> size):")
    for cid, size in rep_kmeans['cluster_sizes_top10'][:10]:
        print(f" - {cid:>4} -> {size:,}")
    print("=" * 92 + "\n")
    
    # Save K-Means results
    out_csv_kmeans = save_clustered_csv(df_kmeans, "outputs/clustered_kmeans_main.csv")
    print(f"✅ K-Means CSV saved to: {out_csv_kmeans}")
    
    try:
        out_map_kmeans = make_cluster_map(
            df_kmeans,
            output_html="outputs/map_kmeans_main.html",
            sample_n=25000,
            random_state=42,
        )
        print(f"✅ K-Means map saved to: {out_map_kmeans}")
    except Exception as e:
        print(f"⚠️  K-Means map failed: {e}")

    # ---- STEP 6: TEMPORAL CLUSTERING ----

    print("\n" + "="*80)
    print("[STEP 6/8] TEMPORAL CLUSTERING - Algorithm 3: Temporal K-Means")
    print("="*80)
    
    df_temporal, rep_temporal = run_temporal_kmeans(
        df_clean,
        datetime_col="datetime_taken",
        n_clusters=6,
        cluster_col="temporal_cluster",
        features=['month', 'day_of_week', 'season', 'hour_bucket'],
        random_state=42,
    )
    print_temporal_report(rep_temporal)
    
    # Analyze temporal clusters
    temporal_summary = analyze_temporal_clusters(
        df_temporal,
        datetime_col="datetime_taken",
        cluster_col="temporal_cluster",
    )
    
    print("\n📅 TEMPORAL CLUSTER DESCRIPTIONS:")
    print("=" * 92)
    for _, row in temporal_summary.iterrows():
        print(f"{row['description']}")
    print("=" * 92 + "\n")
    
    # Save temporal results
    out_csv_temporal = save_clustered_csv(df_temporal, "outputs/clustered_temporal_main.csv")
    print(f"✅ Temporal CSV saved to: {out_csv_temporal}")
    
    temporal_summary.to_csv("outputs/temporal_clusters_summary.csv", index=False)
    print(f"✅ Temporal summary saved to: outputs/temporal_clusters_summary.csv")

    # ---- STEP 7: Generate comparison report ----

    print("\n" + "="*80)
    print("[STEP 7/8] Generating algorithm comparison report...")
    print("="*80)
    
    comparison_data = {
        "Algorithm": [
            "DBSCAN (Spatial)",
            "K-Means (Spatial)", 
            "Temporal K-Means"
        ],
        "Type": ["Spatial", "Spatial", "Temporal"],
        "N_Clusters": [
            rep_dbscan.n_clusters,
            rep_kmeans['n_clusters'],
            rep_temporal['n_clusters']
        ],
        "Silhouette": [
            "N/A",
            f"{rep_kmeans['silhouette_score']:.4f}",
            f"{rep_temporal['silhouette_score']:.4f}"
        ],
        "Noise/Other": [
            f"{rep_dbscan.noise_ratio*100:.1f}% noise",
            "0% noise",
            f"{rep_temporal.get('noise_ratio', 0)*100:.1f}% noise"
        ],
        "Strength": [
            "Finds dense hotspots, flexible",
            "Balanced, interpretable k clusters",
            "Identifies temporal patterns"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("outputs/algorithm_comparison_main.csv", index=False)
    print("\n" + comparison_df.to_string(index=False))
    print(f"\n✅ Comparison saved to: outputs/algorithm_comparison_main.csv")

    # ---- STEP 8: Summary ----

    print("\n" + "="*80)
    print("[STEP 8/8] Pipeline Complete ✅")
    print("="*80)
    print("\n📊 Generated outputs:")
    print("  Maps:")
    print("   - outputs/map_cleaned_session2.html (cleaned data)")
    print("   - outputs/map_dbscan_main.html (DBSCAN clusters)")
    print("   - outputs/map_kmeans_main.html (K-Means clusters)")
    print("\n  Data:")
    print("   - outputs/clustered_dbscan_main.csv")
    print("   - outputs/clustered_kmeans_main.csv")
    print("   - outputs/clustered_temporal_main.csv")
    print("   - outputs/temporal_clusters_summary.csv")
    print("   - outputs/algorithm_comparison_main.csv")
    print("\n📄 Analysis:")
    print("   - See CLUSTERING_ANALYSIS.md for detailed comparison")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed: {e}")
        traceback.print_exc()
