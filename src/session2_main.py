# src/session2_main.py
# Session 2 — Complete pipeline with algorithm comparison, text mining, and temporal analysis
# Features:
# 1. Optimize and compare 3 clustering algorithms (DBSCAN, K-Means, HDBSCAN)
# 2. Finalize text mining for automatic cluster naming (TF-IDF + descriptive keywords)
# 3. Explore temporal patterns in the Flickr data
# 4. Generate cluster descriptions and enhanced maps with cluster names

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def main() -> None:
    """
    Complete Session 2 pipeline:
    load_data -> clean -> comparison -> text_mining -> temporal_analysis -> outputs
    """
    
    # ---- Paths ----
    csv_path = "../flickr_data2.csv"
    os.makedirs("outputs", exist_ok=True)
    
    print("="*80)
    print("SESSION 2 — COMPLETE CLUSTERING & TEXT MINING PIPELINE")
    print("="*80)
    
    # ---- 1) Load + Clean ----
    print("\n[STEP 1/5] Loading and cleaning data...")
    from load_data import load_data, print_report
    from cleaning import clean_data, print_cleaning_report
    
    df_raw, rep_raw = load_data(csv_path)
    df_clean, rep_clean = clean_data(df_raw)
    print_cleaning_report(rep_clean)
    
    # ---- 2) Algorithm Comparison ----
    print("\n[STEP 2/5] Comparing 3 clustering algorithms...")
    from comparison import (
        compare_algorithms,
        print_comparison_table,
        save_comparison_csv,
        plot_elbow_silhouette,
        generate_recommendation,
    )
    
    comparison_df = compare_algorithms(
        df_clean,
        dbscan_params={"eps_meters": 50.0, "min_samples": 50, "deduplicate_coords": True},
        kmeans_params={"n_clusters": 50},
        hdbscan_params={"min_cluster_size": 50, "min_samples": 50},
    )
    
    print_comparison_table(comparison_df)
    out_comparison_csv = save_comparison_csv(comparison_df)
    print(f"[OK] Comparison metrics saved to: {out_comparison_csv}")
    
    # Plot elbow curve
    try:
        out_elbow = plot_elbow_silhouette(df_clean, k_range=range(20, 81, 10))
        print(f"[OK] Elbow plot saved to: {out_elbow}")
    except Exception as e:
        print(f"[WARN] Could not generate elbow plot: {e}")
    
    # Print recommendation
    recommendation = generate_recommendation(comparison_df)
    
    # ---- 3) Optimal Clustering (DBSCAN recommended) ----
    print("\n[STEP 3/5] Running optimal clustering (DBSCAN)...")
    from clustering import run_dbscan_geo, print_cluster_report, save_clustered_csv, make_cluster_map
    
    df_clustered, rep_cluster = run_dbscan_geo(
        df_clean,
        eps_meters=50.0,
        min_samples=50,
        deduplicate_coords=True,
        coord_precision=4,
    )
    print_cluster_report(rep_cluster)
    
    out_clustered_csv = save_clustered_csv(df_clustered, "outputs/clustered.csv")
    print(f"[OK] Clustered data saved to: {out_clustered_csv}")
    
    # ---- 4) Text Mining for Cluster Names ----
    print("\n[STEP 4/5] Generating cluster descriptions with TF-IDF...")
    from text_mining import (
        preprocess_text,
        extract_cluster_descriptions,
        save_descriptions_csv,
        print_cluster_descriptions,
        create_wordcloud_for_cluster,
    )
    
    df_clustered = preprocess_text(df_clustered, text_col="text")
    descriptions = extract_cluster_descriptions(
        df_clustered,
        cluster_col="cluster",
        text_col="text",
        top_n_keywords=10,
        min_df=2,
        max_df=0.8,
    )
    
    print_cluster_descriptions(descriptions, top_n=15)
    out_descriptions_csv = save_descriptions_csv(descriptions)
    print(f"[OK] Cluster descriptions saved to: {out_descriptions_csv}")
    
    # Create wordclouds for top 5 clusters
    print("\nGenerating word clouds for top 5 clusters...")
    for desc in descriptions[:5]:
        try:
            out_wordcloud = create_wordcloud_for_cluster(
                df_clustered,
                desc.cluster_id,
                output_path=f"outputs/wordcloud_cluster_{desc.cluster_id}.png",
            )
            if out_wordcloud:
                print(f"  [OK] Wordcloud for cluster {desc.cluster_id}: {out_wordcloud}")
        except Exception as e:
            print(f"  [WARN] Could not create wordcloud for cluster {desc.cluster_id}: {e}")
    
    # ---- 5) Temporal Analysis ----
    print("\n[STEP 5/5] Exploring temporal patterns...")
    temporal_results = perform_temporal_analysis(df_clustered)
    
    # ---- 6) Enhanced Cluster Map with Names ----
    print("\n[STEP 6] Creating enhanced cluster map with cluster names...")
    try:
        from visualization import create_cluster_map_with_names
        
        out_cluster_map = create_cluster_map_with_names(
            df_clustered,
            descriptions=descriptions,
            output_html="outputs/map_clusters_named.html",
            sample_n=25000,
        )
        print(f"[OK] Enhanced cluster map with names saved to: {out_cluster_map}")
    except Exception as e:
        print(f"[WARN] Could not create enhanced cluster map: {e}")
    
    # ---- Final Summary ----
    print("\n" + "="*80)
    print("✅ SESSION 2 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {out_comparison_csv} (algorithm comparison metrics)")
    print(f"  - {out_clustered_csv} (clustered data with DBSCAN)")
    print(f"  - {out_descriptions_csv} (TF-IDF cluster descriptions)")
    print(f"  - outputs/map_clusters.html (interactive cluster map)")
    print(f"  - outputs/wordcloud_cluster_*.png (word clouds for top clusters)")
    print(f"\n📊 Key Results:")
    print(f"  - Total clusters: {rep_cluster.n_clusters}")
    print(f"  - Noise points: {rep_cluster.noise_points} ({rep_cluster.noise_ratio*100:.1f}%)")
    print(f"  - Largest cluster: {rep_cluster.cluster_sizes_top10[0][1]:,} photos")
    print(f"\n💡 Recommendation: Use DBSCAN for discovering POIs in urban areas")
    print("="*80 + "\n")


def perform_temporal_analysis(df_clustered: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns in the Flickr data.
    - Photos per month
    - Photos per cluster over time
    - Most active time periods
    """
    print("\nTemporal Analysis:")
    print("-" * 80)
    
    # Ensure taken_dt is datetime
    if "taken_dt" in df_clustered.columns:
        df_clustered["taken_dt"] = pd.to_datetime(df_clustered["taken_dt"], errors="coerce")
        
        # Photos per month
        monthly_counts = df_clustered.set_index("taken_dt").resample("MS").size()
        print(f"\n  Total time span: {df_clustered['taken_dt'].min()} to {df_clustered['taken_dt'].max()}")
        print(f"  Peak month: {monthly_counts.idxmax().strftime('%B %Y')} ({monthly_counts.max()} photos)")
        print(f"  Average per month: {monthly_counts.mean():.0f} photos")
        
        # Cluster activity heatmap
        cluster_temporal = df_clustered.groupby("cluster").agg({
            "taken_dt": ["min", "max", "count"]
        }).round(0)
        
        print(f"\n  Top 5 most photographed clusters (by date range):")
        for cluster_id in df_clustered.groupby("cluster").size().nlargest(5).index:
            cluster_photos = df_clustered[df_clustered["cluster"] == cluster_id]
            min_date = cluster_photos["taken_dt"].min()
            max_date = cluster_photos["taken_dt"].max()
            n_photos = len(cluster_photos)
            print(f"    Cluster {cluster_id}: {n_photos:,} photos ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
        
        # Save temporal analysis
        temporal_csv = "outputs/temporal_analysis.csv"
        temporal_data = []
        for cluster_id in df_clustered["cluster"].unique():
            if cluster_id == -1:
                continue
            cluster_df = df_clustered[df_clustered["cluster"] == cluster_id]
            temporal_data.append({
                "cluster_id": int(cluster_id),
                "n_photos": len(cluster_df),
                "min_date": cluster_df["taken_dt"].min(),
                "max_date": cluster_df["taken_dt"].max(),
                "span_days": (cluster_df["taken_dt"].max() - cluster_df["taken_dt"].min()).days,
            })
        
        temporal_df = pd.DataFrame(temporal_data).sort_values("n_photos", ascending=False)
        temporal_df.to_csv(temporal_csv, index=False)
        print(f"\n  [OK] Temporal analysis saved to: {temporal_csv}")
        
        return {
            "monthly_counts": monthly_counts.to_dict(),
            "temporal_csv": temporal_csv,
        }
    else:
        print("  [WARN] No 'taken_dt' column found. Skipping temporal analysis.")
        return {}


if __name__ == "__main__":
    main()
