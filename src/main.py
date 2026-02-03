# src/main.py
# End-to-end Session 1 pipeline:
# load_data -> cleaning -> visualization (map) -> clustering (DBSCAN) -> cluster map + CSV

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
    analyze_temporal_clusters,
    make_cluster_map,
    save_clustered_csv,
    print_cluster_report,
    ClusterReport,
)

from text_mining import (
    preprocess_text,
    extract_cluster_descriptions,
    extract_keywords_by_frequency,
    combine_cluster_names,
    save_descriptions_csv,
    print_cluster_descriptions,
)


def main() -> None:
    # ---- Paths (run from project root or src/) ----
    csv_path = "../flickr_data2.csv"  # Fichier à la racine du projet

    # ========== STEP 1: Load + Explore ==========
    print("\n" + "="*60)
    print("[STEP 1/7] Loading data...")
    print("="*60)
    df_raw, rep_raw = load_data(csv_path)
    print_report(rep_raw)

    # ---- 2) Clean ----
    df_clean, rep_clean = clean_data(df_raw)
    print_cleaning_report(rep_clean)

    # ---- 3) Map (sampled) ----
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
    print(f"[OK] Session 1 map saved to: {out_map}")

    # ---- 4) Clustering - Algorithm 1: DBSCAN ----
    print("\n" + "="*60)
    print("[STEP 4/6] DBSCAN Clustering...")
    print("="*60)
    
    df_dbscan, rep_dbscan = run_dbscan_geo(
        df_clean,
        eps_meters=120.0,
        min_samples=50,
        cluster_col="cluster",
        deduplicate_coords=True,
        coord_precision=4,
    )
    print_cluster_report(rep_dbscan)
    
    try:
        out_map_dbscan = make_cluster_map(
            df_dbscan,
            output_html="outputs/map_clusters_dbscan.html",
            sample_n=25000,
            random_state=42,
        )
        print(f"[OK] DBSCAN map saved to: {out_map_dbscan}")
    except Exception as e:
        print(f"[WARN] DBSCAN map failed: {e}")

    # ---- 5) Clustering - Algorithm 2: K-Means ----
    print("\n" + "="*60)
    print("[STEP 5/6] K-Means Clustering...")
    print("="*60)
    
    df_kmeans, rep_kmeans = run_kmeans(
        df_clean,
        n_clusters=45,
        cluster_col="cluster",
        random_state=42,
    )
    
    print("\n" + "=" * 80)
    print("K-MEANS REPORT")
    print("=" * 80)
    print(f"Clusters: {rep_kmeans['n_clusters']}")
    print(f"Silhouette: {rep_kmeans['silhouette_score']:.4f}")
    print(f"Davies-Bouldin: {rep_kmeans['davies_bouldin_index']:.4f}")
    print("=" * 80 + "\n")
    
    try:
        out_map_kmeans = make_cluster_map(
            df_kmeans,
            output_html="outputs/map_clusters_kmeans.html",
            sample_n=25000,
            random_state=42,
        )
        print(f"[OK] K-Means map saved to: {out_map_kmeans}")
    except Exception as e:
        print(f"[WARN] K-Means map failed: {e}")

    # ---- 6) Clustering - Algorithm 3: Temporal K-Means ----
    print("\n" + "="*60)
    print("[STEP 6/6] Temporal Clustering...")
    print("="*60)
    
    df_temporal, rep_temporal = run_temporal_kmeans(
        df_clean,
        datetime_col="datetime_taken",
        n_clusters=6,
        cluster_col="temporal_cluster",
        features=['month', 'day_of_week', 'season', 'hour_bucket'],
        random_state=42,
    )
    
    # Analyze temporal clusters
    temporal_summary = analyze_temporal_clusters(
        df_temporal,
        datetime_col="datetime_taken",
        cluster_col="temporal_cluster",
    )
    
    print("\nTemporal Cluster Descriptions:")
    for _, row in temporal_summary.iterrows():
        print(f"  {row['description']}")
    
    # Save temporal results
    save_clustered_csv(df_temporal, "outputs/clustered_temporal_main.csv")
    temporal_summary.to_csv("outputs/temporal_clusters_summary.csv", index=False)
    
    # Generate map for temporal clusters
    print("\n" + "="*60)
    print("Generating Temporal Clustering Map...")
    print("="*60)
    
    try:
        out_map_temporal = make_cluster_map(
            df_temporal,
            lat_col="lat",
            lon_col="long",
            cluster_col="temporal_cluster",
            output_html="outputs/map_clusters_temporal.html",
            sample_n=25000,
            random_state=42,
        )
        print(f"[OK] Temporal clustering map saved to: {out_map_temporal}")
    except Exception as e:
        print(f"[WARN] Temporal map generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nPipeline finished.")
    print("\nGenerated maps:")
    print("  - outputs/map_cleaned_session2.html (cleaned data)")
    print("  - outputs/map_clusters_dbscan.html (DBSCAN spatial)")
    print("  - outputs/map_clusters_kmeans.html (K-Means spatial)")
    print("  - outputs/map_clusters_temporal.html (Temporal patterns)")
    print("\nTemporal clusters saved to outputs/clustered_temporal_main.csv")

    # ---- STEP 7: TEXT MINING - Algorithm 1: TF-IDF ----
    print("\n" + "="*80)
    print("[STEP 7/8] TEXT MINING - Algorithm 1: TF-IDF Keywords")
    print("="*80)
    
    # Preprocess text
    df_kmeans_text = preprocess_text(df_kmeans)
    
    # Extract TF-IDF descriptions
    tfidf_descriptions = extract_cluster_descriptions(
        df_kmeans_text,
        cluster_col="cluster",
        text_col="text",
        top_n_keywords=10,
        min_df=2,
        max_df=0.8,
    )
    
    print_cluster_descriptions(tfidf_descriptions, top_n=5)
    
    # Save TF-IDF results
    out_tfidf = save_descriptions_csv(
        tfidf_descriptions,
        "outputs/cluster_descriptions_tfidf.csv"
    )
    print(f"[OK] TF-IDF descriptions saved to: {out_tfidf}")

    # ---- STEP 8: TEXT MINING - Algorithm 2: BM25 ----
    print("\n" + "="*80)
    print("[STEP 8/8] TEXT MINING - Algorithm 2: BM25 Keywords (Stable)")
    print("="*80)
    
    # Extract BM25-based keywords
    bm25_keywords = extract_keywords_by_frequency(
        df_kmeans_text,
        cluster_col="cluster",
        text_col="text",
        top_n=10,
    )
    
    print("\n" + "="*80)
    print("BM25-BASED KEYWORDS (Top 5 clusters)")
    print("="*80)
    
    for cluster_id in sorted(bm25_keywords.keys())[:5]:
        keywords = bm25_keywords[cluster_id]
        n_photos = int((df_kmeans_text["cluster"] == cluster_id).sum())
        print(f"\nCluster {cluster_id} ({n_photos} photos):")
        for keyword, score in keywords[:5]:
            print(f"  - {keyword:20s} (BM25: {score:.4f})")
    print("=" * 80 + "\n")
    
    # Combine both approaches for final names
    cluster_names = combine_cluster_names(tfidf_descriptions, bm25_keywords)
    
    print("COMBINED CLUSTER NAMES (TF-IDF + BM25):")
    for cid, name in sorted(cluster_names.items())[:10]:
        print(f"  Cluster {cid:2d}: {name}")
    print()

    # Save combined names
    names_df = pd.DataFrame([
        {"cluster_id": cid, "combined_name": name}
        for cid, name in cluster_names.items()
    ])
    names_df.to_csv("outputs/cluster_names_combined.csv", index=False)
    print(f"[OK] Combined cluster names saved to: outputs/cluster_names_combined.csv")

    # ---- STEP 7: TEXT MINING - Algorithm 1: TF-IDF ----
    print("\n" + "="*80)
    print("[STEP 7/8] TEXT MINING - Algorithm 1: TF-IDF Keywords")
    print("="*80)
    
    # Preprocess text
    df_kmeans_text = preprocess_text(df_kmeans)
    
    # Extract TF-IDF descriptions
    tfidf_descriptions = extract_cluster_descriptions(
        df_kmeans_text,
        cluster_col="cluster",
        text_col="text",
        top_n_keywords=10,
        min_df=2,
        max_df=0.8,
    )
    
    print_cluster_descriptions(tfidf_descriptions, top_n=5)
    
    # Save TF-IDF results
    out_tfidf = save_descriptions_csv(
        tfidf_descriptions,
        "outputs/cluster_descriptions_tfidf.csv"
    )
    print(f"[OK] TF-IDF descriptions saved to: {out_tfidf}")

    # ---- STEP 8: TEXT MINING - Algorithm 2: Frequency ----
    print("\n" + "="*80)
    print("[STEP 8/8] TEXT MINING - Algorithm 2: Frequency Keywords")
    print("="*80)
    
    # Extract frequency-based keywords
    frequency_keywords = extract_keywords_by_frequency(
        df_kmeans_text,
        cluster_col="cluster",
        text_col="text",
        top_n=10,
    )
    
    print("\n" + "="*80)
    print("FREQUENCY-BASED KEYWORDS (Top 5 clusters)")
    print("="*80)
    
    for cluster_id in sorted(frequency_keywords.keys())[:5]:
        keywords = frequency_keywords[cluster_id]
        n_photos = int((df_kmeans_text["cluster"] == cluster_id).sum())
        print(f"\nCluster {cluster_id} ({n_photos} photos):")
        for keyword, freq in keywords[:5]:
            print(f"  - {keyword:20s} (freq: {freq:,})")
    print("=" * 80 + "\n")
    
    # Combine both approaches for final names
    cluster_names = combine_cluster_names(tfidf_descriptions, frequency_keywords)
    
    print("📌 COMBINED CLUSTER NAMES (TF-IDF + Frequency):")
    for cid, name in sorted(cluster_names.items())[:10]:
        print(f"  Cluster {cid:2d}: {name}")
    print()

    # Save combined names
    names_df = pd.DataFrame([
        {"cluster_id": cid, "combined_name": name}
        for cid, name in cluster_names.items()
    ])
    names_df.to_csv("outputs/cluster_names_combined.csv", index=False)
    print(f"[OK] Combined cluster names saved to: outputs/cluster_names_combined.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()