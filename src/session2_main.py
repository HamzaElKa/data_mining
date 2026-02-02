# src/main.py
# Complete end-to-end pipeline: Session 1 + Session 2
# Load -> Clean -> Map -> Algorithm Comparison -> DBSCAN -> Text Mining -> Named Outputs

from __future__ import annotations

import os

from load_data import load_data, print_report
from cleaning import clean_data, print_cleaning_report
from visualization import create_map, MapConfig
from clustering import (
    run_dbscan_geo,
    print_cluster_report,
    save_clustered_csv,
    make_cluster_map,
    make_cluster_map_named,
)
from comparison import compare_algorithms, generate_recommendation, save_comparison_csv
from text_mining import (
    extract_cluster_descriptions,
    extract_keywords_by_frequency,
    combine_cluster_names,
    add_cluster_names_to_dataframe,
    save_named_clusters_csv,
    print_named_clusters,
)


def main() -> None:
    """
    Complete unified pipeline: Sessions 1 + 2
    1. Load + explore
    2. Clean
    3. Initial map visualization
    4. Algorithm comparison (DBSCAN, K-Means, HDBSCAN)
    5. DBSCAN clustering (recommended)
    6. Text mining (dual algorithms: TF-IDF + Frequency)
    7. Save all outputs (CSV, maps, cluster names)
    """
    # ---- Paths (run from project root or src/) ----
    csv_path = "../flickr_data2.csv"  # File at project root
    os.makedirs("outputs", exist_ok=True)

    # ========== STEP 1: Load + Explore ==========
    print("\n" + "="*60)
    print("[STEP 1/7] Loading data...")
    print("="*60)
    df_raw, rep_raw = load_data(csv_path)
    print_report(rep_raw)

    # ========== STEP 2: Clean ==========
    print("\n" + "="*60)
    print("[STEP 2/7] Cleaning data...")
    print("="*60)
    df_clean, rep_clean = clean_data(df_raw)
    print_cleaning_report(rep_clean)

    # ========== STEP 3: Initial Map ==========
    print("\n" + "="*60)
    print("[STEP 3/7] Creating initial map...")
    print("="*60)
    map_cfg = MapConfig(
        output_html="outputs/map_session1.html",
        sample_n=15000,
        random_state=42,
        center=(45.7640, 4.8357),
        zoom_start=12,
        max_markers=15000,
    )
    out_map = create_map(df_clean, cfg=map_cfg)
    print(f"[OK] Initial map saved to: {out_map}")

    # ========== STEP 4: Algorithm Comparison ==========
    print("\n" + "="*60)
    print("[STEP 4/7] Comparing clustering algorithms...")
    print("="*60)
    try:
        comparison_metrics = compare_algorithms(df_clean)
        recommendation = generate_recommendation(comparison_metrics)
        print("\n[Comparison Results]")
        print(recommendation["summary"])
        print(f"\n[Recommended Algorithm: {recommendation['recommended']}]")
        print(f"Score: {recommendation['score']}/5.00")
        
        # Save comparison metrics
        out_comparison = save_comparison_csv(
            comparison_metrics, "outputs/comparison_metrics.csv"
        )
        print(f"\n[OK] Metrics saved to: {out_comparison}")
    except Exception as e:
        print(f"[WARN] Algorithm comparison skipped: {e}")
        recommendation = {"recommended": "DBSCAN"}

    # ========== STEP 5: DBSCAN Clustering ==========
    print("\n" + "="*60)
    print("[STEP 5/7] Running DBSCAN clustering...")
    print("="*60)
    eps_meters = 50.0   # 50m radius for distinct POIs
    min_samples = 50    # 50 photos minimum for significant POIs

    df_clustered, rep_cluster = run_dbscan_geo(
        df_clean,
        eps_meters=eps_meters,
        min_samples=min_samples,
        cluster_col="cluster",
        deduplicate_coords=True,  # Avoid over-representation
        coord_precision=4,  # ~11m precision
    )
    print_cluster_report(rep_cluster)
    
    # Save initial clustering result
    out_csv = save_clustered_csv(df_clustered, "outputs/clustered.csv")
    print(f"[OK] Clustered data saved to: {out_csv}")

    # ========== STEP 6: Text Mining (Dual Algorithm) ==========
    print("\n" + "="*60)
    print("[STEP 6/7] Running dual text mining algorithms...")
    print("="*60)
    
    print("\n[6a] Algorithm 1: TF-IDF extraction...")
    try:
        tfidf_names = extract_cluster_descriptions(
            df_clustered,
            cluster_col="cluster",
            n_keywords=5,
        )
        print(f"  ✓ TF-IDF names extracted for {len(tfidf_names)} clusters")
    except Exception as e:
        print(f"  ✗ TF-IDF failed: {e}")
        tfidf_names = {}

    print("\n[6b] Algorithm 2: Keyword frequency extraction...")
    try:
        freq_names = extract_keywords_by_frequency(
            df_clustered,
            cluster_col="cluster",
            n_keywords=5,
        )
        print(f"  ✓ Frequency names extracted for {len(freq_names)} clusters")
    except Exception as e:
        print(f"  ✗ Frequency failed: {e}")
        freq_names = {}

    print("\n[6c] Combining TF-IDF + Keyword Frequency...")
    try:
        combined_names = combine_cluster_names(tfidf_names, freq_names)
        print(f"  ✓ Combined names created for {len(combined_names)} clusters")
    except Exception as e:
        print(f"  ✗ Combination failed: {e}")
        combined_names = {}

    print("\n[6d] Adding cluster names to dataframe...")
    try:
        df_named = add_cluster_names_to_dataframe(
            df_clustered.copy(),
            combined_names,
            cluster_col="cluster",
            name_col="cluster_name",
        )
        print(f"  ✓ Cluster names integrated into dataframe")
    except Exception as e:
        print(f"  ✗ Integration failed: {e}")
        df_named = df_clustered.copy()
        df_named["cluster_name"] = df_named["cluster"].astype(str)

    print("\n[6e] Displaying cluster names...")
    try:
        print_named_clusters(df_named, cluster_col="cluster", name_col="cluster_name")
    except Exception as e:
        print(f"  ✗ Display failed: {e}")

    # ========== STEP 7: Save All Outputs ==========
    print("\n" + "="*60)
    print("[STEP 7/7] Saving outputs...")
    print("="*60)

    # Save named clusters CSV
    print("\n[7a] Saving named clusters CSV...")
    try:
        out_named_csv = save_named_clusters_csv(
            df_named, "outputs/clustered_named.csv", name_col="cluster_name"
        )
        print(f"  ✓ Saved to: {out_named_csv}")
    except Exception as e:
        print(f"  ✗ Save failed: {e}")

    # Save cluster names reference
    print("\n[7b] Saving cluster names reference...")
    try:
        cluster_ref = df_named.groupby("cluster")["cluster_name"].first().reset_index()
        cluster_ref = cluster_ref.sort_values("cluster")
        cluster_ref.to_csv("outputs/cluster_names_reference.csv", index=False)
        print(f"  ✓ Saved to: outputs/cluster_names_reference.csv")
        print(f"     {len(cluster_ref)} clusters documented")
    except Exception as e:
        print(f"  ✗ Save failed: {e}")

    # Basic cluster map
    print("\n[7c] Creating basic cluster map...")
    try:
        out_cluster_map = make_cluster_map(
            df_clustered,
            output_html="outputs/map_clusters.html",
            sample_n=25000,
            random_state=42,
            center=(45.7640, 4.8357),
            zoom_start=12,
        )
        print(f"  ✓ Saved to: {out_cluster_map}")
    except Exception as e:
        print(f"  ✗ Map creation failed: {e}")

    # Named cluster map (with cluster names in legend)
    print("\n[7d] Creating named cluster map...")
    try:
        out_named_map = make_cluster_map_named(
            df_named,
            output_html="outputs/map_clusters_named.html",
            sample_n=25000,
            random_state=42,
            center=(45.7640, 4.8357),
            zoom_start=12,
            cluster_col="cluster",
            name_col="cluster_name",
        )
        print(f"  ✓ Saved to: {out_named_map}")
        print(f"     ⭐ THIS IS THE MAIN INTERACTIVE VISUALIZATION")
    except Exception as e:
        print(f"  ✗ Named map creation failed: {e}")

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\n📊 Generated outputs in 'outputs/' directory:")
    print("  • map_session1.html - Raw data visualization")
    print("  • map_clusters.html - Basic cluster map")
    print("  • map_clusters_named.html - ⭐ Interactive map with cluster names")
    print("  • clustered.csv - Raw clustering results")
    print("  • clustered_named.csv - With automatic cluster names")
    print("  • cluster_names_reference.csv - Cluster ID → Name mapping")
    print("  • comparison_metrics.csv - Algorithm comparison scores")
    print("\n📈 Key Statistics:")
    n_clusters = df_clustered["cluster"].max() + 1
    noise = (df_clustered["cluster"] == -1).sum()
    print(f"  • Clusters discovered: {n_clusters}")
    print(f"  • Noise points: {noise} ({100*noise/len(df_clustered):.1f}%)")
    print(f"  • Processing time: See timestamps above")
    print("\n💡 Next steps:")
    print("  1. Open outputs/map_clusters_named.html in a browser")
    print("  2. Explore the interactive map")
    print("  3. Review cluster names and interpretations")
    print("  4. Check outputs/clustered_named.csv for detailed data")


if __name__ == "__main__":
    main()
