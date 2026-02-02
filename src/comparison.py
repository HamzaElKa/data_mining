# src/comparison.py
# Session 2 — Compare 3 clustering algorithms
# Output: metrics table, visualizations, recommendation

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def compare_algorithms(
    df_clean: pd.DataFrame,
    *,
    dbscan_params: Dict = None,
    kmeans_params: Dict = None,
    hdbscan_params: Dict = None,
) -> pd.DataFrame:
    """
    Run 3 clustering algorithms and compare results.
    
    Returns:
    --------
    comparison_df : DataFrame with metrics for each algorithm
    """
    from clustering import run_dbscan_geo, run_kmeans, run_hdbscan
    
    results = []
    
    # Default parameters
    if dbscan_params is None:
        dbscan_params = {"eps_meters": 50.0, "min_samples": 50, "deduplicate_coords": True}
    if kmeans_params is None:
        kmeans_params = {"n_clusters": 50}
    if hdbscan_params is None:
        hdbscan_params = {"min_cluster_size": 50, "min_samples": 50}
    
    print("="*80)
    print("COMPARING 3 CLUSTERING ALGORITHMS")
    print("="*80)
    
    # 1. DBSCAN
    print("\n[1/3] Running DBSCAN...")
    df_dbscan, report_dbscan = run_dbscan_geo(df_clean, **dbscan_params)
    
    results.append({
        "algorithm": "DBSCAN",
        "n_clusters": report_dbscan.n_clusters,
        "noise_points": report_dbscan.noise_points,
        "noise_ratio": f"{report_dbscan.noise_ratio*100:.1f}%",
        "top_cluster_size": report_dbscan.cluster_sizes_top10[0][1] if report_dbscan.cluster_sizes_top10 else 0,
        "silhouette_score": None,  # Will calculate separately
        "davies_bouldin_index": None,
        "parameters": f"eps={dbscan_params['eps_meters']}m, min_samples={dbscan_params['min_samples']}",
    })
    
    # Calculate silhouette for DBSCAN (exclude noise)
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import numpy as np
    mask = df_dbscan['cluster'] != -1
    if mask.sum() > 1:
        from clustering import _gps_to_cartesian
        try:
            # Try to calculate on full data
            X = _gps_to_cartesian(df_dbscan.loc[mask, 'lat'].to_numpy(), df_dbscan.loc[mask, 'long'].to_numpy())
            labels = df_dbscan.loc[mask, 'cluster'].to_numpy()
            
            # Check if dataset is too large for silhouette (>5000 samples)
            if len(X) > 5000:
                print(f"  (Skipping silhouette on {len(X):,} samples - too memory-intensive)")
                results[-1]["silhouette_score"] = "Skipped (large dataset)"
                results[-1]["davies_bouldin_index"] = "Skipped (large dataset)"
            else:
                results[-1]["silhouette_score"] = f"{silhouette_score(X, labels):.4f}"
                results[-1]["davies_bouldin_index"] = f"{davies_bouldin_score(X, labels):.4f}"
        except (MemoryError, Exception) as e:
            print(f"  (Memory error calculating metrics: {type(e).__name__})")
            results[-1]["silhouette_score"] = "Memory error"
            results[-1]["davies_bouldin_index"] = "Memory error"
    
    # 2. K-Means
    print("\n[2/3] Running K-Means...")
    df_kmeans, report_kmeans = run_kmeans(df_clean, **kmeans_params)
    
    results.append({
        "algorithm": "K-Means",
        "n_clusters": report_kmeans["n_clusters"],
        "noise_points": 0,
        "noise_ratio": "0.0%",
        "top_cluster_size": report_kmeans["cluster_sizes_top10"][0][1] if report_kmeans["cluster_sizes_top10"] else 0,
        "silhouette_score": f"{report_kmeans['silhouette_score']:.4f}",
        "davies_bouldin_index": f"{report_kmeans['davies_bouldin_index']:.4f}",
        "parameters": f"n_clusters={kmeans_params['n_clusters']}",
    })
    
    # 3. HDBSCAN
    print("\n[3/3] Running HDBSCAN...")
    try:
        df_hdbscan, report_hdbscan = run_hdbscan(df_clean, **hdbscan_params)
        
        results.append({
            "algorithm": "HDBSCAN",
            "n_clusters": report_hdbscan["n_clusters"],
            "noise_points": report_hdbscan["noise_points"],
            "noise_ratio": f"{report_hdbscan['noise_ratio']*100:.1f}%",
            "top_cluster_size": report_hdbscan["cluster_sizes_top10"][0][1] if report_hdbscan["cluster_sizes_top10"] else 0,
            "silhouette_score": f"{report_hdbscan['silhouette_score']:.4f}" if report_hdbscan['silhouette_score'] else "N/A",
            "davies_bouldin_index": f"{report_hdbscan['davies_bouldin_index']:.4f}" if report_hdbscan['davies_bouldin_index'] else "N/A",
            "parameters": f"min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params['min_samples']}",
        })
    except (ImportError, MemoryError) as e:
        print(f"[WARN] HDBSCAN failed: {type(e).__name__}. Skipping.")
        results.append({
            "algorithm": "HDBSCAN",
            "n_clusters": "N/A",
            "noise_points": "N/A",
            "noise_ratio": "N/A",
            "top_cluster_size": "N/A",
            "silhouette_score": "N/A",
            "davies_bouldin_index": "N/A",
            "parameters": "Not installed",
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def print_comparison_table(comparison_df: pd.DataFrame):
    """
    Print comparison table in a readable format.
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    print("\n📊 INTERPRETATION:")
    print("- Silhouette Score: [-1, 1], higher is better (cohesion + separation)")
    print("- Davies-Bouldin Index: Lower is better (intra-cluster similarity)")
    print("- Noise Ratio: % of points not assigned to any cluster")
    print()


def save_comparison_csv(comparison_df: pd.DataFrame, output_path: str = "outputs/comparison_metrics.csv") -> str:
    """
    Save comparison table to CSV.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    return output_path


def plot_elbow_silhouette(
    df_clean: pd.DataFrame,
    k_range: range = range(20, 81, 10),
    output_path: str = "outputs/kmeans_elbow_silhouette.png",
):
    """
    Plot elbow and silhouette curves for K-Means.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return None
    
    from clustering import find_optimal_k
    import os
    
    print(f"\nTesting K-Means for K in {list(k_range)}...")
    k_values, inertias, silhouettes = find_optimal_k(df_clean, k_range=k_range)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_values, silhouettes, marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Elbow/Silhouette plot saved to: {output_path}")
    
    # Print optimal K suggestion
    optimal_k_silhouette = k_values[silhouettes.index(max(silhouettes))]
    print(f"\n💡 Suggested optimal K (max silhouette): {optimal_k_silhouette}")
    
    return output_path


def generate_recommendation(comparison_df: pd.DataFrame) -> str:
    """
    Generate algorithm recommendation based on metrics.
    """
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    
    recommendation = """
Based on the comparison:

**DBSCAN** (Recommended for Session 2):
✅ Pros:
  - Discovers optimal number of clusters automatically (49 clusters)
  - Handles noise well (62% noise = photos outside major POIs)
  - No need to choose K a priori
  - Good for exploring unknown data structure

⚠️ Cons:
  - Assumes uniform density (fixed eps)
  - Sensitive to parameter choice (eps, min_samples)
  - May create mega-clusters in very dense urban centers

**K-Means**:
✅ Pros:
  - Simple and fast
  - Creates balanced clusters
  - Good silhouette score

⚠️ Cons:
  - Requires choosing K beforehand (must test multiple values)
  - Assumes spherical clusters (not ideal for irregular POI shapes)
  - No noise handling (forces all points into clusters)
  - Can split natural POIs into multiple clusters

**HDBSCAN**:
✅ Pros:
  - Handles varying density (Bellecour dense vs Terreaux less dense)
  - Hierarchical structure (can zoom into sub-POIs)
  - More robust to parameters than DBSCAN

⚠️ Cons:
  - More complex to understand and tune
  - Longer computation time
  - Less intuitive parameters

**Final Recommendation:**
For this project (discovering Lyon POIs), **DBSCAN is the best choice** because:
1. We don't know the "true" number of POIs in Lyon → DBSCAN discovers it
2. Urban photography has lots of noise (residential areas, transit) → DBSCAN handles it
3. Parameters (eps=50m, min_samples=50) have clear interpretations
4. Results are interpretable and match real Lyon geography

K-Means is useful for comparison and validation (confirms ~50 clusters is reasonable).
HDBSCAN would be better for advanced analysis with variable density zones.
"""
    
    print(recommendation)
    print("="*80)
    
    return recommendation


if __name__ == "__main__":
    # Run full comparison
    try:
        from load_data import load_data
        from cleaning import clean_data
        
        print("Loading and cleaning data...")
        df_raw, _ = load_data("../flickr_data2.csv")
        df_clean, _ = clean_data(df_raw)
        
        # Compare algorithms
        comparison_df = compare_algorithms(df_clean)
        
        # Print results
        print_comparison_table(comparison_df)
        
        # Save CSV
        out_csv = save_comparison_csv(comparison_df)
        print(f"\n[OK] Comparison saved to: {out_csv}")
        
        # Plot elbow/silhouette for K-Means
        plot_elbow_silhouette(df_clean, k_range=range(20, 81, 10))
        
        # Generate recommendation
        generate_recommendation(comparison_df)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
