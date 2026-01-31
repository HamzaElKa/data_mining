"""
Quick comparison of 3 algorithms (without K grid search)
Session 2 - Data Mining
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_data
from cleaning import clean_data
from clustering import run_dbscan_geo, run_kmeans, run_hdbscan, _gps_to_cartesian
from visualization import create_map
from sklearn.metrics import silhouette_score, davies_bouldin_score

def main():
    print("="*80)
    print("COMPARAISON RAPIDE DES 3 ALGORITHMES")
    print("="*80)
    
    # Load data
    print("\n[0/3] Loading and cleaning data...")
    csv_path = Path(__file__).parent.parent / "flickr_data2.csv"
    df_raw, _ = load_data(str(csv_path))
    df_clean, _ = clean_data(df_raw)
    print(f"✓ Loaded {len(df_clean)} cleaned photos")
    
    results = []
    
    # 1. DBSCAN
    print("\n[1/3] Running DBSCAN (eps=50m, min_samples=50)...")
    df_dbscan, report_dbscan = run_dbscan_geo(
        df_clean,
        eps_meters=50.0,
        min_samples=50,
        deduplicate_coords=True
    )
    
    # Calculate silhouette (exclude noise)
    mask = df_dbscan['cluster'] != -1
    if mask.sum() > 1:
        X = _gps_to_cartesian(
            df_dbscan.loc[mask, 'lat'].to_numpy(),
            df_dbscan.loc[mask, 'long'].to_numpy()
        )
        labels = df_dbscan.loc[mask, 'cluster'].to_numpy()
        sil_dbscan = silhouette_score(X, labels)
        db_dbscan = davies_bouldin_score(X, labels)
    else:
        sil_dbscan = None
        db_dbscan = None
    
    results.append({
        "Algorithme": "DBSCAN",
        "Clusters": report_dbscan.n_clusters,
        "Bruit (%)": f"{report_dbscan.noise_ratio*100:.1f}",
        "Top Cluster": report_dbscan.cluster_sizes_top10[0][1] if report_dbscan.cluster_sizes_top10 else 0,
        "Silhouette": f"{sil_dbscan:.4f}" if sil_dbscan else "N/A",
        "Davies-Bouldin": f"{db_dbscan:.4f}" if db_dbscan else "N/A",
        "Paramètres": "eps=50m, min_samples=50",
    })
    
    print(f"✓ DBSCAN: {report_dbscan.n_clusters} clusters, {report_dbscan.noise_ratio*100:.1f}% bruit")
    if sil_dbscan:
        print(f"  Silhouette: {sil_dbscan:.4f}, Davies-Bouldin: {db_dbscan:.4f}")
    
    # 2. K-Means
    print("\n[2/3] Running K-Means (K=50)...")
    df_kmeans, report_kmeans = run_kmeans(df_clean, n_clusters=50)
    
    results.append({
        "Algorithme": "K-Means",
        "Clusters": report_kmeans["n_clusters"],
        "Bruit (%)": "0.0",
        "Top Cluster": report_kmeans["cluster_sizes_top10"][0][1] if report_kmeans["cluster_sizes_top10"] else 0,
        "Silhouette": f"{report_kmeans['silhouette_score']:.4f}",
        "Davies-Bouldin": f"{report_kmeans['davies_bouldin_index']:.4f}",
        "Paramètres": "K=50",
    })
    
    print(f"✓ K-Means: {report_kmeans['n_clusters']} clusters, 0.0% bruit")
    print(f"  Silhouette: {report_kmeans['silhouette_score']:.4f}, Davies-Bouldin: {report_kmeans['davies_bouldin_index']:.4f}")
    
    # 3. HDBSCAN
    print("\n[3/3] Running HDBSCAN (min_cluster_size=50)...")
    try:
        df_hdbscan, report_hdbscan = run_hdbscan(
            df_clean,
            min_cluster_size=50,
            min_samples=50
        )
        
        results.append({
            "Algorithme": "HDBSCAN",
            "Clusters": report_hdbscan["n_clusters"],
            "Bruit (%)": f"{report_hdbscan['noise_ratio']*100:.1f}",
            "Top Cluster": report_hdbscan["cluster_sizes_top10"][0][1] if report_hdbscan["cluster_sizes_top10"] else 0,
            "Silhouette": f"{report_hdbscan['silhouette_score']:.4f}" if report_hdbscan['silhouette_score'] else "N/A",
            "Davies-Bouldin": f"{report_hdbscan['davies_bouldin_index']:.4f}" if report_hdbscan['davies_bouldin_index'] else "N/A",
            "Paramètres": "min_cluster_size=50",
        })
        
        print(f"✓ HDBSCAN: {report_hdbscan['n_clusters']} clusters, {report_hdbscan['noise_ratio']*100:.1f}% bruit")
        if report_hdbscan['silhouette_score']:
            print(f"  Silhouette: {report_hdbscan['silhouette_score']:.4f}, Davies-Bouldin: {report_hdbscan['davies_bouldin_index']:.4f}")
    
    except ImportError:
        print("[WARN] HDBSCAN not installed. Skipping.")
        results.append({
            "Algorithme": "HDBSCAN",
            "Clusters": "N/A",
            "Bruit (%)": "N/A",
            "Top Cluster": "N/A",
            "Silhouette": "N/A",
            "Davies-Bouldin": "N/A",
            "Paramètres": "Not installed",
        })
    
    # Create comparison table
    print("\n" + "="*80)
    print("RÉSULTATS COMPARAISON")
    print("="*80 + "\n")
    
    df_comparison = pd.DataFrame(results)
    print(df_comparison.to_string(index=False))
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "outputs" / "comparison_quick.csv"
    df_comparison.to_csv(output_path, index=False)
    print(f"\n✓ Saved comparison to: {output_path}")
    
    # Generate maps for each algorithm
    print("\n" + "="*80)
    print("GÉNÉRATION CARTES HTML")
    print("="*80)
    
    from visualization import MapConfig
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Map 1: DBSCAN
    print("\n[1/3] Generating DBSCAN map...")
    cfg_dbscan = MapConfig()
    # Temporary save in outputs folder
    import folium
    m_dbscan = folium.Map(location=[45.764, 4.836], zoom_start=13, control_scale=True)
    
    # Add colored markers based on cluster
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    # Sample for performance
    df_sample = df_dbscan.sample(min(5000, len(df_dbscan)), random_state=42)
    
    # Get unique clusters (excluding noise)
    clusters = sorted([c for c in df_sample['cluster'].unique() if c != -1])
    n_clusters = len(clusters)
    
    # Color map
    if n_clusters > 0:
        cmap = cm.get_cmap('tab20', n_clusters)
        
        for idx, row in df_sample.iterrows():
            cluster_id = row['cluster']
            
            if cluster_id == -1:
                # Noise in gray
                color = 'gray'
                popup_text = f"Cluster: Noise<br>Lat: {row['lat']:.5f}<br>Lon: {row['long']:.5f}"
            else:
                # Cluster color
                cluster_idx = clusters.index(cluster_id)
                rgba = cmap(cluster_idx)
                color = colors.rgb2hex(rgba[:3])
                popup_text = f"Cluster: {cluster_id}<br>Lat: {row['lat']:.5f}<br>Lon: {row['long']:.5f}"
            
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=200),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m_dbscan)
    
    map_path_dbscan = output_dir / "map_dbscan.html"
    m_dbscan.save(str(map_path_dbscan))
    print(f"✓ DBSCAN map saved: {map_path_dbscan}")
    
    # Map 2: K-Means
    print("\n[2/3] Generating K-Means map...")
    m_kmeans = folium.Map(location=[45.764, 4.836], zoom_start=13, control_scale=True)
    
    df_sample_km = df_kmeans.sample(min(5000, len(df_kmeans)), random_state=42)
    clusters_km = sorted(df_sample_km['cluster'].unique())
    n_clusters_km = len(clusters_km)
    
    if n_clusters_km > 0:
        cmap_km = cm.get_cmap('tab20', min(n_clusters_km, 20))
        
        for idx, row in df_sample_km.iterrows():
            cluster_id = row['cluster']
            cluster_idx = clusters_km.index(cluster_id) % 20
            rgba = cmap_km(cluster_idx)
            color = colors.rgb2hex(rgba[:3])
            popup_text = f"Cluster: {cluster_id}<br>Lat: {row['lat']:.5f}<br>Lon: {row['long']:.5f}"
            
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=3,
                popup=folium.Popup(popup_text, max_width=200),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m_kmeans)
    
    map_path_kmeans = output_dir / "map_kmeans.html"
    m_kmeans.save(str(map_path_kmeans))
    print(f"✓ K-Means map saved: {map_path_kmeans}")
    
    # Map 3: HDBSCAN
    if 'df_hdbscan' in locals():
        print("\n[3/3] Generating HDBSCAN map...")
        m_hdbscan = folium.Map(location=[45.764, 4.836], zoom_start=13, control_scale=True)
        
        df_sample_hdb = df_hdbscan.sample(min(5000, len(df_hdbscan)), random_state=42)
        clusters_hdb = sorted([c for c in df_sample_hdb['cluster'].unique() if c != -1])
        n_clusters_hdb = len(clusters_hdb)
        
        if n_clusters_hdb > 0:
            # Use viridis for many clusters
            cmap_hdb = cm.get_cmap('viridis', min(n_clusters_hdb, 256))
            
            for idx, row in df_sample_hdb.iterrows():
                cluster_id = row['cluster']
                
                if cluster_id == -1:
                    color = 'gray'
                    popup_text = f"Cluster: Noise<br>Lat: {row['lat']:.5f}<br>Lon: {row['long']:.5f}"
                else:
                    cluster_idx = clusters_hdb.index(cluster_id)
                    rgba = cmap_hdb(cluster_idx % 256)
                    color = colors.rgb2hex(rgba[:3])
                    popup_text = f"Cluster: {cluster_id}<br>Lat: {row['lat']:.5f}<br>Lon: {row['long']:.5f}"
                
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=3,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m_hdbscan)
        
        map_path_hdbscan = output_dir / "map_hdbscan.html"
        m_hdbscan.save(str(map_path_hdbscan))
        print(f"✓ HDBSCAN map saved: {map_path_hdbscan}")
    else:
        print("\n[3/3] HDBSCAN map skipped (not installed)")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMANDATION")
    print("="*80)
    print("""
Pour ce projet Grand Lyon :

🥇 RECOMMANDÉ : DBSCAN
   - Découvre automatiquement K (~49 clusters)
   - Gère le bruit (62% photos hors POI majeurs)
   - Paramètres interprétables (eps=50m = 1 pâté de maisons)
   - Formes arbitraires (quais Rhône, Parc Tête d'Or)

🥈 VALIDATION : K-Means
   - Meilleur silhouette (0.48 vs 0.45)
   - Clusters équilibrés
   - Valide que ~50 clusters est cohérent
   - Mais force assignation (pas de gestion bruit)

🥉 EXPLORATOIRE : HDBSCAN
   - Densité variable (Bellecour dense vs Terreaux moins)
   - Résultats similaires à DBSCAN pour Lyon (densité uniforme)
   - Plus complexe à expliquer
    """)

if __name__ == "__main__":
    main()
