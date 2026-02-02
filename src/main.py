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
    make_cluster_map,
    save_clustered_csv,
    make_cluster_map,
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

    # ---- 4) Clustering ----
    # Tune these two hyperparams during Session 1:
    # IMPORTANT: Avec déduplication GPS activée par défaut pour éviter sur-représentation
    # eps_meters: rayon de voisinage (50-200m typique pour POI urbains)
    # min_samples: photos minimum pour former cluster (plus élevé = clusters plus denses)
    
    eps_meters = 50.0  # 50m = bon compromis pour POI distincts
    min_samples = 50   # 50 photos minimum = POI significatifs

    df_clustered, rep_cluster = run_dbscan_geo(
        df_clean,
        eps_meters=eps_meters,
        min_samples=min_samples,
        cluster_col="cluster",
        deduplicate_coords=True,  # CRUCIAL: évite méga-clusters
        coord_precision=4,  # 4 décimales = ~11m de précision
    )
    print_cluster_report(rep_cluster)

    # ---- 5) Save clustered data ----
    out_csv = save_clustered_csv(df_clustered, "outputs/clustered.csv")
    print(f"[OK] Clustered CSV saved to: {out_csv}")

    # ---- 6) Optional: clusters map (requires folium) ----
    try:
        out_cluster_map = make_cluster_map(
            df_clustered,
            output_html="outputs/map_clusters.html",
            sample_n=25000,
            random_state=42,
            center=(45.7640, 4.8357),
            zoom_start=12,
        )
        print(f"[OK] Cluster map saved to: {out_cluster_map}")
    except Exception as e:
        print(f"[WARN] Cluster map not generated (folium missing or error): {e}")

    print("\n✅ Pipeline finished.")
    print("Open these files:")
    print("- outputs/map_session1.html")
    print("- outputs/map_clusters.html (if generated)")
    print("- outputs/clustered.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()