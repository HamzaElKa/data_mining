#!/usr/bin/env python
"""
Text Mining Pipeline - Simple clustering naming with TF-IDF and BM25
Input: Clustered data with text (tags, title)
Output: Cluster names and descriptions
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, List
import pandas as pd

from load_data import load_data
from cleaning import clean_data
from clustering import run_dbscan_geo, save_clustered_csv

from text_mining import (
    preprocess_text,
    extract_cluster_descriptions,
    extract_keywords_bm25,
    combine_cluster_names,
    save_descriptions_csv,
    print_cluster_descriptions,
)
from clustering import make_cluster_map_named


def main() -> None:
    # ---- Load and clean data ----
    print("\n" + "="*80)
    print("[1/4] Loading and cleaning data...")
    print("="*80)
    
    csv_path = "../flickr_data2.csv"
    df_raw, _ = load_data(csv_path)
    df_clean, _ = clean_data(df_raw)
    print(f"[OK] Loaded {len(df_clean):,} photos")

    # ---- Spatial clustering (DBSCAN) ----
    print("\n" + "="*80)
    print("[2/4] Running DBSCAN clustering...")
    print("="*80)
    
    df_dbscan, rep_dbscan = run_dbscan_geo(
        df_clean,
        eps_meters=50.0,
        min_samples=50,
        cluster_col="cluster",
        deduplicate_coords=True,
        coord_precision=4,
    )
    print(f"[OK] {rep_dbscan.n_clusters} clusters created, {rep_dbscan.noise_ratio*100:.1f}% noise")

    # ---- Text preprocessing ----
    print("\n" + "="*80)
    print("[3/4] Preprocessing text...")
    print("="*80)
    
    df_kmeans_text = preprocess_text(df_dbscan)
    print(f"[OK] Text preprocessed")

    # ---- Algorithm 1: TF-IDF ----
    print("\n" + "="*80)
    print("[4a/4] TEXT MINING - Algorithm 1: TF-IDF")
    print("="*80)
    
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
    print(f"\n[OK] TF-IDF saved to: {out_tfidf}")

    # ---- Algorithm 2: BM25 ----
    print("\n" + "="*80)
    print("[4b/4] TEXT MINING - Algorithm 2: BM25")
    print("="*80)
    
    try:
        bm25_descriptions = extract_keywords_bm25(
            df_kmeans_text,
            cluster_col="cluster",
            text_col="text",
            top_n_keywords=10,
        )
        
        print_cluster_descriptions(bm25_descriptions, top_n=5)
        
        # Save BM25 results
        out_bm25 = save_descriptions_csv(
            bm25_descriptions,
            "outputs/cluster_descriptions_bm25.csv"
        )
        print(f"\n[OK] BM25 saved to: {out_bm25}")
        
        # Combine both for final names
        cluster_names_combined = combine_tfidf_bm25(tfidf_descriptions, bm25_descriptions)
        
        print("\n" + "="*80)
        print("COMBINED CLUSTER NAMES (TF-IDF + BM25)")
        print("="*80)
        
        for cid, name in sorted(cluster_names_combined.items())[:15]:
            print(f"  Cluster {cid:2d}: {name}")
        print()

        # Save combined names
        names_df = pd.DataFrame([
            {"cluster_id": cid, "combined_name": name}
            for cid, name in cluster_names_combined.items()
        ])
        names_df.to_csv("outputs/cluster_names_tfidf_bm25.csv", index=False)
        print(f"[OK] Combined names saved to: outputs/cluster_names_tfidf_bm25.csv")
        
        # ---- Generate maps with cluster names ----
        print("\n" + "="*80)
        print("[5/5] GENERATING MAPS WITH CLUSTER NAMES")
        print("="*80)
        
        # Add cluster names to dataframe for map generation
        df_with_names = df_dbscan.copy()
        df_with_names["cluster_name"] = df_with_names["cluster"].map(
            lambda x: cluster_names_combined.get(x, f"Cluster {x}") if x != -1 else "Noise"
        )
        
        # Generate map with names
        map_output = make_cluster_map_named(
            df_with_names,
            lat_col="lat",
            lon_col="long",
            cluster_col="cluster",
            name_col="cluster_name",
            output_html="outputs/map_clusters_named_tfidf_bm25.html",
            sample_n=25000,
        )
        print(f"[OK] Named cluster map saved to: {map_output}")
        
    except Exception as e:
        print(f"\n[WARN] Map generation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Summary ----
    print("\n" + "="*80)
    print("TEXT MINING PIPELINE COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  - outputs/cluster_descriptions_tfidf.csv")
    print("  - outputs/cluster_descriptions_bm25.csv")
    print("  - outputs/cluster_names_tfidf_bm25.csv")
    print("  - outputs/map_clusters_named_tfidf_bm25.html")
    print("\nRead TEXT_MINING_COMPARISON.md for algorithm details!")
    print("="*80 + "\n")


def combine_tfidf_bm25(
    tfidf_descriptions,
    bm25_descriptions,
) -> Dict[int, str]:
    """
    Combine TF-IDF and BM25 descriptions for better cluster names.
    
    Strategy:
    1. Get top 3 keywords from each algorithm
    2. Filter out spam/garbage (single letters, URLs, gibberish patterns)
    3. Prefer human-readable place names (common nouns: musee, parc, gare, etc)
    4. Take best 2 keywords for final name
    """
    # Common spam patterns to filter
    SPAM_KEYWORDS = {
        # Single chars and very short
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        # Technical/gibberish
        'ldoll', 'bjd', 'img', 'pic', 'www', 'com', 'html', 'jpg', 'png',
        # Random hashes/codes
        '4be26f00b02ec9b6a1004dc0', 'squareformat', 'instax', 'paspaillons',
        # Nonsense combos
        'rise', 'endoftheworld', 'abodeofchaos', 'gesamtkuntwerk',
    }
    
    # Good POI keywords (prioritize these)
    POI_KEYWORDS = {
        'museum', 'musee', 'parc', 'park', 'gare', 'basilique', 'fourviere',
        'bellecour', 'confluence', 'presquile', 'vieux', 'lyon', 'place',
        'church', 'cathedral', 'square', 'rue', 'cours', 'theater', 'theatre',
        'market', 'marche', 'old', 'city', 'historic', 'quarter',
    }
    
    cluster_names = {}
    
    for tfidf_desc in tfidf_descriptions:
        cluster_id = tfidf_desc.cluster_id
        
        # Find matching BM25 description
        bm25_desc = next(
            (d for d in bm25_descriptions if d.cluster_id == cluster_id),
            None
        )
        
        if not bm25_desc:
            cluster_names[cluster_id] = tfidf_desc.description
            continue
        
        # Get top keywords from both, filter spam
        tfidf_words = [
            kw.lower() for kw in tfidf_desc.top_keywords[:5]
            if kw.lower() not in SPAM_KEYWORDS and len(kw) > 2
        ]
        bm25_words = [
            kw.lower() for kw in bm25_desc.top_keywords[:5]
            if kw.lower() not in SPAM_KEYWORDS and len(kw) > 2
        ]
        
        # Prefer POI keywords over generic ones
        prioritized_words = []
        
        # First: POI keywords from both
        for word in tfidf_words + bm25_words:
            if word in POI_KEYWORDS and word not in prioritized_words:
                prioritized_words.append(word)
        
        # Then: non-spam keywords
        for word in tfidf_words + bm25_words:
            if word not in prioritized_words:
                prioritized_words.append(word)
        
        # Build final name (top 2 different keywords)
        final_keywords = []
        for word in prioritized_words:
            if word not in final_keywords:
                final_keywords.append(word)
            if len(final_keywords) >= 2:
                break
        
        if final_keywords:
            name = " & ".join([kw.capitalize() for kw in final_keywords[:2]])
        else:
            name = f"Cluster {cluster_id}"
        
        cluster_names[cluster_id] = name
    
    return cluster_names


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
