# src/text_mining.py
# Session 2 — Text pattern mining to describe clusters
# Input: df_clustered with columns: cluster, tags, title
# Output: cluster descriptions using TF-IDF keywords

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ClusterDescription:
    cluster_id: int
    n_photos: int
    top_keywords: List[str]
    tfidf_scores: List[float]
    description: str
    cluster_title: str  # NEW: Auto-generated POI name


def preprocess_text(
    df: pd.DataFrame,
    *,
    tags_col: str = "tags",
    title_col: str = "title",
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Preprocess text data for TF-IDF analysis.
    
    Steps:
    1. Concatenate tags + title
    2. Remove URLs, emails, hashtags
    3. Unidecode (é → e, ç → c) for accent normalization
    4. Lowercase
    5. Remove spam patterns (camera models, websites, social media)
    6. Remove special characters
    7. Remove multiple spaces
    
    Returns:
    --------
    df with new column 'text' (preprocessed and cleaned)
    """
    import re
    
    df = df.copy()
    
    # Ensure tags and title are strings
    if tags_col in df.columns:
        df[tags_col] = df[tags_col].fillna("").astype(str)
    else:
        df[tags_col] = ""
    
    if title_col in df.columns:
        df[title_col] = df[title_col].fillna("").astype(str)
    else:
        df[title_col] = ""
    
    # Concatenate tags + title
    df[text_col] = (df[tags_col] + " " + df[title_col]).str.strip()
    
    # 1. Remove URLs (http://, https://, www.)
    df[text_col] = df[text_col].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '', regex=True
    )
    df[text_col] = df[text_col].str.replace(r'www\.[^\s]+', '', regex=True)
    
    # 2. Remove emails
    df[text_col] = df[text_col].str.replace(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '', regex=True
    )
    
    # 3. Remove hashtags (keep the word, remove #)
    df[text_col] = df[text_col].str.replace(r'#', '', regex=False)
    
    # 4. Replace commas in tags with spaces (tags are comma-separated)
    df[text_col] = df[text_col].str.replace(',', ' ', regex=False)
    
    # 5. Unidecode for accent normalization (é→e, ç→c, ü→u)
    try:
        from unidecode import unidecode
        df[text_col] = df[text_col].apply(lambda x: unidecode(x) if x else "")
    except ImportError:
        # Fallback: manual replacements for French accents
        replacements = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ô': 'o', 'ö': 'o',
            'î': 'i', 'ï': 'i',
            'ç': 'c',
        }
        for old, new in replacements.items():
            df[text_col] = df[text_col].str.replace(old, new, regex=False)
    
    # 6. Lowercase AFTER unidecode (preserve É → e)
    df[text_col] = df[text_col].str.lower()
    
    # 7. Remove spam patterns (camera models, social media handles)
    spam_patterns = [
        r'\bnikon\s*d\d{2,4}\b',  # nikon d750, nikon d7100
        r'\bcanon\s*eos\s*\d+d?\b',  # canon eos 5d, canon eos 70d
        r'\bsigma\s*\d+mm\b',  # sigma 50mm
        r'\btamron\s*\d+mm\b',
        r'\binstagram\b',
        r'\bfollow\s*me\b',
        r'\bcheck\s*out\b',
        r'\blike\s*and\s*subscribe\b',
        r'\b[a-z]+\.com\b',  # any .com domain
        r'\b[a-z]+\.fr\b',   # any .fr domain
    ]
    for pattern in spam_patterns:
        df[text_col] = df[text_col].str.replace(pattern, '', regex=True)
    
    # 8. Remove special characters (keep only letters, numbers, spaces)
    df[text_col] = df[text_col].str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    
    # 9. Remove multiple spaces
    df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True)
    
    # 10. Strip leading/trailing spaces
    df[text_col] = df[text_col].str.strip()
    
    return df


def get_stopwords(languages: List[str] = ["french", "english"]) -> set:
    """
    Get stop words for multiple languages.
    
    Parameters:
    -----------
    languages : list of str
        Languages to include (e.g., ["french", "english"])
    
    Returns:
    --------
    set of stop words
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if not already downloaded
        try:
            stopwords.words("english")
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)
        
        stop_words = set()
        for lang in languages:
            stop_words.update(stopwords.words(lang))
        
    except ImportError:
        # Basic fallback stop words
        stop_words = {
            # French
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "à", "au", "aux",
            "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
            "notre", "nos", "votre", "vos", "leur", "leurs", "je", "tu", "il", "elle", "nous", "vous",
            "ils", "elles", "on", "pour", "par", "avec", "dans", "sur", "sous", "entre", "vers", "chez",
            "que", "qui", "quoi", "dont", "où", "quand", "comment", "pourquoi",
            # English
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "from",
            "by", "with", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could", "may", "might", "must",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        }
    
    # Add custom stop words (too frequent, not meaningful)
    custom_stop = {
        "lyon", "photo", "picture", "flickr", "image", "photography", "france", "rhone",
        "rhône", "pic", "img", "photographer", "camera", "nikon", "canon", "sigma",
    }
    stop_words.update(custom_stop)
    
    return stop_words


def extract_cluster_descriptions(
    df_clustered: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    text_col: str = "text",
    top_n_keywords: int = 10,
    min_df: int = 2,
    max_df: float = 0.8,
) -> List[ClusterDescription]:
    """
    Extract TF-IDF keywords for each cluster to describe areas of interest.
    
    Parameters:
    -----------
    top_n_keywords : int
        Number of top keywords to extract per cluster
    min_df : int
        Minimum document frequency (ignore rare words)
    max_df : float
        Maximum document frequency (ignore very common words)
    
    Returns:
    --------
    List of ClusterDescription objects
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Filter out noise (-1)
    df = df_clustered[df_clustered[cluster_col] != -1].copy()
    
    if text_col not in df.columns:
        df = preprocess_text(df, text_col=text_col)
    
    # Group by cluster and concatenate all texts
    cluster_texts = df.groupby(cluster_col)[text_col].apply(lambda x: " ".join(x)).to_dict()
    
    if not cluster_texts:
        return []
    
    cluster_ids = sorted(cluster_texts.keys())
    documents = [cluster_texts[cid] for cid in cluster_ids]
    
    # Get stop words
    stop_words = get_stopwords(["french", "english"])
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=5000,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),  # Include bigrams (e.g., "place bellecour")
        token_pattern=r'\b[a-z]{3,}\b',  # Min 3 chars, only a-z (unidecode already applied)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        print(f"[WARN] TF-IDF failed: {e}")
        return []
    
    feature_names = vectorizer.get_feature_names_out()
    
    descriptions = []
    
    for idx, cluster_id in enumerate(cluster_ids):
        # Get TF-IDF scores for this cluster
        scores = tfidf_matrix[idx].toarray().flatten()
        
        # Get top N keywords
        top_indices = scores.argsort()[-top_n_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        # Count photos in cluster
        n_photos = int((df[cluster_col] == cluster_id).sum())
        
        # Generate POI title (NEW!)
        cluster_title = generate_cluster_title(top_keywords, max_words=4)
        
        # Generate description (use top 3 keywords)
        if top_keywords:
            description = f"{cluster_title}: {', '.join(top_keywords[:3])}"
        else:
            description = f"Cluster {cluster_id}"
        
        descriptions.append(ClusterDescription(
            cluster_id=int(cluster_id),
            n_photos=n_photos,
            top_keywords=top_keywords,
            tfidf_scores=top_scores,
            description=description,
            cluster_title=cluster_title,  # NEW!
        ))
    
    return descriptions


def save_descriptions_csv(
    descriptions: List[ClusterDescription],
    output_path: str = "outputs/cluster_descriptions.csv",
) -> str:
    """
    Save cluster descriptions to CSV.
    
    Returns:
    --------
    output_path : str
    """
    import os
    
    data = []
    for desc in descriptions:
        data.append({
            "cluster_id": desc.cluster_id,
            "poi_name": desc.cluster_title,  # ← TITRE POI !
            "n_photos": desc.n_photos,
            "top_keywords": ", ".join(desc.top_keywords),
            "tfidf_scores": ", ".join(f"{s:.3f}" for s in desc.tfidf_scores),
            "description": desc.description,
        })
    
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return output_path


def print_cluster_descriptions(descriptions: List[ClusterDescription], top_n: int = 10):
    """
    Print cluster descriptions in a readable format.
    """
    print("\n" + "="*80)
    print("CLUSTER DESCRIPTIONS (TF-IDF Keywords)")
    print("="*80)
    
    for desc in descriptions[:top_n]:
        print(f"\n{desc.description}")
        print(f"  Photos: {desc.n_photos}")
        print(f"  Keywords:")
        for kw, score in zip(desc.top_keywords[:5], desc.tfidf_scores[:5]):
            print(f"    - {kw:20s} (score: {score:.4f})")
    
    print("="*80)


def create_wordcloud_for_cluster(
    df_clustered: pd.DataFrame,
    cluster_id: int,
    *,
    cluster_col: str = "cluster",
    text_col: str = "text",
    output_path: str = None,
    max_words: int = 50,
) -> str:
    """
    Create a word cloud visualization for a specific cluster.
    
    Parameters:
    -----------
    cluster_id : int
        Cluster to visualize
    output_path : str, optional
        Path to save image (default: outputs/wordcloud_cluster_{id}.png)
    max_words : int
        Maximum number of words in cloud
    
    Returns:
    --------
    output_path : str
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] wordcloud or matplotlib not installed. Skipping wordcloud generation.")
        return None
    
    import os
    
    # Filter cluster
    df = df_clustered[df_clustered[cluster_col] == cluster_id].copy()
    
    if len(df) == 0:
        print(f"[WARN] Cluster {cluster_id} has no photos.")
        return None
    
    if text_col not in df.columns:
        df = preprocess_text(df, text_col=text_col)
    
    # Concatenate all texts
    text = " ".join(df[text_col].tolist())
    
    # Get stop words
    stop_words = get_stopwords(["french", "english"])
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stop_words,
        max_words=max_words,
        relative_scaling=0.5,
        colormap="viridis",
    ).generate(text)
    
    # Save
    if output_path is None:
        output_path = f"outputs/wordcloud_cluster_{cluster_id}.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Cluster {cluster_id} - Word Cloud", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path


def combine_cluster_names(
    tfidf_descriptions: List[ClusterDescription],
    frequency_keywords: Dict[int, List[Tuple[str, int]]],
) -> Dict[int, str]:
    """
    Combine TF-IDF and Frequency-based naming.
    
    Strategy:
    1. Use TF-IDF for discriminative keywords (what makes cluster unique)
    2. Use Frequency for common/recognizable terms (place names)
    3. Combine top 1-2 from each for final name
    
    Returns:
    --------
    Dict mapping cluster_id -> cluster_name (str)
    """
    cluster_names = {}
    
    for desc in tfidf_descriptions:
        cluster_id = desc.cluster_id
        
        # Top TF-IDF keywords (tend to be unique to cluster)
        tfidf_keywords = desc.top_keywords[:2]
        
        # Top frequency keywords for this cluster
        freq_keywords = []
        if cluster_id in frequency_keywords:
            freq_keywords = [kw for kw, _ in frequency_keywords[cluster_id][:2]]
        
        # Combine: prefer frequency keywords if they're recognizable place names
        # Otherwise use TF-IDF keywords
        all_keywords = freq_keywords + tfidf_keywords
        
        # Build name from first 2-3 non-duplicate keywords
        seen = set()
        final_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                final_keywords.append(kw)
                seen.add(kw)
            if len(final_keywords) >= 2:
                break
        
        # Capitalize and format
        if final_keywords:
            cluster_name = " & ".join([kw.title() for kw in final_keywords])
        else:
            cluster_name = f"POI {cluster_id}"
        
        cluster_names[cluster_id] = cluster_name
    
    return cluster_names


def add_cluster_names_to_dataframe(
    df_clustered: pd.DataFrame,
    cluster_names: Dict[int, str],
    *,
    cluster_col: str = "cluster",
    name_col: str = "cluster_name",
) -> pd.DataFrame:
    """
    Add cluster names as a new column in the dataframe.
    
    Parameters:
    -----------
    df_clustered : DataFrame with cluster column
    cluster_names : Dict mapping cluster_id -> cluster_name
    name_col : str
        Name of new column to add
    
    Returns:
    --------
    DataFrame with new 'cluster_name' column
    """
    df = df_clustered.copy()
    
    # Map cluster IDs to names
    df[name_col] = df[cluster_col].map(cluster_names)
    
    # Fill any unmapped clusters (shouldn't happen, but just in case)
    df[name_col] = df[name_col].fillna(df[cluster_col].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise"))
    
    return df


def save_named_clusters_csv(
    df_clustered: pd.DataFrame,
    output_path: str = "outputs/clustered_named.csv",
) -> str:
    """Save dataframe with cluster names to CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clustered.to_csv(output_path, index=False)
    return output_path


def print_named_clusters(
    df_clustered: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    name_col: str = "cluster_name",
    top_n: int = 15,
):
    """
    Print cluster names with photo counts.
    """
    print("\n" + "="*80)
    print("NAMED CLUSTERS (TF-IDF + Frequency Combined)")
    print("="*80)
    
    # Group by cluster and name
    cluster_info = df_clustered[df_clustered[cluster_col] != -1].groupby([cluster_col, name_col]).size().reset_index(name='n_photos')
    cluster_info = cluster_info.sort_values('n_photos', ascending=False)
    
    print(f"\nTotal clusters: {len(cluster_info)}")
    print(f"\nTop {top_n} clusters by size:\n")
    
    for idx, row in cluster_info.head(top_n).iterrows():
        cluster_id = row[cluster_col]
        name = row[name_col]
        n_photos = row['n_photos']
        
        bar_length = int(n_photos / cluster_info['n_photos'].max() * 40)
        bar = "█" * bar_length
        
        print(f"  {cluster_id:3d} | {name:30s} | {n_photos:5d} photos | {bar}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test both text mining algorithms
    try:
        from load_data import load_data
        from cleaning import clean_data
        from clustering import run_dbscan_geo
        
        print("Loading and cleaning data...")
        df_raw, _ = load_data("../flickr_data2.csv")
        df_clean, _ = clean_data(df_raw)
        
        print("Running DBSCAN clustering...")
        df_clustered, _ = run_dbscan_geo(
            df_clean,
            eps_meters=50.0,
            min_samples=50,
        )
        
        print("Preprocessing text...")
        df_clustered = preprocess_text(df_clustered)
        
        print("\n" + "="*80)
        print("TEXT MINING: ALGORITHM 1 - TF-IDF (Discriminative Keywords)")
        print("="*80)
        tfidf_descriptions = extract_cluster_descriptions(df_clustered, top_n_keywords=10)
        print_cluster_descriptions(tfidf_descriptions, top_n=10)
        
        out_csv_tfidf = save_descriptions_csv(tfidf_descriptions, "outputs/cluster_descriptions_tfidf.csv")
        print(f"\n[OK] TF-IDF descriptions saved to: {out_csv_tfidf}")
        
        print("\n" + "="*80)
        print("TEXT MINING: ALGORITHM 2 - Keyword Frequency (Recognizable Terms)")
        print("="*80)
        frequency_keywords = extract_keywords_by_frequency(df_clustered, top_n=10)
        
        print("\nTop frequency keywords by cluster:")
        for cluster_id in sorted(frequency_keywords.keys())[:10]:
            keywords = frequency_keywords[cluster_id]
            top_3 = ", ".join([f"{kw}({freq})" for kw, freq in keywords[:3]])
            print(f"  Cluster {cluster_id}: {top_3}")
        
        print("\n" + "="*80)
        print("COMBINING ALGORITHMS: TF-IDF + Frequency")
        print("="*80)
        cluster_names = combine_cluster_names(tfidf_descriptions, frequency_keywords)
        
        df_named = add_cluster_names_to_dataframe(df_clustered, cluster_names)
        print_named_clusters(df_named, top_n=15)
        
        out_csv_named = save_named_clusters_csv(df_named, "outputs/clustered_named.csv")
        print(f"\n[OK] Named clusters saved to: {out_csv_named}")
        
        # Create wordcloud for top 3 clusters
        print("\nGenerating wordclouds for top 3 clusters...")
        for cluster_id in sorted(frequency_keywords.keys())[:3]:
            out_img = create_wordcloud_for_cluster(df_clustered, cluster_id)
            if out_img:
                print(f"[OK] Wordcloud saved: {out_img}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
