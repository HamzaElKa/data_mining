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
        print("[WARN] NLTK not installed. Using basic French/English stopwords.")
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


def generate_cluster_title(keywords: List[str], max_words: int = 4) -> str:
    """
    Generate a clean POI title from TF-IDF keywords.
    
    Rules:
    1. Take top 1-3 keywords (bigrams prioritized)
    2. Capitalize first letter of each word
    3. Remove duplicates
    4. Max 4 words for readability
    
    Examples:
    ---------
    ["place bellecour", "statue", "louis"] → "Place Bellecour"
    ["fourviere", "basilique", "notre dame"] → "Basilique Fourviere"
    ["vieux lyon", "traboules", "medieval"] → "Vieux Lyon"
    """
    if not keywords:
        return "Unknown POI"
    
    # Priority: bigrams (2 words) are better POI names
    bigrams = [kw for kw in keywords[:5] if ' ' in kw]
    unigrams = [kw for kw in keywords[:5] if ' ' not in kw]
    
    selected = []
    word_count = 0
    
    # 1. Start with best bigram (if exists)
    if bigrams:
        best_bigram = bigrams[0]
        selected.append(best_bigram)
        word_count += len(best_bigram.split())
    
    # 2. Add top unigram if space left
    if unigrams and word_count < max_words:
        # Skip if already in bigram
        for unigram in unigrams:
            if word_count >= max_words:
                break
            # Check if not already present
            if not any(unigram in sel for sel in selected):
                selected.append(unigram)
                word_count += 1
    
    # 3. If no bigram, take top 2-3 unigrams
    if not selected:
        selected = unigrams[:min(3, max_words)]
    
    # 4. Join and capitalize
    title = ' '.join(selected)
    
    # Capitalize first letter of each word
    title = ' '.join(word.capitalize() for word in title.split())
    
    # Special cases for French (de, du, des, le, la, les)
    title = title.replace(' De ', ' de ')
    title = title.replace(' Du ', ' du ')
    title = title.replace(' Des ', ' des ')
    title = title.replace(' Le ', ' le ')
    title = title.replace(' La ', ' la ')
    title = title.replace(' Les ', ' les ')
    title = title.replace(' Au ', ' au ')
    title = title.replace(' Et ', ' et ')
    
    return title


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


if __name__ == "__main__":
    # Test TF-IDF on clustered data
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
        
        print("Extracting cluster descriptions (TF-IDF)...")
        descriptions = extract_cluster_descriptions(df_clustered, top_n_keywords=10)
        
        print_cluster_descriptions(descriptions, top_n=10)
        
        out_csv = save_descriptions_csv(descriptions)
        print(f"\n[OK] Descriptions saved to: {out_csv}")
        
        # Create wordcloud for top 3 clusters
        print("\nGenerating wordclouds for top 3 clusters...")
        for desc in descriptions[:3]:
            out_img = create_wordcloud_for_cluster(df_clustered, desc.cluster_id)
            if out_img:
                print(f"[OK] Wordcloud saved: {out_img}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
