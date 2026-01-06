# src/cleaning.py
# Data Mining project 2025-2026
# Cleaning module: robust, paramétrable, documenté
# Dataset Flickr Lyon — préparation pour clustering spatial + text mining + analyse temporelle

"""
Module de nettoyage robuste pour le dataset Flickr géolocalisé.

Pipeline de nettoyage:
1. Normalisation du schéma et des types
2. Nettoyage géographique (bbox Lyon, validation GPS, doublons coords)
3. Nettoyage temporel (parsing dates, validation, extraction features)
4. Nettoyage textuel (normalisation tags/descriptions)
5. Suppression des doublons
6. Génération du rapport de nettoyage

Objectif: dataset cohérent pour DBSCAN/KMeans/Hierarchical + TF-IDF + analyse temporelle
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# ===============================
# Configuration & Setup
# ===============================

@dataclass
class CleaningConfig:
    """
    Configuration paramétrable du nettoyage.
    Tous les filtres peuvent être activés/désactivés pour flexibilité.
    """
    # Bounding box Lyon (avec marge) — valeurs par défaut
    # Lyon centre: ~45.75N, 4.85E
    # Élargi pour Grand Lyon métropole (~59 communes, rayon 20-25km)
    # Inclut: Fourvière, Confluence, Part-Dieu, Aéroport Saint-Exupéry
    bbox_lat_min: float = 45.55
    bbox_lat_max: float = 45.95
    bbox_lon_min: float = 4.65
    bbox_lon_max: float = 5.15
    
    # Dates valides
    min_year: int = 1990  # Avant = suspect (Flickr créé en 2004)
    max_year: Optional[int] = None  # None = aujourd'hui
    
    # Filtres activables
    drop_missing_gps: bool = True
    drop_invalid_gps: bool = True  # lat/lon hors [-90,90] / [-180,180]
    filter_bbox: bool = True  # Restreindre à Lyon
    drop_future_dates: bool = True
    drop_invalid_dates: bool = True  # Dates < min_year ou > max_year
    drop_photo_id_duplicates: bool = True  # Garder 1 seule photo par photo_id
    drop_exact_duplicates: bool = True  # (photo_id, lat, lon, date) identiques
    
    # ⚠️  IMPORTANT: Ne PAS supprimer les photos sans date !
    # Les photos sans date restent utiles pour:
    # - Clustering spatial (Objectif 1) : GPS suffit
    # - Comparaison densités avec/sans contrainte temporelle (Objectif 3)
    # Le filtrage temporel se fera APRÈS dans les notebooks d'analyse
    drop_missing_event_date: bool = False  # Garder False sauf besoin spécifique
    
    # Sampling (pour tests rapides)
    sample_n: Optional[int] = None  # Si défini, prend N lignes aléatoires
    sample_frac: Optional[float] = None  # Si défini, prend frac% des données
    sample_seed: int = 42  # Reproductibilité
    
    # Output paths (optionnels)
    output_csv_path: Optional[str] = None
    output_parquet_path: Optional[str] = None
    output_report_path: Optional[str] = "cleaning_report.json"
    
    # Logging level
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure le logging pour traçabilité."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def make_default_config() -> CleaningConfig:
    """Retourne la configuration par défaut."""
    return CleaningConfig()


# ===============================
# A) Normalisation schéma & types
# ===============================

def _normalize_schema(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalise le schéma:
    - Vérifie présence colonnes critiques
    - Harmonise noms (déjà fait par load_data mais on sécurise)
    - Convertit types appropriés
    """
    logger.info("A) Normalisation du schéma et des types...")
    
    df = df.copy()
    
    # Nettoyer noms de colonnes (strip espaces)
    df.columns = df.columns.astype(str).str.strip()
    
    # Drop colonnes Unnamed (trailing commas CSV)
    unnamed_cols = df.columns[df.columns.str.match(r"^Unnamed")].tolist()
    if unnamed_cols:
        logger.info(f"Suppression colonnes vides: {unnamed_cols}")
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    
    # Colonnes attendues (mapping flexible)
    required_cols = {
        'photo_id': ['id', 'photo_id', 'photoid'],
        'user_id': ['user', 'user_id', 'userid', 'photographer_id'],
        'lat': ['lat', 'latitude'],
        'lon': ['long', 'lon', 'lng', 'longitude'],
    }
    
    # Mapper colonnes
    col_mapping = {}
    for standard_name, variants in required_cols.items():
        found = next((c for c in variants if c in df.columns), None)
        if found and found != standard_name:
            col_mapping[found] = standard_name
        elif not found:
            logger.error(f"Colonne critique manquante: aucun variant de '{standard_name}' trouvé dans {variants}")
            raise ValueError(f"Colonne critique '{standard_name}' absente du dataset")
    
    if col_mapping:
        logger.info(f"Renommage colonnes: {col_mapping}")
        df = df.rename(columns=col_mapping)
    
    # Conversion types
    logger.info("Conversion des types...")
    
    # IDs -> string (éviter overflow, cohérence)
    for col in ['photo_id', 'user_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # GPS -> float
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Harmoniser valeurs manquantes texte
    text_cols = [c for c in df.columns if c in ['tags', 'title', 'description']]
    for col in text_cols:
        if col in df.columns:
            # Remplacer chaînes vides par NaN
            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
    
    logger.info(f"Schéma normalisé: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ===============================
# B) Nettoyage géographique
# ===============================

def _clean_geographic(df: pd.DataFrame, config: CleaningConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Nettoyage géographique complet:
    - Supprime lignes sans GPS
    - Supprime coords invalides (hors bornes planète)
    - Filtre bbox Lyon
    - Détecte outliers (0,0) ou coords absurdes
    """
    logger.info("B) Nettoyage géographique...")
    
    stats = {
        'initial_rows': len(df),
        'missing_gps_dropped': 0,
        'invalid_gps_dropped': 0,
        'bbox_filtered_dropped': 0,
        'zero_zero_dropped': 0,
    }
    
    initial_len = len(df)
    
    # 1. GPS manquants
    if config.drop_missing_gps:
        before = len(df)
        df = df.dropna(subset=['lat', 'lon'])
        dropped = before - len(df)
        stats['missing_gps_dropped'] = dropped
        logger.info(f"  → {dropped:,} lignes sans GPS supprimées")
    
    # 2. GPS invalides (hors bornes planète)
    if config.drop_invalid_gps:
        before = len(df)
        mask_valid = (
            (df['lat'] >= -90) & (df['lat'] <= 90) &
            (df['lon'] >= -180) & (df['lon'] <= 180)
        )
        df = df[mask_valid].copy()
        dropped = before - len(df)
        stats['invalid_gps_dropped'] = dropped
        logger.info(f"  → {dropped:,} lignes avec GPS invalide supprimées")
    
    # 3. Points "0,0" suspects (Golfe de Guinée — très improbable pour Lyon)
    before = len(df)
    mask_zero = (df['lat'].abs() < 0.001) & (df['lon'].abs() < 0.001)
    df = df[~mask_zero].copy()
    dropped = before - len(df)
    stats['zero_zero_dropped'] = dropped
    if dropped > 0:
        logger.warning(f"  → {dropped:,} points suspects (0,0) supprimés")
    
    # 4. Bounding box Lyon
    if config.filter_bbox:
        before = len(df)
        mask_bbox = (
            (df['lat'] >= config.bbox_lat_min) &
            (df['lat'] <= config.bbox_lat_max) &
            (df['lon'] >= config.bbox_lon_min) &
            (df['lon'] <= config.bbox_lon_max)
        )
        df = df[mask_bbox].copy()
        dropped = before - len(df)
        stats['bbox_filtered_dropped'] = dropped
        logger.info(
            f"  → Bbox Lyon [{config.bbox_lat_min}, {config.bbox_lat_max}] × "
            f"[{config.bbox_lon_min}, {config.bbox_lon_max}]: {dropped:,} lignes hors zone"
        )
    
    # Stats finales
    stats['gps_min_lat'] = float(df['lat'].min()) if len(df) > 0 else None
    stats['gps_max_lat'] = float(df['lat'].max()) if len(df) > 0 else None
    stats['gps_min_lon'] = float(df['lon'].min()) if len(df) > 0 else None
    stats['gps_max_lon'] = float(df['lon'].max()) if len(df) > 0 else None
    
    total_dropped = initial_len - len(df)
    logger.info(f"Nettoyage géographique terminé: {total_dropped:,} lignes supprimées au total")
    
    return df, stats


# ===============================
# C) Nettoyage temporel
# ===============================

def _parse_dates(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Parse les dates depuis colonnes split (date_taken_*, date_upload_*).
    Crée colonnes datetime consolidées.
    """
    logger.info("C) Parsing des dates...")
    
    df = df.copy()
    
    # Fonction helper pour construire datetime
    def build_datetime(prefix: str) -> pd.Series:
        required = [f"{prefix}_year", f"{prefix}_month", f"{prefix}_day", 
                    f"{prefix}_hour", f"{prefix}_minute"]
        
        # Vérifier présence colonnes
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Colonnes manquantes pour {prefix}: {missing}")
            return pd.Series([pd.NaT] * len(df), index=df.index)
        
        # Construire DataFrame temporaire
        dt_df = pd.DataFrame({
            'year': pd.to_numeric(df[f"{prefix}_year"], errors='coerce'),
            'month': pd.to_numeric(df[f"{prefix}_month"], errors='coerce'),
            'day': pd.to_numeric(df[f"{prefix}_day"], errors='coerce'),
            'hour': pd.to_numeric(df[f"{prefix}_hour"], errors='coerce'),
            'minute': pd.to_numeric(df[f"{prefix}_minute"], errors='coerce'),
        })
        
        # Parser en datetime
        return pd.to_datetime(dt_df, errors='coerce')
    
    # Parser date_taken et date_upload
    df['date_taken_dt'] = build_datetime('date_taken')
    df['date_upload_dt'] = build_datetime('date_upload')
    
    taken_valid = df['date_taken_dt'].notna().sum()
    upload_valid = df['date_upload_dt'].notna().sum()
    
    logger.info(f"  → date_taken parsée: {taken_valid:,}/{len(df):,} ({taken_valid/len(df)*100:.1f}%)")
    logger.info(f"  → date_upload parsée: {upload_valid:,}/{len(df):,} ({upload_valid/len(df)*100:.1f}%)")
    
    return df


def _create_event_date(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Crée colonne 'event_date' de référence:
    - Priorité: date_taken si valide
    - Sinon: date_upload
    - Ajoute flag 'has_valid_date' pour filtrage flexible après clustering
    
    Important: On garde TOUTES les photos même sans date valide,
    car elles restent utiles pour clustering spatial (Objectif 1).
    Le filtrage temporel se fera APRÈS dans l'analyse (Objectif 3).
    """
    logger.info("Création de la date de référence 'event_date'...")
    
    df = df.copy()
    
    # Priorité date_taken, fallback date_upload
    df['event_date'] = df['date_taken_dt'].fillna(df['date_upload_dt'])
    
    # Flag pour filtrage flexible (crucial pour analyse temporelle)
    df['has_valid_date'] = df['event_date'].notna()
    
    valid = df['has_valid_date'].sum()
    logger.info(f"  → event_date créée: {valid:,}/{len(df):,} valides ({valid/len(df)*100:.1f}%)")
    logger.info(f"  → {len(df) - valid:,} photos gardées sans date (utiles pour clustering spatial)")
    
    return df


def _clean_dates(df: pd.DataFrame, config: CleaningConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Nettoyage dates:
    - Supprimer dates futures
    - Supprimer dates trop anciennes
    - Extraire features temporelles (year, month, day, etc.)
    """
    logger.info("Validation et nettoyage des dates...")
    
    stats = {
        'future_dates_dropped': 0,
        'too_old_dates_dropped': 0,
        'chronology_errors_dropped': 0,
        'missing_event_date_dropped': 0,
        'final_valid_event_dates': 0,
        'date_min': None,
        'date_max': None,
    }
    
    df = df.copy()
    initial_len = len(df)
    
    # Référence temporelle
    now = datetime.now()
    max_year = config.max_year if config.max_year else now.year
    
    # 1. Dates futures (contrôle event_date <= now)
    if config.drop_future_dates:
        before = len(df)
        # Supprimer si event_date > maintenant
        mask_future = df['event_date'] > pd.Timestamp(now)
        df = df[~mask_future].copy()
        dropped = before - len(df)
        stats['future_dates_dropped'] = dropped
        if dropped > 0:
            logger.warning(f"  → {dropped:,} lignes avec dates futures supprimées")
    
    # 1.5 Vérifier chronologie : upload >= taken (comme votre collègue)
    if config.drop_invalid_dates and 'date_taken_dt' in df.columns and 'date_upload_dt' in df.columns:
        before = len(df)
        mask_both_valid = df['date_taken_dt'].notna() & df['date_upload_dt'].notna()
        mask_chronology_ok = df['date_upload_dt'] >= df['date_taken_dt']
        # Supprimer seulement si les deux dates existent ET upload < taken
        mask_chronology_error = mask_both_valid & ~mask_chronology_ok
        df = df[~mask_chronology_error].copy()
        dropped = before - len(df)
        stats['chronology_errors_dropped'] = dropped
        if dropped > 0:
            logger.warning(f"  → {dropped:,} lignes avec upload < taken (erreur chronologie)")
    
    # 2. Dates trop anciennes (min_year) ou trop récentes (max_year)
    if config.drop_invalid_dates:
        before = len(df)
        # Supprimer si year < min_year OU year > max_year
        mask_too_old = df['event_date'].dt.year < config.min_year
        mask_too_new = df['event_date'].dt.year > max_year
        mask_invalid = (mask_too_old | mask_too_new).fillna(False)
        df = df[~mask_invalid].copy()
        dropped = before - len(df)
        stats['too_old_dates_dropped'] = dropped
        if dropped > 0:
            logger.warning(f"  → {dropped:,} lignes avec dates hors [{config.min_year}, {max_year}] supprimées")
    
    # 3. Supprimer lignes sans event_date si configuré
    if config.drop_missing_event_date:
        before = len(df)
        df = df.dropna(subset=['event_date']).copy()
        dropped = before - len(df)
        stats['missing_event_date_dropped'] = dropped
        if dropped > 0:
            logger.info(f"  → {dropped:,} lignes sans event_date valide supprimées")
    
    # Stats finales
    valid_dates = df['event_date'].notna()
    stats['final_valid_event_dates'] = int(valid_dates.sum())
    
    if valid_dates.any():
        stats['date_min'] = str(df.loc[valid_dates, 'event_date'].min())
        stats['date_max'] = str(df.loc[valid_dates, 'event_date'].max())
    
    logger.info(f"  → Période finale: {stats['date_min']} → {stats['date_max']}")
    logger.info(f"  → Dates valides: {stats['final_valid_event_dates']:,}/{len(df):,}")
    
    return df, stats


def _extract_temporal_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Extrait features temporelles depuis event_date:
    - year, month, day, week, hour
    - date (sans heure) pour agrégations
    """
    logger.info("Extraction des features temporelles...")
    
    df = df.copy()
    
    # Extraire uniquement où event_date existe
    mask = df['event_date'].notna()
    
    df['year'] = pd.NA
    df['month'] = pd.NA
    df['day'] = pd.NA
    df['week'] = pd.NA
    df['hour'] = pd.NA
    df['date'] = pd.NaT
    
    if mask.any():
        df.loc[mask, 'year'] = df.loc[mask, 'event_date'].dt.year
        df.loc[mask, 'month'] = df.loc[mask, 'event_date'].dt.month
        df.loc[mask, 'day'] = df.loc[mask, 'event_date'].dt.day
        df.loc[mask, 'week'] = df.loc[mask, 'event_date'].dt.isocalendar().week
        df.loc[mask, 'hour'] = df.loc[mask, 'event_date'].dt.hour
        df.loc[mask, 'date'] = df.loc[mask, 'event_date'].dt.floor('D')
    
    # Convertir en Int64 (nullable integer)
    for col in ['year', 'month', 'day', 'week', 'hour']:
        df[col] = df[col].astype('Int64')
    
    logger.info("  → Features créées: year, month, day, week, hour, date")
    
    return df


# ===============================
# D) Nettoyage textuel
# ===============================

def _clean_text(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Nettoyage et normalisation texte:
    - Tags: parsing, nettoyage, normalisation
    - Titre/description: nettoyage HTML, URLs
    - Création colonne text_merged (concat tags + title + description)
    """
    logger.info("D) Nettoyage textuel...")
    
    df = df.copy()
    
    # 1. Normaliser tags
    df = _normalize_tags(df, logger)
    
    # 2. Normaliser titre et description
    for col in ['title', 'description']:
        if col in df.columns:
            df[col] = _normalize_text_field(df[col], logger, col)
    
    # 3. Créer text_merged (pour TF-IDF ultérieur)
    df = _create_text_merged(df, logger)
    
    return df


def _normalize_tags(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalise les tags:
    - Split si string (séparateurs: espace, virgule)
    - Nettoyage (minuscules, trim, filtre vides/trop courts)
    - Garde liste de tags propres ET version string rejointe
    """
    logger.info("  → Normalisation des tags...")
    
    df = df.copy()
    
    if 'tags' not in df.columns:
        logger.warning("Colonne 'tags' absente, création colonne vide")
        df['tags_clean'] = ""
        return df
    
    # Optimisation: opérations 100% vectorisées pandas (pas de .apply())
    tags_series = (
        df['tags']
        .fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'["\',]', ' ', regex=True)  # Guillemets + virgules → espaces
        .str.replace(r'\s+', ' ', regex=True)     # Espaces multiples → 1 espace
        .str.strip()
    )
    
    # Filtrer mots d'1 caractère avec regex (PLUS RAPIDE que split/join)
    df['tags_clean'] = tags_series.str.replace(r'\b\w{1}\b', '', regex=True).str.strip()
    # Note: garde mots >= 2 caractères automatiquement
    
    non_empty = (df['tags_clean'] != "").sum()
    logger.info(f"    {non_empty:,}/{len(df):,} lignes avec tags nettoyés ({non_empty/len(df)*100:.1f}%)")
    
    return df


def _normalize_text_field(series: pd.Series, logger: logging.Logger, col_name: str) -> pd.Series:
    """
    Nettoie un champ texte (titre ou description):
    - Minuscules
    - Enlever HTML
    - Enlever URLs
    - Normaliser espaces
    """
    def clean_text(text):
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        if not text:
            return ""
        
        # Minuscules
        text = text.lower()
        
        # Enlever HTML (tags simples)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Enlever URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        
        # Normaliser espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    cleaned = series.apply(clean_text)
    non_empty = (cleaned != "").sum()
    logger.info(f"    {col_name}: {non_empty:,}/{len(cleaned):,} non-vides après nettoyage")
    
    return cleaned


def _create_text_merged(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Crée colonne text_merged = tags_clean + title + description.
    Utile pour TF-IDF et text mining ultérieurs.
    """
    logger.info("  → Création de 'text_merged'...")
    
    df = df.copy()
    
    # Colonnes texte disponibles
    text_parts = []
    
    if 'tags_clean' in df.columns:
        text_parts.append(df['tags_clean'].fillna(''))
    
    if 'title' in df.columns:
        text_parts.append(df['title'].fillna(''))
    
    if 'description' in df.columns:
        text_parts.append(df['description'].fillna(''))
    
    # Concaténer avec espace
    if text_parts:
        df['text_merged'] = text_parts[0]
        for part in text_parts[1:]:
            df['text_merged'] = df['text_merged'] + ' ' + part
        
        # Nettoyer espaces multiples
        df['text_merged'] = df['text_merged'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        df['text_merged'] = ""
    
    non_empty = (df['text_merged'] != "").sum()
    logger.info(f"    text_merged créé: {non_empty:,}/{len(df):,} non-vides ({non_empty/len(df)*100:.1f}%)")
    
    return df


# ===============================
# D-bis) Preprocessing avancé pour text mining (OPTIONNEL)
# ===============================

def preprocess_for_text_mining(
    df: pd.DataFrame,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
    languages: List[str] = ['french', 'english'],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Preprocessing avancé pour TF-IDF et text mining (OPTIONNEL).
    
    ⚠️  À UTILISER dans les notebooks d'analyse, PAS dans le cleaning de base.
    
    Args:
        df: DataFrame avec colonne 'text_merged'
        remove_stopwords: Enlever mots vides FR/EN
        apply_stemming: Appliquer stemming (racines)
        languages: Langues pour stopwords/stemming
        logger: Logger (optionnel)
    
    Returns:
        DataFrame avec colonne 'text_processed' ajoutée
    
    Note: Nécessite: pip install nltk
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Preprocessing avancé pour text mining (OPTIONNEL)...")
    
    df = df.copy()
    
    # Check colonne text_merged
    if 'text_merged' not in df.columns:
        logger.warning("Colonne 'text_merged' absente, création colonne vide")
        df['text_processed'] = ""
        return df
    
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer
        
        # Télécharger ressources NLTK si nécessaire
        try:
            stopwords.words('french')
        except LookupError:
            logger.info("Téléchargement ressources NLTK...")
            nltk.download('stopwords', quiet=True)
        
        # Préparer stopwords multi-langues
        stops = set()
        if remove_stopwords:
            for lang in languages:
                try:
                    stops.update(stopwords.words(lang))
                except:
                    logger.warning(f"Stopwords '{lang}' non disponibles")
        
        # Préparer stemmer (français par défaut)
        stemmer = SnowballStemmer('french') if apply_stemming else None
        
        def process_text(text):
            if not text or pd.isna(text):
                return ""
            
            words = str(text).split()
            
            # Filtrer mots courts et stopwords
            filtered = [w for w in words if len(w) >= 3 and w not in stops]
            
            # Stemming optionnel
            if stemmer:
                filtered = [stemmer.stem(w) for w in filtered]
            
            return ' '.join(filtered)
        
        # Appliquer (unavoidable .apply() ici car stemming complexe)
        logger.info("  → Application stemming + stopwords (ceci peut prendre du temps)...")
        df['text_processed'] = df['text_merged'].apply(process_text)
        
        non_empty = (df['text_processed'] != "").sum()
        logger.info(f"  → text_processed créé: {non_empty:,}/{len(df):,} non-vides")
        logger.info(f"  → Exemple: '{df['text_merged'].iloc[0][:50]}...' → '{df['text_processed'].iloc[0][:50]}...'")
        
    except ImportError:
        logger.error("NLTK non installé. Installez avec: pip install nltk")
        df['text_processed'] = df['text_merged']  # Fallback
    
    return df


# ===============================
# E) Suppression doublons
# ===============================

def _remove_duplicates(df: pd.DataFrame, config: CleaningConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Supprime les doublons:
    - Doublons photo_id: garder 1 seule occurrence
    - Doublons exacts (photo_id + lat + lon + event_date): supprimer
    """
    logger.info("E) Suppression des doublons...")
    
    stats = {
        'photo_id_duplicates_dropped': 0,
        'exact_duplicates_dropped': 0,
    }
    
    df = df.copy()
    initial_len = len(df)
    
    # 1. Doublons photo_id (garder première occurrence)
    if config.drop_photo_id_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=['photo_id'], keep='first')
        dropped = before - len(df)
        stats['photo_id_duplicates_dropped'] = dropped
        logger.info(f"  → {dropped:,} doublons photo_id supprimés (gardé 1ère occurrence)")
    
    # 2. Doublons exacts (même photo_id + coords + date)
    if config.drop_exact_duplicates:
        before = len(df)
        dup_cols = ['photo_id', 'lat', 'lon']
        if 'event_date' in df.columns:
            dup_cols.append('event_date')
        
        df = df.drop_duplicates(subset=dup_cols, keep='first')
        dropped = before - len(df)
        stats['exact_duplicates_dropped'] = dropped
        if dropped > 0:
            logger.info(f"  → {dropped:,} doublons exacts supprimés")
    
    total_dropped = initial_len - len(df)
    logger.info(f"Suppression doublons terminée: {total_dropped:,} lignes supprimées")
    
    return df, stats


# ===============================
# F) Sampling (optionnel)
# ===============================

def _apply_sampling(df: pd.DataFrame, config: CleaningConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict]:
    """
    Applique sampling si configuré (pour tests rapides).
    """
    stats = {
        'sampling_applied': False,
        'sample_size': len(df),
        'sample_seed': config.sample_seed,
    }
    
    if config.sample_n is not None:
        logger.info(f"Sampling: prendre {config.sample_n} lignes aléatoires (seed={config.sample_seed})")
        if config.sample_n < len(df):
            df = df.sample(n=config.sample_n, random_state=config.sample_seed)
            stats['sampling_applied'] = True
            stats['sample_size'] = config.sample_n
    
    elif config.sample_frac is not None:
        logger.info(f"Sampling: prendre {config.sample_frac*100:.1f}% des lignes (seed={config.sample_seed})")
        if config.sample_frac < 1.0:
            df = df.sample(frac=config.sample_frac, random_state=config.sample_seed)
            stats['sampling_applied'] = True
            stats['sample_size'] = len(df)
    
    return df, stats


# ===============================
# G) Validation post-nettoyage
# ===============================

def _validate_cleaned_data(df: pd.DataFrame, config: CleaningConfig, logger: logging.Logger) -> Dict:
    """
    Validation automatique après nettoyage:
    - Pas de NaN lat/lon
    - lat/lon dans bbox si activé
    - event_date valides
    - Pas de duplicates photo_id
    """
    logger.info("F) Validation du dataset nettoyé...")
    
    validation = {
        'passed': True,
        'checks': {},
    }
    
    # Check 1: Pas de NaN GPS
    gps_missing = df[['lat', 'lon']].isna().any(axis=1).sum()
    validation['checks']['gps_no_nan'] = bool(gps_missing == 0)
    if gps_missing > 0:
        logger.error(f"❌ VALIDATION FAILED: {gps_missing} lignes avec GPS NaN détectées après nettoyage!")
        validation['passed'] = False
    else:
        logger.info("  ✓ Pas de GPS NaN")
    
    # Check 2: GPS dans bbox si filtre activé
    if config.filter_bbox:
        out_of_bbox = (
            (df['lat'] < config.bbox_lat_min) | (df['lat'] > config.bbox_lat_max) |
            (df['lon'] < config.bbox_lon_min) | (df['lon'] > config.bbox_lon_max)
        ).sum()
        validation['checks']['gps_in_bbox'] = bool(out_of_bbox == 0)
        if out_of_bbox > 0:
            logger.error(f"❌ VALIDATION FAILED: {out_of_bbox} lignes hors bbox après filtrage!")
            validation['passed'] = False
        else:
            logger.info("  ✓ Toutes les coords dans bbox Lyon")
    
    # Check 3: event_date valides (au moins X%)
    valid_dates_pct = (df['event_date'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
    validation['checks']['event_date_valid_pct'] = float(valid_dates_pct)
    if valid_dates_pct < 50:
        logger.warning(f"⚠️  Seulement {valid_dates_pct:.1f}% des lignes ont une event_date valide")
    else:
        logger.info(f"  ✓ {valid_dates_pct:.1f}% des lignes ont event_date valide")
    
    # Check 4: Pas de duplicates photo_id
    if config.drop_photo_id_duplicates:
        dup_count = df['photo_id'].duplicated().sum()
        validation['checks']['no_photo_id_duplicates'] = bool(dup_count == 0)
        if dup_count > 0:
            logger.error(f"❌ VALIDATION FAILED: {dup_count} doublons photo_id détectés!")
            validation['passed'] = False
        else:
            logger.info("  ✓ Pas de doublons photo_id")
    
    if validation['passed']:
        logger.info("✓ Validation réussie: dataset prêt pour clustering et text mining")
    else:
        logger.error("❌ Validation échouée: problèmes détectés dans le dataset nettoyé")
    
    return validation


# ===============================
# Rapport de nettoyage
# ===============================

def _generate_cleaning_report(
    initial_rows: int,
    final_df: pd.DataFrame,
    geo_stats: Dict,
    date_stats: Dict,
    dup_stats: Dict,
    sample_stats: Dict,
    validation: Dict,
    config: CleaningConfig,
    logger: logging.Logger
) -> Dict:
    """
    Génère un rapport complet du nettoyage.
    """
    logger.info("Génération du rapport de nettoyage...")
    
    final_rows = len(final_df)
    
    # Top tags (pour détecter bruit)
    top_tags = []
    total_unique_tags = 0
    if 'tags_clean' in final_df.columns:
        # Extraire tous les tags
        all_tags = []
        for tags_str in final_df['tags_clean'].dropna():
            if tags_str:
                all_tags.extend(tags_str.split())
        
        if all_tags:
            from collections import Counter
            tag_counts = Counter(all_tags)
            top_tags = [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(20)]
            total_unique_tags = len(tag_counts)
    
    # Missing values par colonnes importantes
    important_cols = ['photo_id', 'user_id', 'lat', 'lon', 'event_date', 'tags_clean', 'text_merged']
    missing_by_col = {}
    for col in important_cols:
        if col in final_df.columns:
            if col in ['tags_clean', 'text_merged']:
                # Pour texte: compter les vides
                missing = (final_df[col].isna() | (final_df[col] == "")).sum()
            else:
                missing = final_df[col].isna().sum()
            missing_pct = (missing / final_rows * 100) if final_rows > 0 else 0
            missing_by_col[col] = {
                'count': int(missing),
                'percentage': round(missing_pct, 2)
            }
    
    # Stats clustering readiness (CRUCIAL pour milestones)
    rows_with_gps = int((~final_df[['lat', 'lon']].isna().any(axis=1)).sum()) if final_rows > 0 else 0
    rows_with_date = int(final_df.get('has_valid_date', pd.Series([False]*len(final_df))).sum())
    rows_with_text = int((final_df['text_merged'] != "").sum()) if 'text_merged' in final_df.columns else 0
    
    # Surface bbox (approximation)
    bbox_area_km2 = round(
        (config.bbox_lat_max - config.bbox_lat_min) * 111 *
        (config.bbox_lon_max - config.bbox_lon_min) * 111 * 
        np.cos(np.radians((config.bbox_lat_max + config.bbox_lat_min)/2)),
        2
    )
    
    # Moyenne mots par photo
    avg_words = 0
    if 'text_merged' in final_df.columns and final_rows > 0:
        word_counts = final_df['text_merged'].str.split().str.len()
        avg_words = round(word_counts.mean(), 1) if not word_counts.isna().all() else 0
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'bbox': {
                'lat_min': config.bbox_lat_min,
                'lat_max': config.bbox_lat_max,
                'lon_min': config.bbox_lon_min,
                'lon_max': config.bbox_lon_max,
            },
            'min_year': config.min_year,
            'max_year': config.max_year,
            'filters_enabled': {
                'drop_missing_gps': bool(config.drop_missing_gps),
                'drop_invalid_gps': bool(config.drop_invalid_gps),
                'filter_bbox': bool(config.filter_bbox),
                'drop_future_dates': bool(config.drop_future_dates),
                'drop_invalid_dates': bool(config.drop_invalid_dates),
                'drop_photo_id_duplicates': bool(config.drop_photo_id_duplicates),
                'drop_exact_duplicates': bool(config.drop_exact_duplicates),
                'drop_missing_event_date': bool(config.drop_missing_event_date),
            }
        },
        'rows': {
            'initial': initial_rows,
            'final': final_rows,
            'dropped_total': initial_rows - final_rows,
            'retention_rate': round((final_rows / initial_rows * 100) if initial_rows > 0 else 0, 2)
        },
        'geographic_cleaning': geo_stats,
        'date_cleaning': date_stats,
        'duplicates_cleaning': dup_stats,
        'sampling': sample_stats,
        'missing_values': missing_by_col,
        'top_tags': top_tags,
        'validation': validation,
        'clustering_readiness': {
            'rows_with_valid_gps': rows_with_gps,
            'rows_with_valid_date': rows_with_date,
            'rows_with_text': rows_with_text,
            'bbox_area_km2': bbox_area_km2,
            'density_photos_per_km2': round(final_rows / bbox_area_km2, 2) if bbox_area_km2 > 0 else 0,
        },
        'text_mining_readiness': {
            'avg_words_per_photo': avg_words,
            'total_unique_tags': total_unique_tags,
            'photos_with_tags': int((final_df['tags_clean'] != "").sum()) if 'tags_clean' in final_df.columns else 0,
        },
    }
    
    return report


def _save_report(report: Dict, path: str, logger: logging.Logger) -> None:
    """Sauvegarde le rapport en JSON."""
    try:
        # Convertir les valeurs numpy en types Python natifs
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        report_native = convert_to_native(report)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_native, f, indent=2, ensure_ascii=False)
        logger.info(f"Rapport sauvegardé: {path}")
    except Exception as e:
        logger.error(f"Erreur sauvegarde rapport: {e}")


# ===============================
# API Principale
# ===============================

def clean_dataframe(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fonction principale de nettoyage.
    
    Args:
        df: DataFrame Flickr brut à nettoyer
        config: Configuration du nettoyage (None = config par défaut)
    
    Returns:
        df_clean: DataFrame nettoyé prêt pour clustering/text-mining
        report: Dictionnaire avec statistiques de nettoyage
    
    Raises:
        ValueError: Si colonnes critiques manquantes
    """
    if config is None:
        config = make_default_config()
    
    logger = setup_logging(config.log_level)
    
    logger.info("="*80)
    logger.info("DÉBUT DU NETTOYAGE — Data Mining Lyon Flickr Dataset")
    logger.info("="*80)
    
    initial_rows = len(df)
    logger.info(f"Dataset initial: {initial_rows:,} lignes × {len(df.columns)} colonnes")
    
    # A) Normalisation schéma
    df = _normalize_schema(df, logger)
    
    # B) Nettoyage géographique
    df, geo_stats = _clean_geographic(df, config, logger)
    
    # C) Nettoyage temporel
    df = _parse_dates(df, logger)
    df = _create_event_date(df, logger)
    df, date_stats = _clean_dates(df, config, logger)
    df = _extract_temporal_features(df, logger)
    
    # D) Nettoyage textuel
    df = _clean_text(df, logger)
    
    # E) Suppression doublons
    df, dup_stats = _remove_duplicates(df, config, logger)
    
    # Sampling (optionnel, pour tests)
    df, sample_stats = _apply_sampling(df, config, logger)
    
    # Validation post-nettoyage
    validation = _validate_cleaned_data(df, config, logger)
    
    # Génération rapport
    report = _generate_cleaning_report(
        initial_rows, df, geo_stats, date_stats, dup_stats,
        sample_stats, validation, config, logger
    )
    
    final_rows = len(df)
    logger.info("="*80)
    logger.info(f"NETTOYAGE TERMINÉ: {initial_rows:,} → {final_rows:,} lignes "
                f"({report['rows']['retention_rate']:.1f}% rétention)")
    logger.info("="*80)
    
    return df, report


# ===============================
# Main (exécution script)
# ===============================

def main():
    """
    Point d'entrée si exécuté en script.
    Usage: python cleaning.py
    """
    import sys
    from load_data import load_data
    
    # Configuration
    config = make_default_config()
    
    # Arguments ligne de commande (simple)
    if '--sample' in sys.argv:
        idx = sys.argv.index('--sample')
        if idx + 1 < len(sys.argv):
            config.sample_n = int(sys.argv[idx + 1])
    
    # Paths (utiliser chemins absolus depuis le script)
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    input_path = str(data_dir / "flickr_data2.csv")
    config.output_csv_path = str(data_dir / "flickr_data_clean.csv")
    config.output_parquet_path = str(data_dir / "flickr_data_clean.parquet")
    config.output_report_path = str(data_dir / "cleaning_report.json")
    
    logger = setup_logging(config.log_level)
    
    try:
        # 1. Charger données
        logger.info(f"Chargement du dataset: {input_path}")
        df_raw, load_report = load_data(input_path)
        
        # 2. Nettoyer
        df_clean, cleaning_report = clean_dataframe(df_raw, config)
        
        # 3. Sauvegarder
        if config.output_csv_path:
            logger.info(f"Sauvegarde CSV: {config.output_csv_path}")
            df_clean.to_csv(config.output_csv_path, index=False)
        
        if config.output_parquet_path:
            try:
                logger.info(f"Sauvegarde Parquet: {config.output_parquet_path}")
                df_clean.to_parquet(config.output_parquet_path, index=False)
            except ImportError as e:
                logger.warning(f"Parquet non disponible (pyarrow/fastparquet manquant), ignoré: {e}")
        
        if config.output_report_path:
            _save_report(cleaning_report, config.output_report_path, logger)
        
        # 4. Afficher résumé
        logger.info("\n" + "="*80)
        logger.info("RÉSUMÉ")
        logger.info("="*80)
        logger.info(f"Lignes finales: {len(df_clean):,}")
        logger.info(f"Colonnes finales: {len(df_clean.columns)}")
        logger.info(f"Taux rétention: {cleaning_report['rows']['retention_rate']:.2f}%")
        logger.info(f"Validation: {'✓ PASSED' if cleaning_report['validation']['passed'] else '❌ FAILED'}")
        
        logger.info("\nColonnes disponibles:")
        logger.info(f"  {', '.join(df_clean.columns)}")
        
        logger.info("\nTop 5 tags:")
        for i, tag_info in enumerate(cleaning_report['top_tags'][:5], 1):
            logger.info(f"  {i}. {tag_info['tag']}: {tag_info['count']:,} occurrences")
        
        logger.info("\n✓ Nettoyage terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ===============================
# Fonctions de filtrage avancé (OPTIONNELLES - pour notebooks)
# ===============================

def filter_by_text_quality(
    df: pd.DataFrame,
    min_words: int = 3,
    require_tags: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    NIVEAU 1 - Filtrage sémantique optionnel.
    
    Garde uniquement les photos avec information textuelle exploitable.
    
    Args:
        df: DataFrame à filtrer
        min_words: Minimum de mots dans text_merged
        require_tags: Si True, nécessite des tags non-vides
        logger: Logger optionnel
    
    Returns:
        DataFrame filtré
    
    Usage: Appeler APRÈS clustering spatial, pour décrire les zones.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    initial = len(df)
    df = df.copy()
    
    # Filtrer text_merged vide ou trop court
    if 'text_merged' in df.columns:
        word_counts = df['text_merged'].str.split().str.len().fillna(0)
        df = df[word_counts >= min_words].copy()
        logger.info(f"Filtrage texte: {initial - len(df):,} photos avec < {min_words} mots supprimées")
    
    # Filtrer tags vides si demandé
    if require_tags and 'tags_clean' in df.columns:
        before = len(df)
        df = df[df['tags_clean'] != ""].copy()
        logger.info(f"Filtrage tags: {before - len(df):,} photos sans tags supprimées")
    
    logger.info(f"→ Rétention: {len(df):,}/{initial:,} ({len(df)/initial*100:.1f}%)")
    return df


def filter_stop_tags(
    df: pd.DataFrame,
    stop_tags: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    NIVEAU 1 - Enlève tags ultra-génériques des tags_clean.
    
    Ne modifie PAS la colonne originale 'tags', seulement 'tags_clean'.
    
    Args:
        df: DataFrame
        stop_tags: Liste de tags à supprimer (None = liste par défaut)
        logger: Logger optionnel
    
    Returns:
        DataFrame avec tags_clean filtrés
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if 'tags_clean' not in df.columns:
        return df
    
    # Stop-tags par défaut (ultra-génériques pour Lyon Flickr)
    if stop_tags is None:
        stop_tags = [
            'lyon', 'france', 'photo', 'photos', 'flickr', 'instagram',
            'camera', 'nikon', 'canon', 'iphone', 'android',
            'city', 'ville', 'urban', 'urbain',
        ]
    
    logger.info(f"Filtrage {len(stop_tags)} stop-tags: {stop_tags[:5]}...")
    
    df = df.copy()
    
    def remove_stop_tags(tags_str):
        if not tags_str or pd.isna(tags_str):
            return ""
        words = tags_str.split()
        filtered = [w for w in words if w not in stop_tags]
        return ' '.join(filtered)
    
    df['tags_clean'] = df['tags_clean'].apply(remove_stop_tags)
    
    # Mettre à jour text_merged aussi
    if 'text_merged' in df.columns and 'title' in df.columns:
        # Reconstruire text_merged avec tags filtrés
        df['text_merged'] = (
            df['tags_clean'].fillna('') + ' ' +
            df['title'].fillna('')
        )
        if 'description' in df.columns:
            df['text_merged'] = df['text_merged'] + ' ' + df['description'].fillna('')
        df['text_merged'] = df['text_merged'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    logger.info("→ Stop-tags supprimés de tags_clean et text_merged")
    return df


def filter_by_user_density(
    df: pd.DataFrame,
    max_photos_per_user: int = 1000,
    strategy: str = 'limit',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    NIVEAU 2 - Gère utilisateurs hyper-actifs (bias).
    
    Args:
        df: DataFrame
        max_photos_per_user: Seuil par utilisateur
        strategy: 'limit' (garder N premières) ou 'sample' (échantillon aléatoire)
        logger: Logger optionnel
    
    Returns:
        DataFrame avec utilisateurs rééquilibrés
    
    Note: À utiliser avec précaution, documente bien le choix.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if 'user_id' not in df.columns:
        logger.warning("Colonne user_id absente, skip")
        return df
    
    initial = len(df)
    
    # Compter photos par user
    user_counts = df['user_id'].value_counts()
    heavy_users = user_counts[user_counts > max_photos_per_user]
    
    if len(heavy_users) == 0:
        logger.info(f"Pas d'utilisateurs avec > {max_photos_per_user} photos")
        return df
    
    logger.info(f"Détecté {len(heavy_users)} utilisateurs avec > {max_photos_per_user} photos")
    logger.info(f"  Top 3: {heavy_users.head(3).to_dict()}")
    
    df = df.copy()
    
    # Séparer users normaux et heavy
    normal_mask = ~df['user_id'].isin(heavy_users.index)
    df_normal = df[normal_mask].copy()
    df_heavy = df[~normal_mask].copy()
    
    # Limiter heavy users
    if strategy == 'limit':
        df_heavy = df_heavy.groupby('user_id').head(max_photos_per_user)
    elif strategy == 'sample':
        df_heavy = df_heavy.groupby('user_id').sample(
            n=max_photos_per_user, 
            replace=False,
            random_state=42
        )
    
    # Recombiner
    df_result = pd.concat([df_normal, df_heavy], ignore_index=True)
    
    logger.info(f"→ {initial - len(df_result):,} photos supprimées ({strategy})")
    logger.info(f"→ Rétention: {len(df_result):,}/{initial:,}")
    
    return df_result


def add_spatial_density_flag(
    df: pd.DataFrame,
    eps_km: float = 0.5,
    min_samples: int = 3,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    NIVEAU 2 - Ajoute flag 'is_dense' basé sur mini-DBSCAN.
    
    Identifie les photos isolées vs en cluster.
    Ne supprime PAS, juste flagge pour filtrage optionnel ultérieur.
    
    Args:
        df: DataFrame avec lat/lon
        eps_km: Rayon en km pour densité
        min_samples: Points minimum pour être "dense"
        logger: Logger optionnel
    
    Returns:
        DataFrame avec colonne 'is_dense' ajoutée
    
    Requires: sklearn
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.error("sklearn non installé, skip densité spatiale")
        df['is_dense'] = True
        return df
    
    if 'lat' not in df.columns or 'lon' not in df.columns:
        logger.warning("lat/lon absentes, skip")
        df['is_dense'] = True
        return df
    
    logger.info(f"Calcul densité spatiale (eps={eps_km}km, min_samples={min_samples})...")
    
    df = df.copy()
    coords = df[['lat', 'lon']].dropna().values
    
    if len(coords) == 0:
        df['is_dense'] = False
        return df
    
    # DBSCAN avec distance haversine (approximation)
    # eps en degrés: 1° ≈ 111km → eps_km/111
    eps_deg = eps_km / 111.0
    
    clustering = DBSCAN(eps=eps_deg, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(coords)
    
    # -1 = bruit (points isolés)
    df['is_dense'] = False
    df.loc[df[['lat', 'lon']].notna().all(axis=1), 'is_dense'] = (labels != -1)
    
    dense_count = df['is_dense'].sum()
    logger.info(f"→ {dense_count:,}/{len(df):,} photos dans zones denses ({dense_count/len(df)*100:.1f}%)")
    logger.info(f"→ {len(df) - dense_count:,} photos isolées (noise)")
    
    return df


# ===============================
# Export des fonctions utilitaires
# ===============================

__all__ = [
    'CleaningConfig',
    'make_default_config',
    'clean_dataframe',
    'preprocess_for_text_mining',
    'filter_by_text_quality',
    'filter_stop_tags',
    'filter_by_user_density',
    'add_spatial_density_flag',
]


if __name__ == "__main__":
    main()
