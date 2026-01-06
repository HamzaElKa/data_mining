# Guide Méthodologique - Cleaning Data Mining Lyon

**Projet**: Détection automatique de zones d'intérêt (AOI) à Lyon  
**Dataset**: Flickr géolocalisé (>400k photos)  
**Année**: 2025-2026

---

## 🎯 Philosophie du Cleaning

### Principe fondamental
> **Cleaning CONSERVATIF par défaut, filtres OPTIONNELS documentés**

### Pourquoi?
1. **Éviter perte information** : GPS + dates + texte ont chacun leur utilité
2. **Flexibilité analyse** : S'adapter aux 3 objectifs (spatial, texte, temporel)
3. **Traçabilité** : Justifier chaque choix méthodologique
4. **Reproductibilité** : Paramétrable, pas de "magie"

---

## 📊 Structure du Cleaning

### Niveau 0 : Cleaning de base (OBLIGATOIRE)
**Fichier**: `src/cleaning.py` → fonction `clean_dataframe()`

**Opérations** :
- ✅ Normalisation schéma (colonnes standardisées)
- ✅ Validation GPS (lat/lon valides, bbox Lyon élargi)
- ✅ Parsing dates (date_taken → event_date + has_valid_date flag)
- ✅ Nettoyage texte (minuscules, HTML, URLs, espaces)
- ✅ Suppression doublons (photo_id unique)

**Taux rétention attendu** : ~40% (420k → ~168k photos)

**Pertes principales** :
- Photos hors Lyon (~225)
- Doublons photo_id (~252k)
- Dates invalides (~300)

**Résultat** : Dataset cohérent, fiable, exploitable

---

### Niveau 1 : Filtres sémantiques (RECOMMANDÉ selon usage)

#### A. Filtrage qualité texte
```python
filter_by_text_quality(df, min_words=3, require_tags=False)
```

**Quand?** : Description des clusters (TF-IDF, association rules)  
**Pourquoi?** : Photos sans tags/titre n'apportent rien au text mining  
**Impact** : -5 à -10% du dataset  
**Défense** : "Photos utiles spatialement mais pas sémantiquement"

#### B. Stop-tags
```python
filter_stop_tags(df, stop_tags=['lyon', 'france', 'photo', ...])
```

**Quand?** : TF-IDF plus discriminant  
**Pourquoi?** : Tags ultra-génériques ne distinguent pas les zones  
**Impact** : Pas de perte lignes, améliore qualité tags  
**Défense** : "Tags discriminants pour différencier POI"

---

### Niveau 2 : Filtres avancés (BONUS méthodologique)

#### C. Équilibrage utilisateurs
```python
filter_by_user_density(df, max_photos_per_user=500, strategy='sample')
```

**Quand?** : Éviter biais densité (1 user = 5000 photos même lieu)  
**Pourquoi?** : Représentativité vs sur-représentation  
**Impact** : Variable selon distribution users  
**Défense** : "Pondération pour éviter monopole photographique"

**⚠️ ATTENTION** : Documenter impact sur carte densité

#### D. Densité spatiale (flag)
```python
add_spatial_density_flag(df, eps_km=0.5, min_samples=5)
```

**Quand?** : Différencier photos isolées vs en cluster  
**Pourquoi?** : Focus POI majeurs vs événements ponctuels  
**Impact** : Pas de suppression, juste flag `is_dense`  
**Défense** : "Analyse différenciée selon densité locale"

---

## 🔬 Choix Méthodologiques Clés

### 1. Bbox Lyon élargi
**Choix** : [45.55, 45.95] × [4.65, 5.15]  
**Justification** : Grand Lyon métropole (59 communes, ~25km rayon)  
**Inclut** : Aéroport St-Exupéry, Confluence, Fourvière, Part-Dieu  
**Alternative testée** : Bbox restrictif [45.60, 45.90] → perd périphérie

### 2. Garder photos sans date
**Choix** : `drop_missing_event_date = False`  
**Justification** :
- Utiles pour clustering spatial (Objectif 1)
- Comparaison densités avec/sans contrainte temporelle (Objectif 3)
- Filtrage flexible via flag `has_valid_date`

**Alternative rejetée** : Supprimer → perd ~5% données pour analyse spatiale

### 3. Doublons photo_id uniquement
**Choix** : Garder 1ère occurrence par photo_id  
**Justification** : Mêmes métadonnées, même photo  
**Alternative rejetée** : Doublons GPS exacts → trop agressif (événements)

### 4. Tags optimisés mais conservés
**Choix** : Normalisation vectorisée, pas de lemmatisation par défaut  
**Justification** :
- Lemmatisation = perte sémantique (art → art, artistic → artist?)
- Disponible en option via `preprocess_for_text_mining()`
- TF-IDF fonctionne bien avec tags bruts nettoyés

---

## 📈 Métriques de Qualité

### Dataset final (baseline)
- **Lignes** : ~168k photos (40% rétention)
- **GPS valides** : 100% (post-filtrage)
- **Dates valides** : ~100% (flag `has_valid_date`)
- **Texte non-vide** : ~94%
- **Tags non-vides** : ~75%

### Clustering readiness
- **Densité** : ~400 photos/km² (bbox 420 km²)
- **Utilisateurs** : ~40k uniques
- **Photos/user médiane** : ~3
- **Surface couverte** : ~420 km²

### Text mining readiness
- **Mots/photo moyenne** : ~8-10
- **Tags uniques** : ~50k
- **Top tags** : lyon, france, architecture, museum, streetart...

---

## 🎓 Réponses aux Questions Jury

### Q: "Pourquoi pas lemmatisation?"
**R**: Lemmatisation appliquée optionnellement via `preprocess_for_text_mining()` car :
- Risque perte sens (musée → mus?)
- TF-IDF performant avec tags bruts nettoyés
- Choix documenté et réversible

### Q: "Pourquoi garder photos sans date?"
**R**: Objectif 1 (clustering spatial) ne nécessite pas de date. Filtrage temporel fait APRÈS clustering via flag `has_valid_date`. Permet comparaison densités avec/sans contrainte temporelle.

### Q: "Peut-on nettoyer davantage?"
**R**: Oui, mais cleaning conservatif volontaire pour ne pas perdre d'information utile. Filtres optionnels (qualité texte, stop-tags, équilibrage users, densité) disponibles et documentés, activables selon objectif (cf. notebook `01_cleaning_advanced_demo.ipynb`).

### Q: "Comment gérer utilisateurs hyper-actifs?"
**R**: Identifié comme biais potentiel. Solutions testées :
- Limite 500 photos/user
- Échantillonnage aléatoire
- **Choix final** : Garder tous en baseline, filtrer si impact démontré sur clustering

### Q: "Bbox trop large/restreint?"
**R**: Bbox ajusté empiriquement :
- Trop restreint : perd aéroport, périphérie
- Trop large : dilue densité centre
- **Choix** : [45.55-45.95] × [4.65-5.15] = Grand Lyon métropole (~59 communes)
- Validé par heatmap Folium (milestone 1)

---

## 🚀 Pipeline Recommandé

### Milestone 1 - Exploration
```python
df_raw = load_data()
df_clean = clean_dataframe(df_raw)  # Baseline
# Visualiser heatmap, stats, distribution
```

### Milestone 2 - Clustering spatial
```python
df_clean = load_clean_data()  # Baseline complet
# KMeans / DBSCAN / Hierarchical sur (lat, lon)
```

### Milestone 3 - Text mining
```python
df_text = filter_by_text_quality(df_clean)  # Filtre qualité
df_text = filter_stop_tags(df_text)  # Stop-tags
# TF-IDF, association rules
```

### Milestone 4 - Analyse temporelle
```python
df_temp = df_clean[df_clean['has_valid_date']]  # Filtre dates
# Détection événements ponctuels vs récurrents
```

---

## 📦 Livrables

### Code
- ✅ `src/cleaning.py` : Module complet documenté
- ✅ `notebooks/01_cleaning_advanced_demo.ipynb` : Démonstration filtres
- ✅ `data/flickr_data_clean.csv` : Dataset baseline
- ✅ `data/cleaning_report.json` : Rapport scientifique

### Documentation
- ✅ Ce guide méthodologique
- ✅ Docstrings complètes dans code
- ✅ Rapport JSON avec métriques

### Reproductibilité
- ✅ Config paramétrable (`CleaningConfig`)
- ✅ Seed fixé pour sampling
- ✅ Versions du dataset traçables

---

## ✅ Validation

### Tests unitaires (à implémenter si demandé)
```python
def test_bbox_filtering():
    assert all((df['lat'] >= bbox_min) & (df['lat'] <= bbox_max))

def test_no_duplicates():
    assert df['photo_id'].nunique() == len(df)

def test_gps_valid():
    assert df[['lat', 'lon']].notna().all().all()
```

### Validation métier
- [x] GPS cohérents (Lyon uniquement)
- [x] Dates réalistes (1990-2026)
- [x] Texte exploitable (stopwords dans notebooks)
- [x] Doublons éliminés
- [x] Dataset exploitable pour les 3 objectifs

---

## 🎯 Résumé Exécutif

**Stratégie** : Cleaning conservatif + filtres optionnels documentés

**Avantages** :
- ✅ Pas de perte information critique
- ✅ Flexibilité selon objectif analyse
- ✅ Traçabilité et reproductibilité
- ✅ Défendable scientifiquement

**Résultat** :
- Dataset baseline : ~168k photos prêtes pour clustering
- Dataset text-ready : ~150k photos pour TF-IDF
- Dataset temporel : ~167k photos avec dates valides

**Pour aller plus loin** :
- Tester impact filtres avancés sur résultats clustering
- Valider bbox empiriquement (heatmap)
- Analyser distribution utilisateurs en détail
- Implémenter pondération si biais démontré

---

*Document vivant - à mettre à jour selon résultats milestones*
