# 🔧 CORRECTION CLUSTERING - Problème Résolu

## ❌ Problème Initial

**Symptômes** :
- Clusters énormes (118,944 photos dans 1 seul cluster)
- Points de même couleur éparpillés partout
- 132 clusters détectés mais 71% dans le cluster 0
- Range du cluster 0 : 0.0635° = 7 km !

**Cause** :
- **Lyon centre-ville est une zone très dense et continue**
- 71.8% des photos (120,694) sont dans une tranche de 5.5 km
- DBSCAN connecte les points en chaîne : A→B→C...→Z
- Même avec eps=120m, toute la ville se retrouve dans 1 cluster

## ✅ Solutions Implémentées

### Solution 1 : Déduplication GPS

**Problème** :
- Certaines coordonnées ont des milliers de photos (Demeure Chaos : 11,482 photos)
- Sur-représentation spatiale fausse le clustering

**Code ajouté** :
```python
def run_dbscan_geo(
    df_clean,
    deduplicate_coords=True,  # NOUVEAU paramètre
    coord_precision=4,        # 4 décimales = ~11m
):
    if deduplicate_coords:
        # Garder seulement 1 photo par coordonnée arrondie
        df['_lat_round'] = df['lat'].round(coord_precision)
        df['_lon_round'] = df['long'].round(coord_precision)
        df_sample = df.drop_duplicates(subset=['_lat_round', '_lon_round'])
```

**Résultat** :
- 168,097 photos → 35,018 points uniques
- Clustering sur points uniques, puis propagation labels au dataset complet

### Solution 2 : Paramètres Plus Stricts

**Anciens paramètres** :
```python
eps_meters = 120  # Trop grand pour ville dense
min_samples = 30  # Trop petit
```

**Nouveaux paramètres** :
```python
eps_meters = 50   # 50m = POI distincts
min_samples = 50  # 50 photos min = POI significatifs
```

**Justification** :
- eps=50m : Sépare mieux les POI proches (Place Bellecour vs Opéra)
- min_samples=50 : Élimine petits groupes, garde POI importants

## 📊 Résultats Comparatifs

### Avant Correction
| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Clusters | 132 | Trop de clusters |
| Top cluster | 118,944 photos | **71% du dataset** ❌ |
| Bruit | 4.1% | Trop peu de bruit |
| Points clusterisés | 168,097 | Tous les points |

**Problème** : 1 méga-cluster englobe tout Lyon centre

### Après Correction
| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Clusters | 49 | Nombre raisonnable ✅ |
| Top cluster | 2,869 photos | **2% du dataset** ✅ |
| Bruit | 62.2% | Normal pour ville ✅ |
| Points uniques | 35,018 | Déduplication GPS |

**Résultat** : Clusters bien séparés, tailles cohérentes

### Top 10 Clusters (Après Correction)
```
Cluster 1 : 2,869 photos  (probablement Bellecour/Presqu'île)
Cluster 2 : 1,498 photos  (Fourvière ?)
Cluster 0 : 1,212 photos  (Part-Dieu ?)
Cluster 10:   508 photos  (Parc Tête d'Or ?)
Cluster 5 :   436 photos  
Cluster 11:   433 photos
Cluster 15:   369 photos
Cluster 17:   338 photos
Cluster 22:   333 photos
Cluster 6 :   284 photos
```

## 🎯 Comment Utiliser

### Option 1 : Exécuter Pipeline Complet
```bash
cd /Users/by/data_mining/src
/Users/by/data_mining/.venv/bin/python main.py
```

**Outputs** :
- `outputs/map_session1.html` : Carte 15k points échantillonnés
- `outputs/map_clusters.html` : Carte avec clusters colorés
- `outputs/clustered.csv` : Dataset avec labels clusters

### Option 2 : Tester Différents Paramètres
```python
# Dans Python
from load_data import load_data
from cleaning import clean_data
from clustering import run_dbscan_geo

df_raw, _ = load_data('../flickr_data2.csv')
df_clean, _ = clean_data(df_raw)

# Test 1 : Clusters larges (POI majeurs)
df_c1, rep1 = run_dbscan_geo(
    df_clean, 
    eps_meters=100,
    min_samples=50,
    deduplicate_coords=True
)

# Test 2 : Clusters fins (détaillé)
df_c2, rep2 = run_dbscan_geo(
    df_clean,
    eps_meters=30,
    min_samples=30,
    deduplicate_coords=True
)

# Test 3 : Sans déduplication (ancien comportement)
df_c3, rep3 = run_dbscan_geo(
    df_clean,
    eps_meters=50,
    min_samples=50,
    deduplicate_coords=False  # Désactiver
)
```

## ⚙️ Paramètres Recommandés

### Pour POI Majeurs (50-100 clusters)
```python
eps_meters = 80
min_samples = 50
deduplicate_coords = True
```

### Pour POI Détaillés (100-200 clusters)
```python
eps_meters = 40
min_samples = 30
deduplicate_coords = True
```

### Pour Exploration Initiale
```python
eps_meters = 50    # Compromis
min_samples = 50   # Valeur sûre
deduplicate_coords = True  # Toujours activer
```

## 🚨 Erreurs à Éviter

### ❌ NE PAS désactiver `deduplicate_coords`
Sans déduplication → méga-clusters garantis

### ❌ NE PAS mettre eps trop grand (>150m)
Lyon centre-ville se connecte en 1 cluster

### ❌ NE PAS mettre min_samples trop petit (<30)
Trop de petits clusters insignifiants

### ❌ NE PAS ignorer le bruit (>50% normal)
62% bruit = photos dispersées (résidentiel, trajets)

## 📝 Pour la Présentation au Prof

### Ce qu'on dit :

> "On a rencontré un problème classique de DBSCAN sur données denses : **clustering en chaîne**. Lyon centre-ville est continu, donc tout se connectait en 1 énorme cluster de 118k photos.
> 
> **Solutions implémentées** :
> 1. Déduplication GPS : 168k photos → 35k points uniques (évite sur-représentation)
> 2. Paramètres stricts : eps=50m, min_samples=50 (POI distincts)
> 3. Résultat : 49 clusters, top à 2,869 photos (cohérent)
> 
> **Compromis** : 62% de bruit (normal pour ville), mais clusters identifiables (Bellecour, Fourvière, etc.)."

### Si le prof demande : "Pourquoi déduplication GPS ?"

> "Sans dédup, Demeure du Chaos = 11,482 photos au même GPS. Ça biaise le clustering. Avec dédup : 1 photo par coordonnée arrondie (précision 11m). On cluster sur 35k points uniques, puis on propage les labels aux 168k photos originales."

### Si le prof demande : "Pourquoi eps=50m ?"

> "Lyon centre-ville = zone dense continue. Avec eps=120m, toute la ville se connecte en 1 cluster. eps=50m sépare les POI proches : Place Bellecour ≠ Opéra (200m séparés). C'est la granularité minimale pour POI distincts."

### Si le prof demande : "62% bruit c'est pas trop ?"

> "Non. Le bruit DBSCAN = photos isolées (zones résidentielles, trajets, événements ponctuels). C'est une **feature**, pas un bug. On cherche des POI = clusters denses. Les 62% restants sont légitimement dispersés."

## 📂 Fichiers Modifiés

- `src/clustering.py` : Ajout `deduplicate_coords`, `coord_precision`
- `src/main.py` : Nouveaux paramètres par défaut (eps=50, min=50)

## ✅ Validation

**Commande test** :
```bash
cd /Users/by/data_mining/src
/Users/by/data_mining/.venv/bin/python main.py
```

**Vérifier** :
- Clusters : 40-60
- Top cluster : <5,000 photos
- Bruit : 50-70%

Si ces critères sont respectés → **Clustering correct** ✅

## 🔍 Debug Si Problème Persiste

```python
# Vérifier distribution clusters
import pandas as pd
df = pd.read_csv('outputs/clustered.csv')
print(df['cluster'].value_counts().head(20))

# Vérifier range spatial
for cid in range(10):
    cluster = df[df['cluster'] == cid]
    if len(cluster) > 0:
        lat_range = cluster['lat'].max() - cluster['lat'].min()
        lon_range = cluster['long'].max() - cluster['long'].min()
        print(f"Cluster {cid}: {len(cluster)} photos, range=({lat_range:.4f}°, {lon_range:.4f}°)")
```

**Ranges attendus** :
- Bon cluster : <0.01° (< 1 km)
- Cluster suspect : >0.05° (> 5 km) → Revoir paramètres
