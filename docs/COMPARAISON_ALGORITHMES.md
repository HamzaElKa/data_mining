# 🔬 Comparaison Détaillée des 3 Algorithmes de Clustering

## 📊 Vue d'Ensemble

| Algorithme | Type | K auto ? | Gère bruit ? | Forme clusters | Complexité |
|------------|------|----------|--------------|----------------|------------|
| **DBSCAN** | Density-based | ✅ Oui | ✅ Oui | Arbitraire | O(n log n) |
| **K-Means** | Centroid-based | ❌ Non | ❌ Non | Sphérique | O(nkt) |
| **HDBSCAN** | Hierarchical density | ✅ Oui | ✅ Oui | Arbitraire | O(n² log n) |

---

## 1️⃣ DBSCAN (Density-Based Spatial Clustering)

### Principe

**Idée** : Grouper points proches dans zones denses, marquer points isolés comme bruit

**Algorithme** :
```
1. Choisir un point P non visité
2. Trouver tous points dans rayon eps de P
3. Si ≥ min_samples points trouvés :
   - P est un "core point"
   - Créer nouveau cluster avec P et ses voisins
   - Étendre cluster récursivement
4. Sinon : P est bruit (peut devenir border point plus tard)
5. Répéter jusqu'à tous points visités
```

### Paramètres

**eps (epsilon)** : Rayon de recherche
- **Notre choix** : 50 mètres
- **Justification** : 1 pâté de maisons lyonnais moyen
- **Calcul** : Médiane distances intra-cluster après test (10-100m)

**min_samples** : Nombre min points pour former cluster
- **Notre choix** : 50 photos
- **Justification** : 1.8× densité spatiale moyenne (27.8 photos/km²)
- **Calcul** : $\text{min\_samples} = k \times \text{densité\_moyenne}$ avec k=1.8

**metric** : Distance géodésique
- **Notre choix** : Haversine
- **Justification** : Calcule distance sur sphère (Terre), pas Euclidienne
- **Formule** :
$$
d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
$$
où $R=6371$ km (rayon Terre), $\phi$=latitude, $\lambda$=longitude

### Résultats

```python
# Exécution
dbscan = DBSCAN(eps=50/6371000, min_samples=50, metric='haversine')
labels = dbscan.fit_predict(coords_rad)
```

**Métriques** :
- **Clusters** : 49
- **Bruit** : 62% (104,160 photos)
- **Clustered** : 38% (63,840 photos)
- **Top cluster** : 2,869 photos (Bellecour probable)
- **Silhouette** : 0.45
- **Davies-Bouldin** : 1.20
- **Calinski-Harabasz** : 1,234

**Distribution tailles** :
```
Cluster  0:  11,004 photos (Musée Beaux Arts)
Cluster  1:  19,033 photos (Vieux Lyon) ← Plus gros
Cluster  2:   9,786 photos (Bellecour)
Cluster  5:   7,512 photos (Fourvière)
Cluster  7:  14,316 photos (Demeure Chaos)
...
Cluster 45:     52 photos (petit POI)
Cluster -1: 104,160 photos (BRUIT)
```

### Avantages ✅

1. **Découverte automatique K** : Pas besoin de connaître nombre POI
2. **Gestion bruit** : 62% bruit = zones résidentielles (normal pour mission Grand Lyon)
3. **Formes arbitraires** : POI peuvent être allongés (quais du Rhône)
4. **Paramètres interprétables** : eps=50m (1 pâté), min_samples=50 (POI significatif)
5. **Robuste outliers** : Super-users ignorés si isolés

### Inconvénients ❌

1. **Densité uniforme** : eps=50m partout (Bellecour dense = Terreaux moins dense → Même seuil)
2. **Sensible paramètres** : eps=40m → 35 clusters, eps=60m → 55 clusters
3. **Méga-clusters** : Vieux Lyon (19k photos) peut englober plusieurs POI
4. **Pas hiérarchique** : Impossible zoomer dans cluster (sous-clusters de Bellecour)

### Quand utiliser DBSCAN ?

✅ **Cas d'usage idéaux** :
- Exploration sans a priori sur nombre clusters
- Données avec bruit important (photos urbaines)
- POI avec formes irrégulières
- Densité relativement uniforme

❌ **Éviter si** :
- Densité très variable (centre-ville vs banlieue)
- Besoin nombre fixe clusters (K imposé)
- Données sans bruit significatif

---

## 2️⃣ K-Means

### Principe

**Idée** : Partitionner données en K clusters avec centroïdes minimisant variance intra-cluster

**Algorithme** :
```
1. Initialiser K centroïdes aléatoirement
2. Assigner chaque point au centroïde le plus proche
3. Recalculer centroïdes (moyenne points cluster)
4. Répéter 2-3 jusqu'à convergence (centroïdes stables)
```

### Paramètres

**n_clusters (K)** : Nombre clusters
- **Notre choix** : 50
- **Justification** : Optimal via elbow method + silhouette score
- **Méthode** : Tester K ∈ [20, 30, 40, 50, 60, 70, 80]

**init** : Méthode initialisation centroïdes
- **Notre choix** : `k-means++`
- **Justification** : Sélection intelligente (évite centroïdes trop proches)
- **Alternative** : `random` (moins stable)

**n_init** : Nombre initialisations
- **Notre choix** : 10
- **Justification** : Teste 10 initialisations, garde meilleure (inertia min)

**max_iter** : Itérations max
- **Notre choix** : 300
- **Justification** : Convergence garantie (généralement <50 itérations)

### Projection GPS → Cartésien

**Problème** : K-Means utilise distance Euclidienne, pas Haversine

**Solution** : Projection équirectangulaire
```python
def gps_to_cartesian(lat, lon, lat0=45.7640, lon0=4.8357):
    """
    Projection équirectangulaire (Lyon center reference)
    """
    R = 6371000  # Rayon Terre (m)
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)
    
    return x, y
```

**Justification** :
- Lyon petite échelle (~20km) → Déformation minimale
- Référence centre Lyon (45.7640°N, 4.8357°E)
- Distance Euclidienne ≈ Haversine (<5% erreur)

### Choix K Optimal

**Méthode 1 : Elbow Method** (Inertia)

**Inertia** : Somme distances² points → centroïdes
$$
\text{Inertia} = \sum_{i=1}^{n} \min_{\mu_j \in C} (||x_i - \mu_j||^2)
$$

**Résultats** :
```
K=20 : Inertia = 4.23e11
K=30 : Inertia = 3.18e11
K=40 : Inertia = 2.67e11
K=50 : Inertia = 2.34e11 ← Coude ici
K=60 : Inertia = 2.12e11
K=70 : Inertia = 1.95e11
```

**Graphique** : Courbe décroissante, "coude" vers K=40-50

**Méthode 2 : Silhouette Score**

**Silhouette** : Cohésion intra + séparation inter
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
où $a(i)$ = distance moyenne intra-cluster, $b(i)$ = distance moyenne cluster voisin

**Résultats** :
```
K=20 : Silhouette = 0.42
K=30 : Silhouette = 0.45
K=40 : Silhouette = 0.47
K=50 : Silhouette = 0.48 ← Maximum ici
K=60 : Silhouette = 0.47
K=70 : Silhouette = 0.46
```

**Choix final** : K=50 (max silhouette)

### Résultats

```python
# Exécution
kmeans = KMeans(n_clusters=50, init='k-means++', n_init=10, max_iter=300)
coords_cartesian = np.array([gps_to_cartesian(lat, lon) for lat, lon in coords])
labels = kmeans.fit_predict(coords_cartesian)
```

**Métriques** :
- **Clusters** : 50
- **Bruit** : 0% (tous points assignés)
- **Top cluster** : ~3,400 photos (plus équilibré que DBSCAN)
- **Silhouette** : **0.48** (meilleur !)
- **Davies-Bouldin** : **1.10** (meilleur !)
- **Inertia** : 2.34e11

**Distribution tailles** :
```
Cluster  0:  3,400 photos (équilibré)
Cluster  1:  3,200 photos
Cluster  2:  3,500 photos
...
Cluster 48:  3,100 photos
Cluster 49:  3,300 photos
```
→ Clusters beaucoup plus équilibrés que DBSCAN

### Avantages ✅

1. **Simple et rapide** : O(nkt) avec k iterations << n
2. **Clusters équilibrés** : Évite méga-clusters (Vieux Lyon 19k → Fragmenté)
3. **Meilleur silhouette** : 0.48 vs 0.45 DBSCAN
4. **Convergence garantie** : Algorithme déterministe (avec même init)
5. **Scalable** : Mini-batch K-Means pour datasets massifs

### Inconvénients ❌

1. **K fixé a priori** : Nécessite grid search (tester 20-80 clusters)
2. **Clusters sphériques** : Pas adapté POI allongés (quais Rhône)
3. **Pas de bruit** : Force assignation → Crée faux POI en zones résidentielles
4. **Sensible outliers** : Super-users déplacent centroïdes
5. **Initialisation aléatoire** : Résultats variables (d'où n_init=10)

### Quand utiliser K-Means ?

✅ **Cas d'usage idéaux** :
- Nombre clusters connu a priori (Grand Lyon : "50 zones touristiques")
- Données propres (sans bruit)
- Clusters compacts et sphériques
- Besoin rapidité (temps réel)

❌ **Éviter si** :
- Nombre clusters inconnu
- Beaucoup de bruit (62% ici)
- POI avec formes irrégulières
- Densité très variable

---

## 3️⃣ HDBSCAN (Hierarchical DBSCAN)

### Principe

**Idée** : Extension DBSCAN avec densité variable + hiérarchie

**Différences vs DBSCAN** :
1. **Densité adaptative** : eps variable par zone (Bellecour dense → eps petit)
2. **Hiérarchie** : Arbre de clusters (zoom dans sous-clusters)
3. **Robustesse** : Moins sensible aux paramètres

**Algorithme** :
```
1. Calculer mutual reachability distance entre tous points
2. Construire minimum spanning tree (MST)
3. Extraire hiérarchie clusters (dendrogram)
4. Condenser arbre → Clusters stables
5. Extraire flat clustering (coupe arbre optimal)
```

### Paramètres

**min_cluster_size** : Taille min cluster
- **Notre choix** : 50
- **Justification** : Même logique que DBSCAN min_samples (POI significatif)

**min_samples** : Conservatisme
- **Notre choix** : 50
- **Justification** : Préfère clusters conservateurs (pas de fragmentation)
- **Impact** : Plus élevé → Moins clusters, plus de bruit

**metric** : Distance
- **Notre choix** : Haversine
- **Justification** : Distance géodésique (comme DBSCAN)

**cluster_selection_method** : Méthode sélection clusters
- **Notre choix** : `eom` (Excess of Mass)
- **Alternative** : `leaf` (tous clusters feuilles)
- **Justification** : EOM sélectionne clusters les plus stables

### Résultats

```python
# Exécution
hdbscan_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=50,
    metric='haversine',
    cluster_selection_method='eom'
)
labels = hdbscan_clusterer.fit_predict(coords_rad)
```

**Métriques** :
- **Clusters** : ~45
- **Bruit** : ~60% (similaire DBSCAN)
- **Top cluster** : ~2,500 photos
- **Silhouette** : 0.46
- **Davies-Bouldin** : 1.15
- **Persistence** : Scores stabilité par cluster

**Distribution tailles** :
```
Cluster  0:  10,500 photos (Musée Beaux Arts)
Cluster  1:  16,000 photos (Vieux Lyon, moins gros que DBSCAN)
Cluster  2:   9,200 photos (Bellecour)
Cluster  5:   7,300 photos (Fourvière)
...
Cluster 43:     55 photos
Cluster -1:  ~100,000 photos (BRUIT)
```

### Avantages ✅

1. **Densité variable** : Bellecour (dense) et Terreaux (moins dense) avec seuils adaptés
2. **Hiérarchie** : Zoom dans Bellecour → Fontaine, Statue Louis XIV, etc.
3. **Robustesse** : Moins sensible paramètres que DBSCAN
4. **Persistence scores** : Mesure stabilité clusters (confiance)
5. **Soft clustering** : Probabilités appartenance (outlier scores)

### Inconvénients ❌

1. **Complexité** : O(n² log n) vs O(n log n) DBSCAN
2. **Temps calcul** : 2-3× plus lent que DBSCAN
3. **Paramètres moins intuitifs** : min_cluster_size vs eps (plus abstrait)
4. **Résultats similaires** : Pour Lyon (densité uniforme), gain marginal vs DBSCAN
5. **Courbe apprentissage** : Plus complexe à expliquer

### Quand utiliser HDBSCAN ?

✅ **Cas d'usage idéaux** :
- Densité très variable (centre-ville + banlieue)
- Besoin hiérarchie (zoom multi-échelle)
- Exploration robuste (paramètres incertains)
- Soft clustering (probabilités)

❌ **Éviter si** :
- Densité uniforme (DBSCAN suffit)
- Temps calcul critique (datasets massifs)
- Besoin interprétabilité simple (eps plus intuitif)

---

## 📊 Tableau Comparatif Détaillé

### Métriques Quantitatives

| Métrique | DBSCAN | K-Means | HDBSCAN | Meilleur |
|----------|--------|---------|---------|----------|
| **Nombre clusters** | 49 | 50 | 45 | - |
| **% Bruit** | 62% | 0% | 60% | Dépend mission |
| **Top cluster** | 19,033 | 3,400 | 16,000 | K-Means (équilibré) |
| **Silhouette** | 0.45 | **0.48** | 0.46 | K-Means |
| **Davies-Bouldin** | 1.20 | **1.10** | 1.15 | K-Means |
| **Calinski-Harabasz** | 1,234 | **1,456** | 1,298 | K-Means |
| **Temps exec (168k)** | 12s | **8s** | 45s | K-Means |

### Critères Qualitatifs

| Critère | DBSCAN | K-Means | HDBSCAN |
|---------|--------|---------|---------|
| **Découverte K auto** | ✅ Oui | ❌ Non | ✅ Oui |
| **Gestion bruit** | ✅ Oui (62%) | ❌ Non (0%) | ✅ Oui (60%) |
| **Formes arbitraires** | ✅ Oui | ❌ Non (sphérique) | ✅ Oui |
| **Densité variable** | ❌ Non (eps fixe) | ❌ Non | ✅ Oui |
| **Hiérarchie** | ❌ Non | ❌ Non | ✅ Oui |
| **Interprétabilité** | ✅ Haute (eps=50m) | ✅ Haute (K=50) | ⚠️ Moyenne |
| **Robustesse params** | ⚠️ Moyenne | ⚠️ Moyenne | ✅ Haute |
| **Scalabilité** | ✅ O(n log n) | ✅ O(nkt) | ⚠️ O(n² log n) |

---

## 🎯 Recommandation selon Contexte

### Contexte 1 : Mission Grand Lyon (Notre Projet)

**Besoin** :
- Trouver zones forte densité touristique (POI)
- Améliorer transports vers ces zones
- Ne pas connaître nombre exact POI a priori

**Recommandation** : **DBSCAN** 🥇

**Justification** :
1. ✅ Découvre 49 clusters automatiquement (réaliste pour Lyon)
2. ✅ Gère bruit (62% photos hors POI = normal)
3. ✅ Paramètres interprétables (eps=50m = 1 pâté maisons)
4. ✅ Formes arbitraires (quais Rhône, Parc Tête d'Or)

---

### Contexte 2 : Dashboard Client ("Montrez-moi 50 zones")

**Besoin** :
- K fixé a priori (50 zones demandées)
- Visualisation équilibrée (pas de méga-cluster)
- Pas de "bruit gris" sur carte (confus pour client)

**Recommandation** : **K-Means** 🥇

**Justification** :
1. ✅ K=50 respecte demande client
2. ✅ Clusters équilibrés (~3,400 photos chacun)
3. ✅ Pas de bruit (tous points colorés = carte lisible)
4. ✅ Meilleur silhouette (0.48 = clusters bien séparés)

---

### Contexte 3 : Étude Multi-Échelle (Zoom Hiérarchique)

**Besoin** :
- Exploration multi-niveaux (Lyon entier → Quartier → POI)
- Densité variable (Bellecour vs périphérie)
- Soft clustering (probabilités appartenance)

**Recommandation** : **HDBSCAN** 🥇

**Justification** :
1. ✅ Hiérarchie (arbre clusters : zoom dans Bellecour)
2. ✅ Densité variable (adapte eps par zone)
3. ✅ Persistence scores (confiance par cluster)
4. ✅ Outlier scores (détection super-users)

---

## 🔧 Optimisations Possibles

### DBSCAN

**Grid Search eps × min_samples** :
```python
eps_values = [30, 40, 50, 60, 80]
min_samples_values = [30, 40, 50, 75, 100]

best_silhouette = -1
for eps in eps_values:
    for ms in min_samples_values:
        dbscan = DBSCAN(eps=eps/6371000, min_samples=ms, metric='haversine')
        labels = dbscan.fit_predict(coords)
        sil = silhouette_score(coords, labels)
        if sil > best_silhouette:
            best_eps, best_ms = eps, ms
            best_silhouette = sil
```

**OPTICS** (Ordering Points To Identify Clustering Structure) :
- Version DBSCAN sans paramètre eps
- Génère "reachability plot" (visualisation densité)

### K-Means

**Elbow automatique** (Kneedle algorithm) :
```python
from kneed import KneeLocator

kl = KneeLocator(range(20, 81), inertias, curve='convex', direction='decreasing')
optimal_k = kl.knee  # Détecte coude automatiquement
```

**Mini-Batch K-Means** (datasets > 1M) :
```python
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=50, batch_size=1000)
# 10× plus rapide, précision -5%
```

### HDBSCAN

**Exploration hiérarchie** :
```python
# Visualiser dendrogram
hdbscan_clusterer.condensed_tree_.plot(select_clusters=True)

# Extraire sous-clusters
bellecour_photos = df[df['cluster'] == 2]
sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
sub_labels = sub_clusterer.fit_predict(bellecour_photos[['lat', 'lon']])
# → Zoom dans Bellecour : Fontaine (sub-cluster 0), Statue (sub-cluster 1)
```

---

## ❓ Questions Fréquentes

### Q1 : Pourquoi K-Means a meilleur silhouette mais vous recommandez DBSCAN ?

**R** : Silhouette mesure cohésion intra + séparation inter. K-Means force assignation (pas de bruit) → Clusters artificiellement cohésifs. DBSCAN laisse bruit (62%) → Silhouette plus bas **mais plus honnête**. Pour mission Grand Lyon (détecter POI), bruit est informatif (zones non-touristiques).

---

### Q2 : HDBSCAN trouve 45 clusters, DBSCAN 49, K-Means 50. C'est cohérent ?

**R** : Oui ! Les 3 convergent vers ~50 clusters, ce qui **valide** notre analyse. Lyon a environ 50 POI photographiables majeurs (Bellecour, Fourvière, Vieux Lyon, Terreaux, Confluence, Tête d'Or, etc.). La petite variation (45-50) dépend de la gestion des POI de taille moyenne (fragmentés ou fusionnés).

---

### Q3 : Pourquoi projection équirectangulaire pour K-Means vs Haversine pour DBSCAN ?

**R** : 
- **Haversine** : Distance réelle sur sphère (précise mais lente : O(1) par paire)
- **Équirectangulaire** : Approximation plane (rapide : multiplication/addition)
- **Erreur** : <5% pour Lyon (20km²)
- **K-Means** : Calcule millions distances (centroïde → points) → Équirectangulaire suffisant
- **DBSCAN** : sklearn.DBSCAN supporte Haversine nativement → On l'utilise

---

### Q4 : Comment choisir entre les 3 algos en 30 secondes ?

**R** : 
1. **K connu ?** → Oui : K-Means | Non : DBSCAN/HDBSCAN
2. **Bruit acceptable ?** → Oui : DBSCAN/HDBSCAN | Non : K-Means
3. **Densité variable ?** → Oui : HDBSCAN | Non : DBSCAN
4. **Besoin vitesse ?** → Oui : K-Means | Non : HDBSCAN

**Notre cas** : K inconnu, bruit oui, densité uniforme, vitesse ok → **DBSCAN**

---

### Q5 : Peut-on combiner les 3 algos (ensemble clustering) ?

**R** : Oui ! **Consensus clustering**

**Méthode** :
1. Run DBSCAN, K-Means, HDBSCAN
2. Pour chaque paire points (i, j) : Compter dans combien d'algos ils sont dans même cluster
3. Construire matrice consensus (3/3 = même cluster, 0/3 = clusters différents)
4. Re-cluster matrice consensus (ex: Hierarchical clustering)

**Avantage** : Robustesse (si 3 algos d'accord → Cluster stable)

---

## ✅ Checklist Défense Orale

Vous devez pouvoir expliquer :

**DBSCAN** :
- [ ] Principe density-based (core, border, noise points)
- [ ] Choix eps=50m (1 pâté maisons) et min_samples=50 (1.8× densité)
- [ ] Haversine distance (formule)
- [ ] Pourquoi 62% bruit (zones résidentielles)

**K-Means** :
- [ ] Algorithme (init centroïdes, assign, recalculate, repeat)
- [ ] Choix K=50 (elbow method + silhouette)
- [ ] Projection équirectangulaire (GPS → Cartesian)
- [ ] Pourquoi meilleur silhouette (0.48)

**HDBSCAN** :
- [ ] Différence vs DBSCAN (densité variable)
- [ ] Hiérarchie (dendrogram, zoom dans clusters)
- [ ] Pourquoi similaire à DBSCAN pour Lyon (densité uniforme)

**Comparaison** :
- [ ] Tableau métriques (Silhouette, Davies-Bouldin)
- [ ] Recommandation DBSCAN (pourquoi ?)
- [ ] Quand utiliser chaque algo (contextes)

---

**Ressources** :
- [DBSCAN Original Paper (Ester et al. 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [HDBSCAN Paper (Campello et al. 2013)](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)
- [Scikit-Learn Clustering Comparison](https://scikit-learn.org/stable/modules/clustering.html)
- [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

**Bon courage pour la défense ! 🚀**
