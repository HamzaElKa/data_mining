# 📋 Plan de Travail Session 2

## 🎯 Objectifs Session 2 (5/20 points)

D'après le sujet :
1. ✅ **Compléter le nettoyage des données** (parties manquantes)
2. 🔄 **Tester 3 algorithmes de clustering** + optimiser paramètres
3. 📝 **Implémenter text pattern mining** pour décrire clusters

---

## 📊 État des Lieux Session 1 (Acquis)

### Ce qu'on a déjà ✅

**Nettoyage des données** :
- ✅ Suppression doublons (252k → 168k photos uniques)
- ✅ Gestion nulls (tags/title remplis, GPS 0 nulls)
- ✅ Filtrage bbox Grand Lyon
- ✅ Déduplication GPS (168k → 35k coords uniques)
- ✅ Validation dates, normalisation colonnes

**Clustering** :
- ✅ DBSCAN avec haversine (eps=50m, min_samples=50)
- ✅ Déduplication coords pour éviter méga-cluster
- ✅ 49 clusters détectés, top=2,869 photos, 62% bruit

**Visualisation** :
- ✅ Carte interactive 15k points échantillonnés
- ✅ Carte clusters colorés (sans bruit)
- ✅ Export CSV clustered.csv

**Documentation** :
- ✅ Justification complète paramètres DBSCAN
- ✅ Analyse colonnes/nulls/doublons
- ✅ Explication eps/min_samples/CORE POINT

---

## 🚀 Plan Session 2

### 1️⃣ Compléter le Nettoyage (si nécessaire)

**À vérifier** :
- Dates : Actuellement on garde les nulls → Valider stratégie fallback
- Tags/Title : Normalisés ? Tokenisés ? → Préparer pour text mining
- GPS : Précision 4 décimales suffisante ? → Documenter

**Action** :
> Vérifier que le nettoyage Session 1 est complet. Ajouter preprocessing texte si nécessaire (lowercase, split tags, etc.)

---

### 2️⃣ Implémenter 3 Algorithmes de Clustering

Le sujet demande "try 3 clustering algorithms and optimize parameters"

#### Algo 1 : DBSCAN ✅ (déjà fait)
- **État** : Implémenté, optimisé (eps=50m, min_samples=50)
- **Résultats** : 49 clusters, 62% bruit
- **Avantages** : Pas de K a priori, gère bruit, formes arbitraires
- **Inconvénients** : Sensible paramètres, densité uniforme supposée

#### Algo 2 : K-Means 🔄 (à implémenter)
- **Pourquoi** : Comparaison classique, mentionné dans le sujet
- **Paramètre clé** : K (nombre de clusters)
- **Stratégie** :
  1. Convertir GPS (lat/lon) en coordonnées cartésiennes (X, Y en km)
  2. Tester K = 20, 30, 40, 50, 60, 70
  3. Calculer métriques : Elbow (inertia), Silhouette, Davies-Bouldin
  4. Choisir K optimal
- **Avantages** : Simple, rapide, clusters équilibrés
- **Inconvénients** : K fixé a priori, clusters sphériques uniquement, pas de gestion bruit

#### Algo 3 : HDBSCAN 🔄 (à implémenter)
- **Pourquoi** : Extension DBSCAN, gère densité variable (mentionné sujet)
- **Paramètres clés** : min_cluster_size, min_samples
- **Stratégie** :
  1. Utiliser haversine comme DBSCAN
  2. Tester min_cluster_size = 50, 100, 150
  3. Comparer avec DBSCAN (hierarchie vs flat)
- **Avantages** : Gère densité variable, hiérarchique, robuste
- **Inconvénients** : Plus complexe, paramètres moins intuitifs

**Alternative Algo 3** : Hierarchical Clustering
- Agglomerative avec linkage (ward, average, complete)
- Dendrogramme pour choisir nombre clusters
- Moins adapté gros datasets (O(n²))

**Décision** : HDBSCAN préférable (cohérent avec DBSCAN Session 1)

---

### 3️⃣ Comparer les 3 Algorithmes

**Métriques à calculer** :
- **Silhouette Score** : Cohésion intra-cluster vs séparation inter-cluster ([-1, 1], plus haut = mieux)
- **Davies-Bouldin Index** : Ratio distances intra/inter (plus bas = mieux)
- **Calinski-Harabasz Score** : Ratio variance inter/intra (plus haut = mieux)
- **Distribution clusters** : Tailles clusters, équilibre
- **Bruit** : % points non assignés (pour DBSCAN/HDBSCAN)

**Tableau comparatif** :
| Algo | Clusters | Bruit % | Silhouette | Davies-Bouldin | Top cluster | Temps exec |
|------|----------|---------|------------|----------------|-------------|------------|
| DBSCAN | 49 | 62% | ? | ? | 2,869 | ? |
| K-Means | ? | 0% | ? | ? | ? | ? |
| HDBSCAN | ? | ? | ? | ? | ? | ? |

**Visualisation** :
- 3 cartes côte-à-côte
- Comparaison distribution tailles clusters
- Analyse qualitative (POI détectés)

---

### 4️⃣ Implémenter Text Pattern Mining

Le sujet demande "first text pattern mining algorithm to find words describing a given cluster"

#### Preprocessing Texte

**Sources de texte** :
- `tags` : "lyon,rhone,france,bellecour" (séparés par virgules)
- `title` : "Place Bellecour au coucher de soleil"

**Étapes preprocessing** :
1. **Concaténer** tags + title → `text`
2. **Tokenize** : Split en mots
3. **Lowercase** : "Lyon" → "lyon"
4. **Stop words** : Supprimer "le", "la", "de", "is", "the", "a" (FR + EN)
5. **Fréquences personnalisées** : Supprimer "lyon", "photo", "flickr" (trop fréquents, non distinctifs)

**Librairies** :
- `nltk.corpus.stopwords` (stop words FR/EN)
- `sklearn.feature_extraction.text.TfidfVectorizer` (TF-IDF)

#### TF-IDF pour Description Clusters

**Principe** :
- **TF (Term Frequency)** : Fréquence mot dans cluster
- **IDF (Inverse Document Frequency)** : Rareté mot dans tous clusters
- **TF-IDF = TF × IDF** : Score élevé = mot fréquent dans CE cluster, rare ailleurs

**Algorithme** :
```python
# Pour chaque cluster C :
1. Extraire toutes les photos du cluster
2. Concaténer tous les textes (tags + title)
3. Calculer TF-IDF pour ce document vs corpus (tous les clusters)
4. Extraire top 5-10 mots avec TF-IDF max
5. Générer nom cluster : "Cluster X : mot1, mot2, mot3"
```

**Exemple attendu** :
```
Cluster 1 (Bellecour) : 
  Top keywords : bellecour, place, presqu'île, fontaine, statue
  → Description : "Cluster 1 : Place Bellecour"

Cluster 2 (Fourvière) :
  Top keywords : fourvière, basilique, notre-dame, colline, panorama
  → Description : "Cluster 2 : Basilique de Fourvière"
```

**Output** :
- CSV avec colonnes : cluster_id, top_keywords, description
- Visualisation WordCloud par cluster
- Intégration dans map HTML (popup avec description)

---

## 📚 Livrables Session 2

### Code
1. ✅ `src/clustering.py` : Ajouter `run_kmeans()`, `run_hdbscan()`
2. ✅ `src/text_mining.py` : Nouveau fichier avec preprocessing + TF-IDF
3. ✅ `src/comparison.py` : Nouveau fichier pour comparer algos
4. ✅ `src/main.py` : Update pipeline Session 2

### Outputs
1. ✅ `outputs/clusters_dbscan.csv`
2. ✅ `outputs/clusters_kmeans.csv`
3. ✅ `outputs/clusters_hdbscan.csv`
4. ✅ `outputs/comparison_metrics.csv`
5. ✅ `outputs/cluster_descriptions.csv` (TF-IDF keywords)
6. ✅ `outputs/wordcloud_cluster_X.png` (1 par cluster top)
7. ✅ `outputs/map_comparison.html` (3 algos)

### Documentation
1. ✅ `docs/PRESENTATION_SEANCE2.md` : Guide présentation
2. ✅ `docs/COMPARAISON_ALGOS.md` : Justifications choix
3. ✅ `docs/TEXT_MINING_EXPLICATIONS.md` : TF-IDF détaillé

---

## 🎯 Ordre d'Implémentation

### Phase 1 : Clustering (2-3h)
1. Implémenter K-Means avec recherche K optimal
2. Implémenter HDBSCAN avec tuning paramètres
3. Calculer métriques comparatives
4. Générer visualisations comparatives

### Phase 2 : Text Mining (1-2h)
1. Preprocessing texte (stop words, tokenization)
2. Implémenter TF-IDF par cluster
3. Extraire top keywords
4. Générer descriptions automatiques

### Phase 3 : Documentation (1h)
1. Créer guide présentation Session 2
2. Documenter justifications algos
3. Expliquer TF-IDF simplement

---

## 🗣️ Points Clés pour la Présentation

### Clustering
> "On a testé 3 algorithmes : DBSCAN (déjà optimisé Session 1), K-Means (K=50 optimal via elbow), et HDBSCAN (densité variable). On recommande DBSCAN car meilleur silhouette score et gère le bruit urbain."

### Text Mining
> "On utilise TF-IDF pour décrire chaque cluster. Par exemple, Cluster 1 a les keywords 'bellecour, place, fontaine' → On le nomme automatiquement 'Place Bellecour'. TF-IDF favorise les mots fréquents DANS ce cluster mais rares ailleurs."

### Comparaison
> "K-Means donne des clusters équilibrés mais force à choisir K. DBSCAN détecte le bon nombre (49) mais suppose densité uniforme. HDBSCAN combine les avantages : détection auto + densité variable."

---

## ⏱️ Timeline

- **Aujourd'hui** : Implémenter K-Means + HDBSCAN
- **Demain** : Text mining TF-IDF + comparaison algos
- **Après-demain** : Documentation + visualisations finales
- **Présentation** : 5 min Session 2 (algos + text mining + démo)

---

## ❓ Questions à Se Poser (Anticipation Prof)

### Sur les Algos
- "Pourquoi K-Means si DBSCAN déjà bien ?" → Comparaison demandée par sujet, K-Means = baseline classique
- "Comment vous avez choisi K ?" → Elbow method + silhouette score, K=50 cohérent avec 49 DBSCAN
- "HDBSCAN vs DBSCAN ?" → HDBSCAN gère densité variable (Bellecour vs Terreaux), hiérarchie

### Sur Text Mining
- "Pourquoi TF-IDF et pas juste TF ?" → TF favorise mots fréquents globalement (lyon, photo), TF-IDF favorise mots distinctifs
- "Comment vous gérez tags en plusieurs langues ?" → Stop words FR+EN, keywords multilingues conservés
- "Validation descriptions ?" → Comparaison manuelle avec Wikipédia/Google Maps pour top 10 clusters

---

Prêt à commencer l'implémentation ! 🚀
