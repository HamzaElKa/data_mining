# 🎓 Définition Formelle DBSCAN : CORE POINT

## 📖 Définition Académique

### Qu'est-ce qu'un CORE POINT (Point Cœur) ?

**Définition formelle** :
> Un point **P** est un **CORE POINT** si et seulement si il a **au moins min_samples points** dans son **ε-voisinage** (incluant P lui-même).

**Notation mathématique** :
```
N_ε(P) = {Q ∈ D | distance(P, Q) ≤ ε}
|N_ε(P)| ≥ min_samples  →  P est CORE POINT
```

Où :
- `N_ε(P)` = Voisinage de P dans rayon ε (eps)
- `|N_ε(P)|` = Nombre de points dans ce voisinage (**incluant P**)
- `D` = Dataset complet

---

## 🔬 Explication Détaillée

### Calcul du Voisinage

**Pour chaque point P** :

1. **Définir rayon ε** : Exemple `eps = 50 mètres`

2. **Trouver tous les points Q dans rayon ε** :
   ```python
   voisins = []
   for Q in dataset:
       if distance(P, Q) <= eps:
           voisins.append(Q)
   ```

3. **P lui-même est dans son propre voisinage** :
   ```python
   # distance(P, P) = 0 ≤ eps → P est dans N_ε(P)
   # Donc len(voisins) inclut toujours P
   ```

4. **Compter** :
   ```python
   nb_voisins = len(voisins)  # Inclut P + tous les Q proches
   ```

---

### Condition CORE POINT

**Test** :
```python
if nb_voisins >= min_samples:
    P est CORE POINT ✅
else:
    P n'est PAS CORE POINT ❌
```

**Exemple concret** :

```
Photo A : Place Bellecour centre (45.7578, 4.8320)
eps = 50m
min_samples = 50

Étape 1 : Calculer voisinage
  - Photo A elle-même : distance(A, A) = 0m ≤ 50m ✅
  - Photo B : distance(A, B) = 12m ≤ 50m ✅
  - Photo C : distance(A, C) = 23m ≤ 50m ✅
  - ... (86 autres photos dans 50m)
  - Photo Z : distance(A, Z) = 78m > 50m ❌

Étape 2 : Compter
  N_50m(A) = {A, B, C, ..., Y} = 89 points TOTAL

Étape 3 : Tester condition CORE
  89 >= 50 → Photo A est CORE POINT ✅
```

---

## 🎯 Importance de la Condition CORE

### Pourquoi cette condition existe ?

**Objectif DBSCAN** : Détecter les **zones denses**

**Problème sans seuil** :
```
Sans min_samples :
  - 2 photos proches (distance 10m) → Formeraient cluster ?
  - Zone résidentielle avec 5 photos éparses → Cluster ?
  - Pas de distinction dense vs épars
```

**Avec min_samples = 50** :
```
✅ Place Bellecour : 89 photos dans 50m → DENSE → CORE → Cluster
❌ Rue résidentielle : 5 photos dans 50m → PAS DENSE → PAS CORE → Bruit
```

---

## 📊 Les 3 Catégories de Points

### 1️⃣ CORE POINT (Cœur)

**Condition** : `|N_ε(P)| >= min_samples`

**Caractéristiques** :
- ✅ Peut **initier** un nouveau cluster
- ✅ Peut **étendre** un cluster existant
- ✅ Sera **toujours dans un cluster** (jamais bruit)

**Exemple** :
```
Photo au centre de Place Bellecour :
  - 89 voisins dans 50m
  - 89 >= 50 → CORE POINT
  - Crée ou rejoint cluster "Bellecour"
```

---

### 2️⃣ BORDER POINT (Bord)

**Condition** : 
- `|N_ε(P)| < min_samples` (PAS assez dense pour être CORE)
- **ET** P est dans le voisinage d'au moins 1 CORE POINT

**Caractéristiques** :
- ⚠️ **Ne peut PAS initier** de cluster
- ⚠️ **Ne peut PAS étendre** un cluster
- ✅ Peut **rejoindre** un cluster existant (si voisin d'un CORE)
- ⚠️ Sera dans un cluster **uniquement si attaché à un CORE**

**Exemple** :
```
Photo au bord de Place Bellecour (près fontaine périphérique) :
  - 42 voisins dans 50m
  - 42 < 50 → PAS CORE POINT
  - MAIS distance à Photo_Centre (CORE) = 35m < 50m
  - → BORDER POINT
  - → Rejoint cluster "Bellecour" créé par Photo_Centre
```

**Différence critique CORE vs BORDER** :

| Aspect | CORE POINT | BORDER POINT |
|--------|------------|--------------|
| Densité locale | ≥ min_samples | < min_samples |
| Peut créer cluster ? | ✅ OUI | ❌ NON |
| Peut étendre cluster ? | ✅ OUI | ❌ NON |
| Dans un cluster ? | ✅ TOUJOURS | ⚠️ SI voisin CORE |

**Scénario** :
```
Point A : 60 voisins → CORE → Crée cluster 1
Point B : 40 voisins, voisin de A → BORDER → Rejoint cluster 1
Point C : 30 voisins, voisin de B → BORDER → Rejoint cluster 1

MAIS : Point C n'est PAS voisin de A (distance > eps)
→ C rejoint cluster 1 PAR TRANSITIVITÉ via B (qui est voisin de A)
→ C est BORDER (pas assez dense), ne peut pas étendre le cluster plus loin
```

---

### 3️⃣ NOISE (Bruit)

**Condition** :
- `|N_ε(P)| < min_samples` (PAS assez dense pour être CORE)
- **ET** P n'est dans le voisinage d'AUCUN CORE POINT

**Caractéristiques** :
- ❌ Ne fait partie d'**aucun cluster**
- ❌ Classé comme "bruit" (outlier)
- ⚠️ Peut avoir quelques voisins (mais <min_samples)
- ⚠️ Ces voisins sont aussi du bruit ou trop loin des CORES

**Exemple** :
```
Photo dans rue résidentielle Croix-Rousse :
  - 3 voisins dans 50m
  - 3 < 50 → PAS CORE POINT
  - Tous les CORE POINTS (Bellecour, Fourvière, etc.) sont à >50m
  - → NOISE
  - → cluster_id = -1
```

---

## 🔄 Processus de Classification (Ordre des Opérations)

### Étape 1 : Identifier les CORE POINTS

```python
for each point P in dataset:
    voisins = [Q where distance(P, Q) <= eps]
    if len(voisins) >= min_samples:
        mark P as CORE POINT
```

**Résultat** : Liste des CORE POINTS identifiés

---

### Étape 2 : Former les Clusters (Connexité des CORES)

```python
cluster_id = 0
for each CORE POINT P not yet assigned:
    cluster_id += 1
    assign P to cluster_id
    
    # Parcours en largeur (BFS) pour connecter CORES voisins
    queue = [P]
    while queue not empty:
        current = queue.pop()
        for each voisin V of current (distance <= eps):
            if V is CORE and not assigned:
                assign V to cluster_id
                queue.append(V)  # V peut étendre le cluster
```

**Règle** : 2 CORE POINTS voisins (distance ≤ eps) → Même cluster

---

### Étape 3 : Assigner les BORDER POINTS

```python
for each point P not yet assigned (ni CORE, ni dans cluster):
    for each CORE POINT C:
        if distance(P, C) <= eps:
            assign P to cluster of C
            mark P as BORDER POINT
            break  # P rejoint le 1er cluster trouvé
    if P still not assigned:
        mark P as NOISE
```

**Règle** : Si P est voisin d'un CORE → BORDER (rejoint cluster du CORE)

---

### Étape 4 : Classer le Reste en NOISE

```python
for each point P not yet assigned:
    mark P as NOISE (cluster_id = -1)
```

---

## 🎯 Exemples Concrets Lyon

### Exemple 1 : Place Bellecour (POI Dense)

**Configuration** : eps=50m, min_samples=50

| Point | Coords | Nb Voisins | Classification | Cluster | Raison |
|-------|--------|------------|----------------|---------|--------|
| P1 | (45.7578, 4.8320) | 89 | **CORE** | 1 | 89 ≥ 50 ✅ |
| P2 | (45.7580, 4.8322) | 76 | **CORE** | 1 | 76 ≥ 50, voisin P1 ✅ |
| P3 | (45.7575, 4.8318) | 42 | **BORDER** | 1 | 42 < 50, mais voisin P1 ⚠️ |
| P4 | (45.7570, 4.8315) | 38 | **BORDER** | 1 | 38 < 50, mais voisin P3 qui est dans cluster 1 ⚠️ |

**Résultat** : Cluster 1 = 2,869 photos (Bellecour)
- CORE POINTS : ~1,800 photos (centre place, très dense)
- BORDER POINTS : ~1,069 photos (périphérie, moins dense mais connectée)

---

### Exemple 2 : Rue Résidentielle (Zone Éparse)

**Configuration** : eps=50m, min_samples=50

| Point | Coords | Nb Voisins | Classification | Cluster | Raison |
|-------|--------|------------|----------------|---------|--------|
| P1 | (45.7812, 4.8345) | 3 | **NOISE** | -1 | 3 < 50, aucun CORE proche ❌ |
| P2 | (45.7814, 4.8347) | 2 | **NOISE** | -1 | 2 < 50, aucun CORE proche ❌ |
| P3 | (45.7816, 4.8342) | 4 | **NOISE** | -1 | 4 < 50, aucun CORE proche ❌ |

**Résultat** : 3 photos classées NOISE (pas de cluster)

---

### Exemple 3 : Transition Dense → Épars

**Configuration** : eps=50m, min_samples=50

```
Zone A (Bellecour centre) :
  P1 : 89 voisins → CORE → Cluster 1
  P2 : 76 voisins, dist(P1,P2)=25m → CORE → Cluster 1

Zone B (Bellecour périphérie) :
  P3 : 42 voisins, dist(P2,P3)=35m → BORDER → Cluster 1 (voisin P2 CORE)
  P4 : 38 voisins, dist(P3,P4)=40m → BORDER → Cluster 1 (voisin P3 dans cluster)

Zone C (Rue adjacente hors place) :
  P5 : 6 voisins, dist(P4,P5)=55m → NOISE → -1 (trop loin, pas voisin)
  P6 : 4 voisins, dist(P5,P6)=20m → NOISE → -1 (voisin P5 NOISE, pas CORE)
```

**Observation** :
- CORE POINTS → Initient cluster
- BORDER POINTS → Étendent cluster jusqu'à un certain point
- NOISE → Au-delà, la densité est trop faible

---

## 🗣️ Comment Expliquer au Prof

### Phrase Clé

> "Un point est **CORE POINT** (cœur de cluster) si et seulement si il a **au moins min_samples points dans son voisinage de rayon eps** (incluant lui-même). Seuls les CORE POINTS peuvent initier ou étendre un cluster. Un point qui n'a pas assez de voisins (< min_samples) peut soit être un BORDER POINT (s'il est voisin d'un CORE), soit du NOISE (s'il est isolé)."

---

### Réponse à "C'est quoi min_samples exactement ?"

**R** : 
> "min_samples est le **seuil de densité locale** qui définit si un point est dans une **zone dense** (POI). Un point P doit avoir **au moins min_samples points dans un rayon de eps mètres** (incluant P lui-même) pour être considéré comme **CORE POINT** = cœur d'un cluster.
>
> Dans notre cas (min_samples=50) : Une photo doit avoir au moins 50 photos dans un rayon de 50m pour être un CORE. Si elle en a 49 ou moins, elle ne peut PAS être un cœur de cluster. Elle peut éventuellement être en bordure d'un cluster existant (BORDER), ou être classée bruit (NOISE) si elle est trop isolée."

---

### Exemple Visuel pour le Prof

```
Imaginez Place Bellecour :

eps = 50m (rayon de voisinage)
min_samples = 50 (seuil de densité)

Centre de la place (zone très dense) :
  Photo A : 89 photos dans 50m → 89 ≥ 50 → CORE POINT ✅
  → A peut créer/étendre un cluster

Bord de la place (zone moins dense) :
  Photo B : 42 photos dans 50m → 42 < 50 → PAS CORE ❌
  → B n'est pas assez dense pour être cœur
  MAIS B est à 30m de A (CORE) → B devient BORDER POINT ⚠️
  → B rejoint le cluster créé par A, mais ne peut pas l'étendre

Rue adjacente (zone éparse) :
  Photo C : 3 photos dans 50m → 3 < 50 → PAS CORE ❌
  → C n'est pas assez dense
  ET C est à 60m de A (>50m) → C n'est pas voisin d'un CORE
  → C est NOISE ❌ (hors cluster)
```

---

### Réponse à "Pourquoi 50 et pas 30 ou 100 ?"

**R** :
> "min_samples=50 vient d'un calcul statistique. Densité moyenne de Lyon = 3,577 photos/km². Dans un disque de rayon 50m, on attend **28 photos aléatoirement**. 
>
> 50 photos = **1.8× la densité moyenne** → On cible les zones **sur-représentées** = POI touristiques.
>
> - Avec min_samples=30 : On aurait 67 clusters (trop de micro-POI, zones résidentielles incluses)
> - Avec min_samples=100 : On aurait 18 clusters (on perd les POI moyens comme Terreaux)
> - **min_samples=50 = sweet spot** : 49 clusters, POI majeurs ET moyens détectés"

---

## ✅ Résumé

**CORE POINT** = Point avec **≥ min_samples voisins** dans rayon eps
- Rôle : **Cœur du cluster**, peut initier/étendre
- Condition stricte : `|N_ε(P)| ≥ min_samples`
- Toujours dans un cluster ✅

**BORDER POINT** = Point avec **< min_samples voisins**, mais voisin d'un CORE
- Rôle : **Périphérie du cluster**, rejoint mais ne peut pas étendre
- Condition : Pas CORE, mais distance ≤ eps à un CORE
- Dans un cluster seulement si connecté ⚠️

**NOISE** = Point avec **< min_samples voisins** et pas voisin de CORE
- Rôle : **Hors cluster**, point isolé
- Condition : Pas CORE, pas voisin de CORE
- cluster_id = -1 ❌

**min_samples contrôle la "densité minimale pour être un cœur"** 🎯
