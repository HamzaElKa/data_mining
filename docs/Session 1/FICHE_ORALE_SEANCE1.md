# 🎤 FICHE ORALE SÉANCE 1 - Antisèche Présentation
## À avoir sous les yeux pendant les 10 minutes

---

## 📌 LES 3 CHIFFRES MAGIQUES

```
420,241 lignes brutes
    ↓ (exploration + cleaning)
168,000 lignes finales (40% rétention)
    ↓ (DBSCAN eps=120m min=30)
142 clusters + 27% bruit
```

**Phrase clé** :
> "On a transformé 420k photos brutes en 142 zones d'intérêt candidates pour Grand Lyon."

---

## 🔥 PROBLÈME #1 À CONNAÎTRE PAR CŒUR : Les Doublons

**Le chiffre choc** : 252,000 doublons (60% !)

**Exemple concret** :
- Photo ID `10000577595`
- Titre "Hotel Saint Nizier"
- Apparaît 3× exactement identique
- Découvert via : `cut -d',' -f1 file.csv | sort | uniq -d`

**Pourquoi critique ?**
> "Si on garde les doublons, la densité est artificiellement gonflée ×3. Les clusters deviennent faux."

**Notre décision** :
```python
df.drop_duplicates(subset=['id'], keep='first')
```

---

## 🔥 PROBLÈME #2 : Le Super-Photographe

**Le chiffre choc** : 1 user = 34,230 photos (8.1% du dataset)

**Détails** :
- User ID : `40936370@N00`
- Lieu : Demeure du Chaos (45.837, 4.826)
- 11,482 photos AU MÊME GPS !

**Pourquoi important ?**
> "1 personne ≠ intérêt touristique. Sans correction, Demeure Chaos devient faux POI #1."

**Notre décision** :
- Session 1 : Documenté (pas de filtrage)
- Session 2 : User balancing (max 500 photos/user)

---

## 🗺️ DÉCISION BBOX (Question Fréquente)

**Notre choix** : Bbox LARGE (métropole)
```
45.60-45.90 × 4.70-5.05  → 168k photos
```

**Alternative** : Bbox restrictif (centre)
```
45.72-45.80 × 4.79-4.90  → 130k photos (-23%)
```

**Justification** :
> "Grand Lyon demande analyse métropolitaine. Inclure aéroport, Confluence, périphérie = aligné mission client."

**Conséquence** :
- +38k photos conservées
- +42 clusters détectés

---

## 🤖 POURQUOI DBSCAN ? (Question Certaine)

**Comparaison rapide** :

| Critère | K-Means | DBSCAN ✅ |
|---------|---------|----------|
| Choisir K ? | ❌ Oui | ✅ Non (auto) |
| Gère bruit ? | ❌ Non | ✅ Oui (-1) |
| Distance géo ? | ⚠️ Euclidienne | ✅ Haversine |

**Phrase à dire** :
> "DBSCAN découvre automatiquement le nombre de POI (on ne sait pas si Lyon a 50 ou 200 POI). Il gère le bruit (photos isolées) et utilise distance haversine (géographique)."

---

## ⚙️ HYPERPARAMÈTRES (À Défendre)

**eps = 120 mètres**
- Pourquoi ? Taille moyenne POI lyonnais
- Place Bellecour = ~300m diamètre → rayon 150m
- On a testé : 50m (trop fragmenté), 250m (POI fusionnés)

**min_samples = 30**
- Pourquoi ? Minimum photos pour être "POI significatif"
- POI touristique attendu : >30 photos dans rayon 120m
- Élimine faux positifs (groupes spontanés)

**Phrase à dire** :
> "120m et 30 photos = premier tuning empirique basé sur géographie lyonnaise. On fera grid search en Session 2 avec métriques quantitatives."

---

## 📊 RÉSULTATS (À Annoncer Fièrement)

```
142 clusters trouvés
27% de bruit (45,000 photos isolées)

Top 3 clusters :
  8,500 photos → Bellecour (probablement)
  6,200 photos → Fourvière
  4,800 photos → Parc Tête d'Or
```

**Interprétation** :
> "27% bruit = NORMAL. Photos résidentielles, trajets, événements ponctuels. DBSCAN identifie ce qui n'est PAS un POI permanent."

---

## 🎨 VISUALISATION (Démo À Faire)

**2 cartes générées** :
1. `outputs/map_session1.html` → 15k points échantillonnés
2. `outputs/map_clusters.html` → 142 clusters colorés

**Pourquoi 15k sur 168k ?**
> "**SÉPARATION CLAIRE** : Clustering utilise 168k photos (complet). Carte utilise 15k (affichage). Contrainte technique navigateur, pas méthodologique."

**À montrer** :
- Ouvrir `map_clusters.html`
- Zoomer sur centre-ville → clusters denses visibles
- Cliquer popup → info photo

---

## ❓ QUESTIONS PIÈGES - RÉPONSES FLASH

### "Vous avez perdu 60% des données !"
> "On a éliminé 252k DOUBLONS, pas observations uniques. Qualité > quantité. 168k photos fiables > 420k avec biais."

### "Comment vous savez que eps=120m est optimal ?"
> "Empirique basé sur POI lyonnais. Grid search en Session 2 avec silhouette score pour optimiser."

### "27% bruit c'est pas trop ?"
> "Normal pour DBSCAN. Photos isolées (trajets, zones résidentielles) ne SONT PAS des POI. C'est une feature, pas un bug."

### "Pourquoi pas K-Means d'abord ?"
> "K-Means nécessite K fixé. On ne sait pas combien de POI Lyon a. DBSCAN découvre automatiquement."

### "Pourquoi bbox large ?"
> "Grand Lyon = métropole. Bbox restrictif perd 38k photos dont Demeure Chaos (11k photos). Mission client = vision complète."

### "Votre exploration a influencé quoi ?"
> "TOUT. Doublons → architecture cleaning. Super-user → roadmap Session 2. Distribution spatiale → choix bbox. 168k points → sample 15k pour perf. Exploration = fondement méthodologique."

---

## 🕐 TIMING PRÉSENTATION (10 MIN)

**0:00-1:00** | Contexte
- Grand Lyon, 420k photos Flickr, 3 objectifs

**1:00-2:30** | Exploration (Focus doublons + super-user)
- 252k doublons (60%) → Exemple "Hotel Saint Nizier"
- 1 user = 34k photos → Demeure Chaos

**2:30-4:30** | Cleaning
- Pipeline : doublons → bbox → dates → tags
- 420k → 168k (justifié)
- Bbox large vs restrictif (+38k photos)

**4:30-5:30** | Visualisation
- **DÉMO** : Ouvrir `map_clusters.html`
- Folium + MarkerCluster
- 15k échantillonnés (perf) vs 168k analysés (clustering)

**5:30-7:30** | Clustering
- Pourquoi DBSCAN (vs K-Means)
- eps=120m, min=30 (justification)
- 142 clusters, 27% bruit
- Top 3 : Bellecour, Fourvière, Parc

**7:30-9:00** | Insights + Conséquences
- Distribution hétérogène (centre dense)
- POI identifiables visuellement
- Influence exploration sur toutes décisions

**9:00-10:00** | Conclusion + Pistes Session 2
- Objectifs atteints ✅
- Amélioration : métriques (silhouette), comparaison (K-Means), validation (Wikidata)

---

## 💬 PHRASES À PLACER ABSOLUMENT

1. **Sur les doublons** :
> "60% de doublons : On ne l'aurait JAMAIS vu sans analyse systématique. C'est la valeur d'une exploration rigoureuse."

2. **Sur DBSCAN** :
> "DBSCAN est le choix naturel pour Session 1 : pas besoin de K, gère le bruit, distance géographique."

3. **Sur l'échantillonnage** :
> "Clustering utilise 168k photos. Les 15k c'est uniquement l'affichage HTML. Analyse complète, visualisation partielle."

4. **Sur bbox** :
> "Bbox élargi = vision métropolitaine alignée avec mission Grand Lyon. On ne peut pas ignorer 38k photos arbitrairement."

5. **Sur influence exploration** :
> "L'exploration n'est pas cosmétique. Chaque problème détecté = décision architecturale. Exploration = fondement méthodologique."

6. **Conclusion** :
> "On a construit un pipeline robuste et documenté qui transforme 420k photos brutes en 142 zones d'intérêt candidates. Chaque choix est justifié, traçable, reproductible."

---

## 🎯 SI TEMPS LIMITÉ (5 MIN AU LIEU DE 10)

**VERSION ULTRA-CONDENSÉE** :

1. **Contexte** (30s) : Grand Lyon, 420k photos Flickr, détecter POI
2. **Problème clé** (1min) : 252k doublons (60%) → Suppression obligatoire
3. **Cleaning** (1min) : 420k → 168k, bbox large justifié
4. **Clustering** (1min30) : DBSCAN (auto K, gère bruit), 142 clusters
5. **Démo** (30s) : Montrer `map_clusters.html`
6. **Conclusion** (30s) : Pipeline robuste, résultats validés visuellement

---

## 📁 FICHIERS À AVOIR OUVERTS

✅ `outputs/map_clusters.html` (prêt à montrer)  
✅ `src/main.py` (montrer pipeline si demandé)  
✅ `outputs/clustered.csv` (prouver 168k lignes)  
✅ Terminal avec : `wc -l outputs/clustered.csv`  

---

## 🛡️ DÉFENSE ULTIME (Si Prof Vraiment Sceptique)

**Montrer terminal** :
```bash
# Prouver doublons
cut -d',' -f1 flickr_data2.csv | sort | uniq -d | wc -l
# → ~120,000 IDs dupliqués

# Prouver super-user
awk -F',' '{print $2}' flickr_data2.csv | sort | uniq -c | sort -rn | head -1
# → 34230 40936370@N00

# Prouver clustering complet
wc -l outputs/clustered.csv
# → 168001 (168k + header)
```

**Message** :
> "Tout est vérifiable. On ne cache rien, on assume nos choix avec preuves."

---

**BON COURAGE ! VOUS ÊTES BLINDÉ 💪**
