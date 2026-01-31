# 📚 Index Documentation Session 2

Navigation rapide vers tous les documents du projet.

---

## 🚀 Démarrage Rapide

### Pour Présentation Orale (5-7 min)
➡️ **[GUIDE_PRESENTATION_ORALE.md](GUIDE_PRESENTATION_ORALE.md)**
- Script complet 5-7 min
- 10 slides recommandés
- Questions pièges + réponses
- Timing détaillé

### Pour Lecture Rapide (1 page)
➡️ **[RESUME_EXECUTIF.md](RESUME_EXECUTIF.md)**
- Résultats en 1 page
- Tableau comparaison
- Recommandation
- Checklist complétée

---

## 📊 Résultats & Comparaison

### Résultats Finaux
➡️ **[RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md)**
- Tableau métriques 3 algos (DBSCAN, K-Means, HDBSCAN)
- Interprétation Silhouette / Davies-Bouldin
- Validation TF-IDF (10/10 POI corrects)
- Recommandation DBSCAN (justification complète)
- Questions pièges attendues

**Contenu** :
- DBSCAN : 49 clusters, 62% bruit
- K-Means : 50 clusters, silhouette 0.51
- HDBSCAN : 588 clusters (trop fragmenté)
- Top 10 POI identifiés avec GPS

---

## 🔬 Explications Techniques

### Text Mining (TF-IDF)
➡️ **[EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md)**
- Formule mathématique : TF × IDF
- Comparaison vs simple fréquence (exemple "lyon" vs "bellecour")
- Preprocessing (stop words FR/EN/custom)
- Implémentation Python (sklearn TfidfVectorizer)
- Validation résultats (Google Maps)
- Questions fréquentes (10 Q&A)

**Points clés** :
- TF : Fréquence dans cluster
- IDF : Inverse fréquence globale
- Bigrams : "place bellecour" (noms composés)
- Stop words : "lyon", "photo", "flickr" ignorés

### Comparaison Algorithmes
➡️ **[COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md)**
- DBSCAN détaillé (density-based, eps=50m, haversine)
- K-Means détaillé (centroid-based, K=50, projection équirectangulaire)
- HDBSCAN détaillé (hierarchical, densité variable)
- Tableau comparatif 8 critères
- Quand utiliser chaque algo (3 contextes)
- Optimisations possibles (grid search, mini-batch)

**Métriques expliquées** :
- Silhouette : [-1, 1], cohésion + séparation
- Davies-Bouldin : [0, ∞], intra/inter distances
- Calinski-Harabasz : Variance ratio

---

## 🎓 Guides Présentation

### Présentation Session 2 Complète
➡️ **[PRESENTATION_SEANCE2.md](PRESENTATION_SEANCE2.md)**
- Ce qu'on a réalisé (nettoyage, 3 algos, TF-IDF)
- Tableau récapitulatif 3 algorithmes
- Comparaison détaillée (avantages/inconvénients)
- Recommandation finale (DBSCAN)
- Phrases clés pour présentation
- Questions pièges (7 Q&A)
- Outputs générés (cartes, CSV, wordclouds)

**Utilisation** : Préparation défense orale complète

### Plan Stratégique Session 2
➡️ **[PLAN_SESSION2.md](PLAN_SESSION2.md)**
- Objectifs Session 2 (3 points)
- Timeline implémentation
- Ordre de développement (clustering → text mining)
- Dépendances à installer
- Tests de validation

**Utilisation** : Comprendre stratégie globale projet

---

## 📖 Autres Documents

### README Principal
➡️ **[../README.md](../README.md)**
- Vue d'ensemble projet
- Installation & exécution
- Structure projet (arborescence)
- Commandes principales
- FAQ (5 questions)

### Documents Session 1
- **EXPLICATION_PARAMETRES_DBSCAN.md** : Justification eps=50m, min_samples=50
- **CLEANING_METHODOLOGY.md** : Nettoyage 252k → 168k
- **BBOX_COMPARISON.md** : Bounding box Lyon vs données

---

## 🗂️ Organisation par Besoin

### "Je présente dans 1 heure"
1. [RESUME_EXECUTIF.md](RESUME_EXECUTIF.md) (1 page, 5 min lecture)
2. [GUIDE_PRESENTATION_ORALE.md](GUIDE_PRESENTATION_ORALE.md) (script 5-7 min)
3. [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) (questions pièges)

### "Je veux comprendre TF-IDF"
1. [EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md) (formule, exemples)
2. [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) (validation POI)

### "Je veux comparer les 3 algorithmes"
1. [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) (détails techniques)
2. [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) (métriques finales)
3. [PRESENTATION_SEANCE2.md](PRESENTATION_SEANCE2.md) (recommandation)

### "Le prof pose une question technique"
1. [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) (section FAQ)
2. [EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md) (section Questions Fréquentes)
3. [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) (Questions Pièges)

### "Je code / je debug"
1. [../README.md](../README.md) (installation, commandes)
2. [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) (paramètres optimaux)
3. Code source : `src/clustering.py`, `src/text_mining.py`

---

## 📁 Structure Complète Documentation

```
docs/
├── INDEX.md                        ← Ce fichier (navigation)
│
├── Session 2 (Nouveaux) ✨
│   ├── RESUME_EXECUTIF.md          ← Résumé 1 page (démarrage rapide)
│   ├── GUIDE_PRESENTATION_ORALE.md ← Script 5-7 min + slides + Q&A
│   ├── RESULTATS_COMPARAISON.md    ← Métriques finales + recommandation
│   ├── EXPLICATION_TF_IDF.md       ← Text mining détaillé (formule, validation)
│   ├── COMPARAISON_ALGORITHMES.md  ← 3 algos détaillés (DBSCAN, K-Means, HDBSCAN)
│   └── PLAN_SESSION2.md            ← Stratégie implémentation
│
└── Session 1 (Existants)
    ├── EXPLICATION_PARAMETRES_DBSCAN.md  ← Justification eps=50m, min_samples=50
    ├── CLEANING_METHODOLOGY.md           ← Nettoyage données (252k → 168k)
    └── BBOX_COMPARISON.md                ← Validation bounding box Lyon
```

---

## 🔍 Recherche par Mot-Clé

### DBSCAN
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Section 1
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Recommandation
- [EXPLICATION_PARAMETRES_DBSCAN.md](EXPLICATION_PARAMETRES_DBSCAN.md)

### K-Means
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Section 2
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Validation

### HDBSCAN
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Section 3
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Exploratoire

### TF-IDF
- [EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md) - Tout le document
- [PRESENTATION_SEANCE2.md](PRESENTATION_SEANCE2.md) - Section Text Mining

### Silhouette Score
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Métriques
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Tableau

### Davies-Bouldin
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Métriques
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Tableau

### Validation POI
- [EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md) - Section Résultats
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Top 10 POI

### Questions Pièges
- [GUIDE_PRESENTATION_ORALE.md](GUIDE_PRESENTATION_ORALE.md) - Section Q&A
- [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md) - Questions Pièges
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - FAQ

### Paramètres
- [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md) - Sections paramètres
- [EXPLICATION_PARAMETRES_DBSCAN.md](EXPLICATION_PARAMETRES_DBSCAN.md)

---

## 📊 Statistiques Documentation

**Fichiers créés Session 2** : 6  
**Pages totales** : ~100  
**Questions pièges couvertes** : 25+  
**Exemples concrets** : 50+  
**Graphiques/Tableaux** : 30+  

---

## ✅ Checklist Utilisation

### Avant Présentation
- [ ] Lire [RESUME_EXECUTIF.md](RESUME_EXECUTIF.md) (5 min)
- [ ] Lire [GUIDE_PRESENTATION_ORALE.md](GUIDE_PRESENTATION_ORALE.md) (15 min)
- [ ] Préparer 3 réponses questions pièges (10 min)
- [ ] Tester timing présentation (5-7 min)

### Pendant Préparation
- [ ] Comprendre TF-IDF : [EXPLICATION_TF_IDF.md](EXPLICATION_TF_IDF.md)
- [ ] Comprendre 3 algos : [COMPARAISON_ALGORITHMES.md](COMPARAISON_ALGORITHMES.md)
- [ ] Connaître métriques finales : [RESULTATS_COMPARAISON.md](RESULTATS_COMPARAISON.md)

### Si Question Inattendue
- [ ] Chercher mot-clé dans index ci-dessus
- [ ] Lire section FAQ du document concerné
- [ ] Si pas de réponse : "Je ne sais pas, je propose X"

---

## 🎯 Objectif Documentation

**Mission** : Vous permettre de défendre chaque choix technique devant le professeur avec :
1. ✅ Arguments quantitatifs (métriques)
2. ✅ Arguments qualitatifs (validation POI)
3. ✅ Justifications interprétables (eps=50m = 1 pâté de maisons)
4. ✅ Comparaisons objectives (3 algos)

**Style** : Explications pédagogiques avec exemples concrets Lyon (Bellecour, Fourvière).

---

**Bonne préparation ! Tout est documenté 📚**
