# 🎯 CHIFFRES CLÉS ANALYSE DONNÉES - À CONNAÎTRE PAR CŒUR

## 📊 LES 10 CHIFFRES MAGIQUES

### 1. Volume Global
```
420,241 lignes brutes
168,000 photo_id uniques  
252,241 doublons (60%)
```
**Phrase** : "On a 420k lignes mais seulement 168k photos uniques. 60% sont des doublons."

---

### 2. Distribution Users
```
5,158 utilisateurs
81.5 photos/user (moyenne)
~8 photos/user (médiane)
```
**Phrase** : "5,158 photographes avec moyenne 81 photos, mais médiane 8 → distribution très inégale."

---

### 3. Top Users (CRITIQUE)
```
User #1: 34,230 photos (8.1%)
Top 10: 90,762 photos (21.6%)
Top 1%: 125,000 photos (29.7%)
```
**Phrase** : "1 seul user = 8% du dataset. Les 10 plus actifs = 21.6%. C'est une concentration énorme."

---

### 4. Hotspots GPS
```
45.837, 4.826 → 11,482 photos (Demeure Chaos)
45.765, 4.849 → 3,525 photos (Part-Dieu)
45.785, 4.853 → 2,219 photos (Parc Tête d'Or)
```
**Phrase** : "11,482 photos au même GPS : Demeure du Chaos, causé par le super-user."

---

### 5. Complétude Texte
```
Tags présents: 316,730 (75.4%)
Tags absents: 103,510 (24.6%)
Sans tags NI titre: 23,584 (5.6%)
```
**Phrase** : "25% sans tags, mais seulement 5.6% sans AUCUN texte. GPS valide → on garde pour Session 1."

---

### 6. Zone Géographique
```
Latitude: 45.552 - 45.950 (0.398° = ~44 km)
Longitude: 4.651 - 5.103 (0.452° = ~32 km)
Hors bbox >5.05: ~700 photos (0.2%)
```
**Phrase** : "Zone 44×32 km. On met bbox à 5.05 (inclut métropole), exclut seulement 700 photos trop lointaines."

---

### 7. Qualité GPS
```
GPS complet: 419,741 (99.9%)
GPS manquant: ~500 (0.1%)
```
**Phrase** : "Quasi-complet sur GPS (99.9%). 500 lignes sans coords → suppression obligatoire."

---

### 8. Dates Invalides
```
Dates problématiques: ~300 (0.07%)
Période: 2004-2025 (21 ans)
Pic activité: 2009-2014 (50% des photos)
```
**Phrase** : "300 dates invalides seulement. Impact négligeable. On flag, on ne supprime pas."

---

### 9. Impact Nettoyage
```
420,241 → 168,000 lignes finales
Perte: 252,241 lignes (60%)
Raison principale: Doublons ID
```
**Phrase** : "60% de perte MAIS ce sont des doublons, pas des observations uniques. Correction, pas perte."

---

### 10. Résultats Clustering
```
142 clusters détectés
27% de bruit (45,000 photos)
Top cluster: 8,500 photos (probablement Bellecour)
```
**Phrase** : "142 clusters trouvés, 27% bruit (photos isolées = normal). Top 3 = 19,500 photos (12%)."

---

## 🔥 LES 3 DÉCOUVERTES À CITER ABSOLUMENT

### Découverte #1 : Les Doublons Massifs
```
Photo ID 10000577595
Titre: "Hotel Saint Nizier"
GPS: 45.764936, 4.834033
Apparaît: 3 fois IDENTIQUE
```
**Impact** : "Sans déduplication, clustering faux (densité ×3 artificielle)."

---

### Découverte #2 : Le Super-Photographe
```
User: 40936370@N00
Photos: 34,230 (8.1% du dataset)
Lieu: Demeure du Chaos (45.837, 4.826)
Répétition GPS: 11,482 fois même coordonnée
```
**Impact** : "1 artiste ≠ POI touristique. Biais majeur si pas pondéré."

---

### Découverte #3 : Distribution Power Law
```
Top 1% users (52 users) = 30% des photos
Top 20% users (1,032) = 81% des photos
Bottom 80% users (4,126) = 19% des photos
```
**Impact** : "Pas distribution normale. Majorité = touristes occasionnels, minorité = super-actifs."

---

## 📋 TABLEAU DÉCISIONS (Version Ultra-Courte)

| Donnée | Quantité | Décision | Raison 1 Phrase |
|--------|----------|----------|----------------|
| Doublons ID | 252k (60%) | ❌ Supprimer | Erreur collecte, biais densité |
| Super-users | 90k (21%) | ✅ Garder S1 | Explorer puis décider S2 |
| GPS manquants | 500 (0.1%) | ❌ Supprimer | Clustering impossible |
| GPS hors bbox | 700 (0.2%) | ❌ Supprimer | Hors métropole Grand Lyon |
| Texte manquant | 23k (5.6%) | ✅ Garder | GPS valide → utile spatial |
| Dates invalides | 300 (0.07%) | ✅ Garder | Impact négligeable, GPS ok |

---

## 💬 PHRASES DÉFENSE ANALYSE

### Si le prof dit : "Vous avez vraiment analysé ?"

**Réponse Version Courte** :
> "Oui. Distribution users → loi puissance détectée (top 10 = 21.6%). Hotspots GPS → 11k photos même lieu identifié. Doublons → 60% quantifiés avec exemples. Texte → 25% manquant analysé impact par session. Chaque chiffre est vérifiable."

**Réponse Version Longue** (si temps) :
> "Notre analyse a été systématique en 6 phases :
> 1. **Volume global** : 420k lignes, 168k uniques → doublons détectés
> 2. **Distribution users** : 5,158 users, top 10 = 21.6% → concentration documentée
> 3. **Distribution spatiale** : Hotspots GPS quantifiés, Demeure Chaos identifié
> 4. **Complétude données** : 25% sans tags, 5.6% sans texte complet → flags créés
> 5. **Qualité temporelle** : 300 invalides (0.07%), pic 2009-2014 (50%)
> 6. **Décisions justifiées** : Chaque choix basé sur quantification impact
> 
> Résultat : 420k → 168k (doublons corrigés), 142 clusters, architecture modulaire (flags)."

---

### Si le prof dit : "Pourquoi garder photos sans texte ?"

> "23,584 photos (5.6%) sans tags ni titre MAIS GPS valide. Utile pour clustering spatial (Session 1). On crée flag `has_text` pour filtrer en Session 2 (text mining). Architecture modulaire : chaque session utilise ce qui lui sert."

---

### Si le prof dit : "Comment vous savez que c'est des doublons ?"

> "Test concret : `cut -d',' -f1 flickr_data2.csv | sort | uniq -c`. Exemple photo 10000577595 apparaît 3× avec EXACTEMENT mêmes coordonnées, user, date, tags, titre. Ce n'est pas 3 photos différentes, c'est 1 photo dupliquée 3 fois. Bug collecte API Flickr."

---

### Si le prof dit : "Vous avez perdu 60% des données !"

> "Mauvaise interprétation. On n'a pas perdu 252k OBSERVATIONS. On a corrigé 252k ERREURS DE COMPTAGE. 168k photos uniques comptées 1-3 fois chacune → 168k photos uniques. C'est une correction, pas une perte."

---

### Si le prof dit : "Et le super-photographe ?"

> "User 40936370@N00 : 34,230 photos (8.1%). Investigation GPS → 11,482 photos à 45.837,4.826 (Demeure du Chaos, musée art contemporain). Probablement Thierry Ehrmann, conservateur. 1 artiste ≠ tendance touristique Grand Lyon. Session 1 : on documente. Session 2 : on implémente user balancing (max 500 photos/user)."

---

## 🎯 STORYTELLING ANALYSE (Pour Intro Présentation)

**Version Narrative** :

> "Quand on a ouvert flickr_data2.csv, on a vu 420,241 lignes. Premier réflexe : 'Super, beaucoup de données !'
> 
> Mais on a ANALYSÉ systématiquement :
> 
> **Distribution users** : Top 10 = 21.6%. Un seul user = 8.1% ! Alerte rouge.
> 
> **Investigation** : 11,482 photos au même GPS. Google Maps → Demeure du Chaos, 20km de Lyon. Un artiste documente son musée. C'est pas un POI touristique.
> 
> **Doublons** : `uniq -d` sur les IDs → 252k doublons. Test photo 'Hotel Saint Nizier' : 3× identique. Erreur collecte.
> 
> **Texte** : 25% sans tags. Mais GPS valide → utile pour spatial. On flag au lieu de supprimer.
> 
> **Résultat** : 420k lignes brutes → 168k photos uniques fiables. On a COMPRIS nos données avant de les nettoyer."

---

## 📊 GRAPHIQUES CONCEPTUELS (À Dessiner au Tableau si Demandé)

### Graphique 1 : Distribution Users (Longue Traîne)
```
Photos
  ^
  |  █  
  |  █  
  |  █ ▆
  |  █ ▆ ▄
  |  █ ▆ ▄ ▂
  |  █ ▆ ▄ ▂ ▁▁▁▁▁▁▁▁▁▁▁▁▁→
  +----------------------------> Users
     1% 5%  20%          100%
     
Légende :
  █ = Super-photographes (1%)
  ▆ = Actifs (4%)
  ▄ = Amateurs (15%)
  ▂ = Occasionnels (80%)
```

### Graphique 2 : Évolution Temporelle
```
Photos
  ^
  |        ████████
  |      ████████████
  |    ██████████████
  |   ███████████████▆▆
  |  ▂▄▆███████████████▆▅▄▂
  | ▁▁▁███████████████▅▄▂▁▁
  +-----------------------> Années
  2004                  2025
  
  Légende :
  2004-2008 : Montée Flickr
  2009-2014 : Âge d'or (50%)
  2015-2025 : Déclin (Instagram)
```

### Graphique 3 : Distribution Spatiale
```
      Nord
       ↑
   [Demeure] 11k ← OUTLIER
       |
   [Parc] 2k
       |
   [Centre] 8k
←      |      →
Ouest  |  Est
   [Vieux] 2k  [Part-Dieu] 3.5k
       |
   [Confluence] 1k
       ↓
      Sud
```

---

## 🔢 MÉMO CALCULS RAPIDES

### Calcul Pourcentages
```
Doublons: 252k / 420k = 60%
Super-user: 34k / 420k = 8.1%
Top 10: 90k / 420k = 21.6%
Sans texte: 23k / 420k = 5.6%
```

### Calcul Bbox
```
Bbox métropole: 168k photos
Bbox restrictif: 130k photos
Différence: +38k (+29%)
```

### Calcul Concentration
```
Top 1% users = 52 users
52 / 5,158 = 1%
125k / 420k = 29.7%
```

---

**CE DOCUMENT = ANTISÈCHE PARFAITE POUR L'ORAL** 🎯

**Imprimer ou avoir sous les yeux pendant la présentation !**
