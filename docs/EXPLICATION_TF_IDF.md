# 📚 Explication TF-IDF - Text Pattern Mining

## 🎯 Objectif

**Problème** : Comment décrire automatiquement un cluster de photos avec quelques mots-clés ?

**Solution** : TF-IDF (Term Frequency - Inverse Document Frequency)

**Exemple concret** :
- Cluster 2 a 9,786 photos autour de Place Bellecour
- Tags : "lyon, bellecour, place, france, rhone, city, square..."
- **TF-IDF trouve** : "bellecour" (score 0.4456), "place bellecour" (score 0.2992)
- **Description générée** : "Cluster 2 : Place Bellecour"

---

## 📐 Principe Mathématique

### Formule Complète

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

Où :
- $t$ = terme (mot)
- $d$ = document (1 cluster = 1 document)
- $D$ = corpus (tous les clusters)

### Composante 1 : TF (Term Frequency)

**Mesure** : Fréquence du mot **dans le cluster**

$$
\text{TF}(t, d) = \frac{\text{Nombre occurrences de } t \text{ dans } d}{\text{Nombre total mots dans } d}
$$

**Exemple** :
- Cluster 2 : 50,000 mots total
- Mot "bellecour" apparaît 500 fois
- $\text{TF}(\text{"bellecour"}, C_2) = \frac{500}{50000} = 0.01$

**Interprétation** : Plus TF est élevé, plus le mot est **important dans ce cluster**.

### Composante 2 : IDF (Inverse Document Frequency)

**Mesure** : Rareté du mot **dans le corpus global**

$$
\text{IDF}(t, D) = \log \left( \frac{\text{Nombre total clusters}}{\text{Nombre clusters contenant } t} \right)
$$

**Exemple** :
- 49 clusters total
- Mot "bellecour" apparaît dans 3 clusters
- $\text{IDF}(\text{"bellecour"}) = \log \left( \frac{49}{3} \right) = \log(16.33) = 2.79$

**Interprétation** : Plus IDF est élevé, plus le mot est **rare globalement**.

### Combinaison : TF-IDF

$$
\text{TF-IDF}(\text{"bellecour"}, C_2) = 0.01 \times 2.79 = 0.0279
$$

**Normalisé** (par sklearn) : 0.4456 (après normalisation L2)

**Interprétation** : 
- Score élevé = Mot fréquent dans ce cluster **ET** rare ailleurs
- "bellecour" est **caractéristique** du cluster 2

---

## 🔍 Comparaison avec Simple Fréquence

### Pourquoi pas juste compter les mots ?

**Exemple : Mot "lyon"**

| Cluster | Occurrences "lyon" | Fréquence | TF-IDF |
|---------|---------------------|-----------|--------|
| 0 (Musée Beaux Arts) | 8,000 | **0.72** | 0.02 |
| 1 (Vieux Lyon) | 14,000 | **0.73** | 0.02 |
| 2 (Bellecour) | 7,000 | **0.70** | 0.02 |
| 5 (Fourvière) | 5,500 | **0.73** | 0.02 |

**Problème** :
- "lyon" apparaît partout (tous les photographes taguent "lyon")
- Fréquence élevée mais **pas discriminante**
- **IDF faible** → $\text{IDF}(\text{"lyon"}) = \log(49/49) = \log(1) = 0$
- **TF-IDF ≈ 0** → Mot ignoré

**Exemple : Mot "bellecour"**

| Cluster | Occurrences "bellecour" | Fréquence | TF-IDF |
|---------|-------------------------|-----------|--------|
| 0 | 2 | 0.0002 | 0.001 |
| 1 | 5 | 0.0003 | 0.001 |
| 2 | **500** | **0.05** | **0.4456** |
| 5 | 1 | 0.0001 | 0.0005 |

**Avantage TF-IDF** :
- "bellecour" rare globalement (3/49 clusters)
- **IDF élevé** → $\text{IDF}(\text{"bellecour"}) = \log(49/3) = 2.79$
- Fréquent dans cluster 2 → **TF élevé**
- **TF-IDF élevé** → Mot **caractéristique** du cluster 2

---

## 🛠️ Implémentation Python

### Étape 1 : Preprocessing Texte

**Objectif** : Nettoyer les tags et titles pour l'analyse

```python
def preprocess_text(df):
    """
    Prépare la colonne 'text' en combinant tags + title
    """
    # 1. Concatenation
    df['text'] = (df['tags'] + " " + df['title']).str.strip()
    
    # 2. Lowercase
    df['text'] = df['text'].str.lower()
    
    # 3. Remplacer virgules par espaces (tokenization)
    df['text'] = df['text'].str.replace(",", " ")
    
    # 4. Nettoyer espaces multiples
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    
    return df
```

**Exemple** :
- **Input** : `tags="Lyon,Bellecour,Place"`, `title="Fountain at Place Bellecour"`
- **Output** : `text="lyon bellecour place fountain at place bellecour"`

### Étape 2 : Stop Words

**Pourquoi** : Supprimer mots trop fréquents et non-informatifs

**Stop words FR** : `le, la, les, de, du, des, un, une, et, ou, mais...`  
**Stop words EN** : `the, a, an, of, in, on, at, is, are...`  
**Custom** : `lyon, rhone, photo, flickr, france`

```python
from nltk.corpus import stopwords

def get_stopwords():
    """
    Combine French + English + custom stop words
    """
    fr_stop = set(stopwords.words('french'))
    en_stop = set(stopwords.words('english'))
    custom = {'lyon', 'rhone', 'photo', 'flickr', 'france', 'auvergne'}
    
    return fr_stop | en_stop | custom
```

**Exemple** :
- **Avant** : "photo of the basilique de fourvière in lyon france"
- **Après** : "basilique fourvière" (mots informatifs uniquement)

### Étape 3 : TF-IDF avec Sklearn

**Code** :
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_cluster_descriptions(df, top_n_keywords=10):
    """
    Calcule TF-IDF par cluster et extrait top N keywords
    """
    # 1. Grouper textes par cluster
    cluster_texts = df.groupby('cluster')['text'].apply(lambda x: ' '.join(x))
    
    # 2. Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500,           # Top 500 mots globalement
        ngram_range=(1, 2),         # Unigrams + Bigrams
        stop_words=list(get_stopwords()),
        max_df=0.8,                 # Ignore mots dans >80% clusters
        min_df=2                    # Ignore mots dans <2 clusters
    )
    
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Extraire top keywords par cluster
    descriptions = {}
    for cluster_id in range(tfidf_matrix.shape[0]):
        # Scores TF-IDF pour ce cluster
        scores = tfidf_matrix[cluster_id].toarray()[0]
        
        # Top N indices
        top_indices = scores.argsort()[-top_n_keywords:][::-1]
        
        # Top N keywords avec scores
        keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        descriptions[cluster_id] = keywords
    
    return descriptions
```

**Paramètres clés** :
- `ngram_range=(1, 2)` : Unigrams ("bellecour") + Bigrams ("place bellecour")
- `max_df=0.8` : Ignore mots présents dans >80% clusters (trop communs)
- `min_df=2` : Ignore mots présents dans <2 clusters (typos, trop rares)

### Étape 4 : Génération Wordcloud

**Objectif** : Visualiser les keywords TF-IDF

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud_for_cluster(keywords, cluster_id, output_path):
    """
    Génère un wordcloud depuis les keywords TF-IDF
    """
    # 1. Créer dictionnaire {mot: score}
    word_freq = {word: score for word, score in keywords}
    
    # 2. Générer wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate_from_frequencies(word_freq)
    
    # 3. Sauvegarder
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {cluster_id} - Top Keywords (TF-IDF)')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**Résultat** : Image PNG avec mots dimensionnés selon score TF-IDF

---

## 📊 Résultats Détaillés

### Top 10 Clusters

**Cluster 0 : Musée des Beaux Arts** (11,004 photos)
```
Keywords:
1. beaux arts      (0.2731) ← TF-IDF élevé, spécifique
2. musée           (0.2536)
3. placedesterreaux (0.2471)
4. terreaux        (0.1852)
5. fontainebartholdi (0.1634)
```
**Validation** : Google Maps → Place des Terreaux + Musée des Beaux Arts ✅

---

**Cluster 1 : Vieux Lyon / Cathédrale Saint-Jean** (19,033 photos)
```
Keywords:
1. saint jean      (0.3139) ← Bigram !
2. vieuxlyon       (0.3102)
3. saint           (0.2987)
4. cathédrale      (0.2201)
5. vieux           (0.1876)
```
**Validation** : Google Maps → Vieux Lyon + Cathédrale Saint-Jean-Baptiste ✅

---

**Cluster 2 : Place Bellecour** (9,786 photos)
```
Keywords:
1. bellecour       (0.4456) ← Score max ! Très spécifique
2. place bellecour (0.2992) ← Bigram confirmant
3. place           (0.2541)
4. statue          (0.1423)
5. fontaine        (0.1201)
```
**Validation** : Google Maps → Place Bellecour (plus grande place piétonne Europe) ✅

---

**Cluster 3 : Parc de la Tête d'Or** (977 photos)
```
Keywords:
1. parc            (0.3912)
2. tedor           (0.3672) ← "tedor" = "tête d'or" (typo commune)
3. parcdelat       (0.3201)
4. zoo             (0.2134)
5. lac             (0.1876)
```
**Validation** : Google Maps → Parc de la Tête d'Or (zoo + lac) ✅

---

**Cluster 5 : Basilique de Fourvière** (7,512 photos)
```
Keywords:
1. basilique       (0.4280)
2. fourvière       (0.3304)
3. dame            (0.2876) ← Notre-Dame de Fourvière
4. esplanade       (0.1654)
5. colline         (0.1432)
```
**Validation** : Google Maps → Basilique Notre-Dame de Fourvière (colline) ✅

---

**Cluster 7 : Demeure du Chaos** (14,316 photos)
```
Keywords:
1. chaos           (0.5123) ← Score max ! Super-user
2. demeureduchaos  (0.4876)
3. thierry         (0.3214) ← Thierry Ehrmann (propriétaire)
4. organe          (0.2987) ← Musée de l'Organe
5. abode           (0.2543) ← "The Abode of Chaos"
```
**Validation** : Un seul utilisateur (34k photos) avec tags systématiques "chaos" ✅  
**Note** : C'est un musée d'art contemporain à Saint-Romain-au-Mont-d'Or (Lyon périphérie)

---

### Validation Globale

**Méthode** : 
1. Extraire GPS moyen du cluster
2. Chercher sur Google Maps
3. Comparer avec keywords TF-IDF

**Résultats** :
- ✅ 10/10 clusters top validés manuellement
- ✅ Keywords correspondent aux monuments réels
- ✅ Bigrams utiles ("place bellecour", "saint jean")

---

## 🎯 Avantages TF-IDF

### 1. Automatique
- Pas besoin de liste POI a priori
- Génère descriptions en quelques secondes

### 2. Discriminant
- Ignore mots trop fréquents ("lyon", "photo")
- Sélectionne mots **spécifiques** à chaque cluster

### 3. Interprétable
- Scores TF-IDF explicables (TF × IDF)
- Top keywords = description naturelle

### 4. Robuste
- Gère typos courantes ("tedor" = "tête d'or")
- Bigrams capturent noms composés ("place bellecour")

---

## 🔧 Améliorations Possibles

### 1. Lemmatisation

**Problème** : "basilique" vs "basiliques" (pluriel)

**Solution** : Lemmatisation (ramener à forme canonique)

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# "basiliques" → "basilique"
# "cathédrales" → "cathédrale"
```

### 2. Named Entity Recognition (NER)

**Problème** : "place bellecour" devrait être 1 entité, pas 2 mots

**Solution** : Utiliser spaCy pour détecter entités nommées

```python
import spacy
nlp = spacy.load("fr_core_news_sm")

doc = nlp("Photo de la Place Bellecour à Lyon")
for ent in doc.ents:
    if ent.label_ == "LOC":  # Location
        print(ent.text)  # "Place Bellecour", "Lyon"
```

### 3. Association Rules

**Objectif** : Trouver co-occurrences ("basilique" + "fourvière")

**Méthode** : Apriori algorithm

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Support: P(basilique + fourvière)
# Confidence: P(fourvière | basilique)
# Lift: P(basilique + fourvière) / (P(basilique) × P(fourvière))
```

### 4. GPT pour Descriptions Naturelles

**Objectif** : Générer descriptions en langage naturel

**Exemple** :
- **Input** : Keywords = ["basilique", "fourvière", "dame", "esplanade"]
- **GPT Output** : "Ce cluster représente la Basilique Notre-Dame de Fourvière, située sur la colline de Fourvière avec son esplanade panoramique."

---

## ❓ Questions Fréquentes

### Q1 : Pourquoi TF-IDF vs simple fréquence ?

**R** : Simple fréquence favorise mots trop généraux ("lyon", "photo"). TF-IDF pénalise mots fréquents globalement (IDF faible) et favorise mots spécifiques à un cluster.

**Exemple concret** :
- "lyon" → Fréquence haute partout → IDF ≈ 0 → TF-IDF ≈ 0 → Ignoré
- "bellecour" → Fréquence haute cluster 2 uniquement → IDF élevé → TF-IDF élevé → Sélectionné

---

### Q2 : Pourquoi bigrams (ngram_range=(1,2)) ?

**R** : Beaucoup de POI sont des noms composés : "Place Bellecour", "Saint Jean", "Tête d'Or". Bigrams capturent ces expressions complètes.

**Exemple** :
- Unigrams seuls : "place", "bellecour" (séparés, moins informatif)
- Bigrams : "place bellecour" (1 entité, plus informatif)

---

### Q3 : Comment choisir max_df et min_df ?

**R** :
- **max_df=0.8** : Ignore mots dans >80% clusters (trop généraux, genre "lyon")
- **min_df=2** : Ignore mots dans <2 clusters (typos, super-users)

**Intuition** :
- max_df : Supprimer bruit global ("the", "le", "of")
- min_df : Supprimer bruit local (typos : "belllecour", "lyyon")

---

### Q4 : Comment valider les descriptions générées ?

**R** : Validation manuelle + géographique

**Méthode** :
1. Extraire GPS moyen du cluster
2. Chercher sur Google Maps / Wikipedia
3. Comparer avec keywords TF-IDF

**Exemple Cluster 2** :
- GPS moyen : (45.758, 4.832)
- Google Maps : "Place Bellecour"
- Keywords TF-IDF : "bellecour", "place bellecour"
- **Match ! ✅**

---

### Q5 : Pourquoi custom stop words (lyon, flickr, photo) ?

**R** : Ces mots apparaissent dans **tous** les clusters (tous les photographes taguent "lyon", "flickr", "photo"). Ils ne sont pas discriminants.

**Preuve** :
- IDF("lyon") = log(49/49) = log(1) = 0
- TF-IDF("lyon") ≈ 0 partout
- → Ajouter en stop word pour accélérer calcul

---

### Q6 : TF-IDF vs Word2Vec / BERT ?

**R** :
- **TF-IDF** : Simple, rapide, interprétable (scores explicables)
- **Word2Vec** : Embeddings sémantiques (détecte synonymes : "basilique" ≈ "église")
- **BERT** : Contexte (différence "place" lieu vs "place" espace libre)

**Pour ce projet** :
- TF-IDF suffit (tags courts, pas de contexte complexe)
- Word2Vec/BERT seraient utiles pour des descriptions longues (reviews TripAdvisor)

---

## 📈 Impact sur le Projet

### Session 2 : Objectif Text Pattern Mining ✅

**Exigence sujet** :
> "Implement a first text pattern mining algorithm to find words describing a given cluster"

**Réalisé** :
- ✅ TF-IDF implémenté
- ✅ Preprocessing texte (stop words, tokenization)
- ✅ Top 10 keywords par cluster
- ✅ Validation manuelle (10/10 clusters corrects)
- ✅ Wordclouds générés

### Utilité pour Grand Lyon

**Mission** : Améliorer transports vers zones touristiques

**Avant TF-IDF** : 
- 49 clusters numérotés (Cluster 0, 1, 2...)
- Grand Lyon doit deviner ce que chaque cluster représente

**Après TF-IDF** :
- Cluster 2 = "Place Bellecour" (9,786 photos)
- Cluster 5 = "Basilique Fourvière" (7,512 photos)
- Grand Lyon peut prioriser transports vers POI identifiés

**Impact business** :
- Identification automatique POI majeurs
- Priorisation budgets transport (Bellecour = 9,786 photos → Bus ligne C3 à renforcer)

---

## ✅ Checklist Compréhension

Vous devez pouvoir expliquer :

- [ ] Formule TF-IDF (TF × IDF)
- [ ] Pourquoi IDF pénalise mots fréquents globalement
- [ ] Exemple concret ("lyon" vs "bellecour")
- [ ] Preprocessing (stop words FR/EN/custom)
- [ ] Pourquoi bigrams (noms composés : "place bellecour")
- [ ] Validation descriptions (GPS + Google Maps)
- [ ] Avantages TF-IDF vs simple fréquence
- [ ] Paramètres sklearn (max_df, min_df, ngram_range)

---

**Ressources** :
- [Wikipédia TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF)
- [Sklearn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK Stop Words](https://www.nltk.org/howto/corpus.html#stopwords-corpus)

**Bon courage ! 🚀**
