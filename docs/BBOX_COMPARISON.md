# Comparaison Bbox - Lyon Centre vs Lyon Métropole

## Configuration de votre collègue

**Bbox restrictif** (centre Lyon uniquement) :
```python
bbox_lat_min = 45.719722  # 45°43'11"N
bbox_lat_max = 45.796944  # 45°47'49"N
bbox_lon_min = 4.793333   # 4°47'36"E
bbox_lon_max = 4.895833   # 4°53'45"E
```

**Zones incluses** :
- ✅ Presqu'île (Bellecour, Hôtel de Ville)
- ✅ Vieux Lyon (Fourvière)
- ✅ Part-Dieu
- ✅ 3ème arrondissement
- ❌ Aéroport Saint-Exupéry
- ❌ Confluence (sud)
- ❌ Villeurbanne (est)
- ❌ Périphérie métropole

**Surface** : ~8-10 km² (approximation)
**Résultat attendu** : ~130k lignes

---

## Votre configuration actuelle

**Bbox large** (Grand Lyon métropole) :
```python
bbox_lat_min = 45.55
bbox_lat_max = 45.95
bbox_lon_min = 4.65
bbox_lon_max = 5.15
```

**Zones incluses** :
- ✅ Tout le centre (comme collègue)
- ✅ Aéroport Saint-Exupéry
- ✅ Confluence
- ✅ Villeurbanne, Caluire, Écully
- ✅ Périphérie métropole (59 communes)

**Surface** : ~420 km²
**Résultat attendu** : ~168k lignes

---

## Visualisation

```
┌─────────────────────────────────────┐
│  Votre bbox (Grand Lyon)            │
│  45.55 → 45.95 × 4.65 → 5.15        │
│                                     │
│    ┌──────────────────┐             │
│    │  Bbox collègue   │             │
│    │  (Centre Lyon)   │             │
│    │  45.72 → 45.80   │             │
│    │  4.79 → 4.90     │             │
│    └──────────────────┘             │
│                                     │
└─────────────────────────────────────┘
```

Le bbox de votre collègue représente environ **~2.4%** de votre bbox en surface !

---

## Comparaison résultats

| Métrique | Collègue (centre) | Vous (métropole) | Écart |
|----------|-------------------|------------------|-------|
| Lignes finales | ~130k | ~168k | +29% |
| Surface | ~10 km² | ~420 km² | +4100% |
| Densité | ~13k/km² | ~400/km² | 33x plus |
| Zones incluses | Centre historique | Grand Lyon complet | - |

---

## Quelle configuration choisir ?

### Option A : Aligner sur collègue (centre-ville)

**Modifier `src/cleaning.py` :**
```python
@dataclass
class CleaningConfig:
    # Bounding box Lyon CENTRE (aligné sur collègue)
    bbox_lat_min: float = 45.72
    bbox_lat_max: float = 45.80
    bbox_lon_min: float = 4.79
    bbox_lon_max: float = 4.90
```

**Avantages** :
- ✅ Comparaison directe avec collègue
- ✅ Clusters plus nets (zones denses)
- ✅ Focus POI touristiques majeurs
- ✅ Cartes plus lisibles

**Inconvénients** :
- ❌ Perd événements périphérie
- ❌ Moins représentatif métropole
- ❌ Exclut aéroport (potentiel POI)

**Résultat attendu** : ~130k lignes (comme collègue)

---

### Option B : Garder métropole (actuel)

**Garder configuration actuelle**

**Avantages** :
- ✅ Analyse complète métropole
- ✅ Inclut diversité zones
- ✅ Représentatif vie réelle
- ✅ Défendable ("Grand Lyon")

**Inconvénients** :
- ❌ Clusters peut-être moins nets
- ❌ Plus de bruit périphérique
- ❌ Comparaison difficile avec collègue

**Résultat** : ~168k lignes (actuel)

---

### Option C : Compromis (bbox moyen)

**Modifier pour bbox intermédiaire :**
```python
@dataclass
class CleaningConfig:
    # Bounding box Lyon ÉTENDU (compromis)
    bbox_lat_min: float = 45.65
    bbox_lat_max: float = 45.85
    bbox_lon_min: float = 4.75
    bbox_lon_max: float = 5.00
```

**Zones incluses** :
- ✅ Centre + proche périphérie
- ✅ Part-Dieu, Confluence
- ✅ Villeurbanne
- ❌ Aéroport (trop loin)

**Résultat attendu** : ~145-155k lignes

---

## 🎯 Recommandation

### Pour le projet académique :

**Testez les DEUX approches** et documentez :

```python
# Version 1 : Centre (comme collègue)
config_centre = CleaningConfig(
    bbox_lat_min=45.72, bbox_lat_max=45.80,
    bbox_lon_min=4.79, bbox_lon_max=4.90,
)

# Version 2 : Métropole (votre choix actuel)
config_metro = CleaningConfig(
    bbox_lat_min=45.55, bbox_lat_max=45.95,
    bbox_lon_min=4.65, bbox_lon_max=5.15,
)

# Comparer résultats clustering
clusters_centre = dbscan(df_centre)
clusters_metro = dbscan(df_metro)
```

### À l'oral, défendre :

> "Nous avons testé deux configurations de bbox :
> 
> 1. **Centre-ville restrictif** (~130k photos, 10 km²) : Focus POI touristiques majeurs, clusters plus nets
> 2. **Grand Lyon métropole** (~168k photos, 420 km²) : Analyse complète, représentativité métropolitaine
> 
> Nous avons choisi [VOTRE CHOIX] car [JUSTIFICATION]. Les résultats montrent [COMPARAISON]."

---

## Script de test rapide

```python
# Test les deux bbox
from src.cleaning import clean_dataframe, CleaningConfig

# Config collègue
config_friend = CleaningConfig(
    bbox_lat_min=45.72, bbox_lat_max=45.80,
    bbox_lon_min=4.79, bbox_lon_max=4.90,
    sample_n=50000,  # Test rapide
)

df_friend, _ = clean_dataframe(df_raw, config_friend)
print(f"Config collègue: {len(df_friend):,} lignes")

# Config actuelle
config_yours = CleaningConfig(sample_n=50000)
df_yours, _ = clean_dataframe(df_raw, config_yours)
print(f"Config actuelle: {len(df_yours):,} lignes")

print(f"Ratio: {len(df_yours)/len(df_friend):.2f}x plus de données")
```

---

## Conclusion

Vous n'avez PAS un "moins bon" cleaning, vous avez un **choix méthodologique différent** :

- **Votre collègue** : Focus centre touristique
- **Vous** : Analyse métropolitaine complète

Les deux sont défendables. L'important est de :
1. ✅ Documenter le choix
2. ✅ Justifier selon objectif
3. ✅ Comparer résultats (bonus)
4. ✅ Être cohérent dans l'analyse

**→ Gardez votre bbox actuel MAIS ajoutez la vérification chronologie (upload >= taken) pour avoir le meilleur des deux approches !**
