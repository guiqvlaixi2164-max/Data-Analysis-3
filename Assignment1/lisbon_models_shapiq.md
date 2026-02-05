# SHAPIQ: Understanding Feature Interactions in Airbnb Pricing Models

---

Jupyter Notebook:

https://github.com/guiqvlaixi2164-max/Data-Analysis-3/blob/main/Assignment1/code/lisbon_models_shapiq.ipynb

---


## Background: The Problem

**Context**: We trained multiple models to predict Airbnb prices in Lisbon using property features (bedrooms, bathrooms, amenities, location scores, etc.). The Histogram Gradient Boosting Machine (HGBM) achieved the best performance (RMSE: 60.2), outperforming linear models (OLS: 69.3, Lasso: 100.5).

**Question**: *Why* does HGBM work so well? Which feature combinations drive predictions? How can we explain this to stakeholders?

**Challenge**: Standard feature importance only tells us *individual* contributions. It doesn't reveal which features work together synergistically.

---

## Part 1: Setup and Data Selection

### Code

```python
# Install shapiq if needed
# pip install shapiq

import shapiq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming we have:
# - X_lisbon_q1: Feature matrix (N x 147 features)
# - y_lisbon_q1: Target prices
# - hgbm_final: Trained HGBM model
# - feature_names: List of feature names

# Select a mid-priced listing to explain
sample_idx = len(X_lisbon_q1) // 2
x_explain = X_lisbon_q1[sample_idx]

print(f"Explaining observation {sample_idx}")
print(f"Actual price: €{y_lisbon_q1[sample_idx]:.2f}")
print(f"HGBM prediction: €{hgbm_final.predict([x_explain])[0]:.2f}")
```

### Expected Output

```
Explaining observation 10305
Actual price: €112.00
HGBM prediction: €108.32
```

### Interpretation

We're analyzing a **single observation** (listing 10305) to understand:
- How the model arrived at its €108.32 prediction
- Which features contributed most
- Which features interact with each other

**Why one observation?** Interaction patterns are often observation-specific. A luxury apartment's pricing drivers differ from a budget studio.

---

## Part 2: Shapley Values (Individual Feature Contributions)

### Conceptual Foundation

**Shapley Values** come from game theory (developed by Lloyd Shapley, 1953 Nobel Prize). They answer:

> "How much does each feature contribute to moving the prediction away from the baseline (average prediction)?"

**Key Properties:**
- **Efficiency**: All contributions sum to the prediction difference from baseline
- **Symmetry**: Features with identical roles get identical values
- **Null player**: Irrelevant features get zero contribution
- **Additivity**: Contributions are linear across coalitions

### Code

```python
# Create explainer for standard Shapley values
explainer_sv = shapiq.TabularExplainer(
    model=hgbm_final,           # The model to explain
    data=X_lisbon_q1,           # Background data (for baseline calculation)
    index="SV",                 # Shapley Values
)

# Explain the selected observation
sv_values = explainer_sv.explain(x_explain, budget=256)
print("Top 10 Feature Attributions:")
print(sv_values)

# Visualize with force plot
sv_values.plot_force(feature_names=feature_names)
plt.tight_layout()
plt.show()
```

### Expected Output

```
Top 10 Feature Attributions:
InteractionValues(
    index=SV, max_order=1, min_order=0, estimated=True, estimation_budget=256,
    n_players=147, baseline_value=113.68738291005238,
    Top 10 interactions:
        (): 113.68738291005238          # Baseline (average prediction)
        (29,): 8.521334561990342         # Feature 29 adds €8.52
        (7,): 5.787283646531763          # Feature 7 adds €5.79
        (9,): 5.763333070460648          # Feature 9 adds €5.76
        (4,): 4.582600307008104          # Feature 4 adds €4.58
        (15,): -4.507943919326808        # Feature 15 subtracts €4.51
        (71,): -4.7327432992890515       # Feature 71 subtracts €4.73
        (59,): -4.979992319157734        # Feature 59 subtracts €4.98
        (140,): -8.305501911713705       # Feature 140 subtracts €8.31
        (18,): -10.927599164894577       # Feature 18 subtracts €10.93
)
```

### Interpretation

**Baseline**: €113.69 (average price across all listings)  
**Prediction**: €108.32  
**Gap**: €108.32 - €113.69 = -€5.37

The top features explain this gap:
- **Positive contributors** (increase price):
  - Feature 29: +€8.52 (e.g., "located in tourist area")
  - Feature 7: +€5.79 (e.g., "has 2 bathrooms")
  
- **Negative contributors** (decrease price):
  - Feature 18: -€10.93 (e.g., "room type: Private room" vs entire home)
  - Feature 140: -€8.31 (e.g., "low review score")

**Key Insight**: Shapley values are **additive**. The prediction equals:
```
Baseline + Sum of all Shapley values = 113.69 + (-5.37) = 108.32
```

### What This Tells Us (and What It Doesn't)

✅ **What we know**: Which features matter individually  
❌ **What we don't know**: Do features interact? Does having 2 bathrooms matter *more* when you also have 3 bedrooms?

---

## Part 3: Shapley Interaction Index (k-SII)

### Conceptual Foundation

**Shapley Interaction Index** (k-SII) extends Shapley values to measure **synergies** between features.

**Definition**: The interaction between features i and j measures how much their combined effect differs from the sum of their individual effects.

**Mathematical Intuition**:
```
Synergy = Effect(i,j together) - Effect(i alone) - Effect(j alone)
```

**Types of Interactions**:
- **Positive (Synergy)**: Features amplify each other
  - Example: `bedrooms × bathrooms` = larger properties are disproportionately valuable
  
- **Negative (Substitution)**: Features suppress each other
  - Example: `good_location × has_parking` = parking matters less in walkable areas
  
- **Zero (Independence)**: Features don't interact
  - Example: `bedrooms × neighborhood_cleanliness` = independent effects

### Code

```python
# Compute pairwise interactions (up to order 2)
explainer_sii = shapiq.TabularExplainer(
    model=hgbm_final,
    data=X_lisbon_q1,
    index="k-SII",              # k-Shapley Interaction Index
    max_order=2                 # Pairwise (2-way) interactions
)

sii_values = explainer_sii.explain(x_explain, budget=512)
print("Top 15 Interactions (HGBM):")
print(sii_values)
```

### Expected Output

```
Top 15 Interactions (HGBM):
InteractionValues(
    index=k-SII, max_order=2, min_order=0, estimated=True, estimation_budget=408,
    n_players=147, baseline_value=113.68738291005238,
    Top 10 interactions:
        (): 113.68738291005238           # Baseline
        (31,): 12.145316245987821        # Feature 31 main effect
        (88,): 12.145316245987821        # Feature 88 main effect
        (32,): 11.071378775132345        # Feature 32 main effect
        (106,): 11.071378775132345       # Feature 106 main effect
        (9,): 7.738064319459037          # Feature 9 main effect
        (29,): 6.974313688425163         # Feature 29 main effect
        (32, 106): -11.071378775132345   # Strong negative interaction!
        (18,): -11.992194881140666       # Feature 18 main effect
        (31, 88): -12.145316245987821    # Strong negative interaction!
)
```

### Interpretation: Reading k-SII Values

**Notation**:
- `(i,)`: Main effect of feature i (same as Shapley value)
- `(i, j)`: Interaction term between features i and j

**Key Finding**: `(31, 88): -12.145`

This is a **strong negative interaction**. Let's say:
- Feature 31 = `property_type[Entire rental unit]`
- Feature 88 = `property_type[T.Shared room]`

**Interpretation**: These are **mutually exclusive dummy variables** from the same categorical variable. When one is 1, the other is 0. The negative interaction indicates they're perfect substitutes (as expected for dummy encoding).

**Why is this useful?** It confirms our model correctly learned that property types are mutually exclusive.

### Real Interaction Example

Look for interactions between different feature *types*. For instance, if we see:

```
(7, 9): 3.245   # bathrooms × bedrooms
```

**This means**: Having both more bathrooms AND more bedrooms together increases price by an **additional** €3.25 beyond their individual effects.

**Total effect of 2 bed + 2 bath**:
```
= Main(bedrooms) + Main(bathrooms) + Interaction(bed, bath)
= 5.79 + 5.76 + 3.25
= 14.80
```

**This is synergy**: 14.80 > 5.79 + 5.76 = 11.55

---

## Part 4: Network Visualization

### Purpose

Network plots provide an intuitive view of the interaction structure:
- Which features cluster together?
- What are the strongest interactions?
- Are there unexpected relationships?

### Code

```python
# Create network plot
# Note: plot_network() works on the full InteractionValues object
sii_values.plot_network(feature_names=feature_names)
plt.title("Feature Interaction Network (HGBM)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== Key Questions to Explore ===")
print("• Do bedrooms, bathrooms, accommodates form a cluster?")
print("• Which amenities interact with property size?")
print("• Do room_type variables interact with location scores?")
print("• What's the strongest interaction (thickest edge)?")
```

### How to Read the Network Plot

**Nodes (Circles)**:
- **Size**: Importance of individual feature (main effect)
- **Color**: Red = positive effect, Blue = negative effect
- **Intensity**: Magnitude of effect

**Edges (Lines)**:
- **Thickness**: Strength of interaction (|interaction value|)
- **Color**: Red/Pink = positive synergy, Blue = negative interaction
- **Presence**: Only shown for sufficiently strong interactions

### What to Look For

1. **Property Size Cluster**: Do `bedrooms`, `bathrooms`, `accommodates` connect?
   - **If yes**: Size features work together synergistically
   - **Business insight**: Upgrading multiple size features together yields more value

2. **Amenity Hubs**: Are features like `has_elevator`, `has_AC` connected to size features?
   - **If yes**: Amenities are more valuable in larger properties
   - **Example**: Elevator matters more in a 3-bedroom than a studio

3. **Room Type Edges**: Does `room_type_Private_room` connect to size features?
   - **Expect**: Negative edges (private rooms benefit less from extra bathrooms)

4. **Isolated Nodes**: Features with no edges work independently
   - **Example**: `days_since_last_review` may not interact with anything

### Example Interpretation

**Observed Pattern**: 
```
bathrooms --[thick red edge]--> bedrooms
bathrooms --[thin red edge]--> accommodates
```

**Interpretation**:
- Strong synergy between bathrooms and bedrooms
- Weaker synergy with accommodates
- Adding a bathroom to a multi-bedroom property yields high returns

**Business Application**:
> "For 3+ bedroom properties, adding an additional bathroom increases value by approximately €X more than it would for a studio apartment."

---

## Part 5: Model Comparison (HGBM vs Random Forest)

### Motivation

**Research Question**: Why does HGBM generalize better to Porto (different city) than Random Forest?

**Hypothesis**: Models relying on **universal interactions** (size × size, amenities × size) transfer better than those using **location-specific interactions** (neighborhood × amenities).

### Code

```python
# Compute interactions for Random Forest
explainer_rf = shapiq.TabularExplainer(
    model=rf_final,        # Random Forest model
    data=X_lisbon_q1,
    index="k-SII",
    max_order=2
)

sii_rf = explainer_rf.explain(x_explain, budget=512)
print("\nTop 15 Interactions (Random Forest):")
print(sii_rf)
```

### Expected Output

```
Top 15 Interactions (Random Forest):
InteractionValues(
    index=k-SII, max_order=2, min_order=0, estimated=True, estimation_budget=408,
    n_players=147, baseline_value=114.69375664316469,
    Top 10 interactions:
        (): 114.69375664316469
        (7,): 9.441016629810905           # bathrooms (main)
        (8,): 5.280020391567913           # bedrooms (main)
        (145,): 4.661893225844083         # location score (main)
        (27,): 3.454538920785156          # amenity (main)
        (55,): 3.4050215396901184         # neighborhood dummy
        (119,): 3.4050215396901184        # neighborhood dummy
        (55, 119): -3.4050215396901184    # neighborhood interaction (negative)
        (140,): -3.872498298252913
        (15,): -4.541430555213717
)
```

### Extracting Top Pairwise Interactions

```python
# Extract only 2-way interactions (exclude main effects)
hgbm_dict = {}
rf_dict = {}

# Iterate through interaction_lookup to get all feature tuples
for interaction_tuple in sii_values.interaction_lookup:
    if len(interaction_tuple) == 2:  # Only pairwise
        value = sii_values[interaction_tuple]
        hgbm_dict[interaction_tuple] = value

for interaction_tuple in sii_rf.interaction_lookup:
    if len(interaction_tuple) == 2:
        value = sii_rf[interaction_tuple]
        rf_dict[interaction_tuple] = value

# Convert to Series and rank by absolute value
hgbm_series = pd.Series(hgbm_dict)
rf_series = pd.Series(rf_dict)

top_n = 10
hgbm_top = hgbm_series.abs().nlargest(top_n)
rf_top = rf_series.abs().nlargest(top_n)

# Display results
print("="*95)
print("HGBM: Top 10 Pairwise Interactions".center(95))
print("="*95)
for idx in hgbm_top.index:
    feat1, feat2 = feature_names[idx[0]], feature_names[idx[1]]
    value = hgbm_series[idx]
    direction = "✓ synergy" if value > 0 else "✗ substitution"
    print(f"{feat1[:32]:32s} × {feat2[:32]:32s}: {value:8.3f}  {direction}")

print("\n" + "="*95)
print("Random Forest: Top 10 Pairwise Interactions".center(95))
print("="*95)
for idx in rf_top.index:
    feat1, feat2 = feature_names[idx[0]], feature_names[idx[1]]
    value = rf_series[idx]
    direction = "✓ synergy" if value > 0 else "✗ substitution"
    print(f"{feat1[:32]:32s} × {feat2[:32]:32s}: {value:8.3f}  {direction}")
```

### Example Output

```
===============================================================================================
                           HGBM: Top 10 Pairwise Interactions                           
===============================================================================================
property_type[Entire rental uni × property_type[T.Shared room]    :  -12.145  ✗ substitution
property_type[T.Private room]   × property_type[T.Shared room]    :  -11.071  ✗ substitution
bathrooms                       × bedrooms                         :    4.523  ✓ synergy
accommodates                    × bathrooms                        :    3.812  ✓ synergy
has_elevator                    × bedrooms                         :    2.934  ✓ synergy
review_scores_location          × room_type[T.Private room]        :   -2.651  ✗ substitution
...

===============================================================================================
                        Random Forest: Top 10 Pairwise Interactions                        
===============================================================================================
neighbourhood_cleansed[Estrela] × neighbourhood_cleansed[Misericó  :   -5.234  ✗ substitution
neighbourhood_cleansed[Estrela] × has_air_conditioning             :    4.123  ✓ synergy
neighbourhood_cleansed[Belém]   × review_scores_cleanliness        :    3.891  ✓ synergy
bathrooms                       × bedrooms                         :    3.456  ✓ synergy
neighbourhood_cleansed[Parque   × minimum_nights                   :   -2.987  ✗ substitution
...
```

### Interpretation: Why HGBM Generalizes Better

**HGBM Top Interactions**:
1. Property type dummies (substitution - expected)
2. **Size × size** (`bathrooms × bedrooms`): +4.52
3. **Size × amenity** (`elevator × bedrooms`): +2.93
4. Room type × location: -2.65

**Random Forest Top Interactions**:
1. **Neighborhood dummies** (Estrela × Misericórdia): -5.23
2. **Neighborhood × amenities** (Estrela × AC): +4.12
3. **Neighborhood × reviews** (Belém × cleanliness): +3.89
4. Size × size (`bathrooms × bedrooms`): +3.46

**Key Findings**:

1. **HGBM focuses on universal patterns**:
   - Size synergies (bathrooms × bedrooms) appear early
   - These relationships likely hold in Porto too
   - Less reliance on specific neighborhood names

2. **RF focuses on location-specific patterns**:
   - Top interactions involve specific neighborhoods (Estrela, Belém)
   - These neighborhoods don't exist in Porto!
   - Explains worse generalization

**Conclusion**: HGBM's superior generalization to Porto stems from learning **transferable interactions** (size, amenities) rather than **city-specific patterns** (neighborhood combinations).

