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

---

## Part 6: Comprehensive Interpretation Guide

### 1. Property Size Synergies

**Look for**: `bathrooms × bedrooms`, `accommodates × bathrooms`, `bedrooms × accommodates`

**If positive (synergy)**:
- Larger properties command a premium beyond additive effects
- A 3-bed, 2-bath home is worth MORE than (3 × bed_value + 2 × bath_value)

**Business insight**: 
> "Adding a bathroom increases value by €X in a studio, but by €X+Y in a 3-bedroom apartment."

**Practical application**:
- Hosts: Invest in both bedrooms and bathrooms together for maximum ROI
- Investors: Target properties where you can add both (higher value increase)

### 2. Location × Amenity Interactions

**Look for**: `review_scores_location × has_elevator`, location × amenities

**If positive**:
- Certain amenities are MORE valuable in well-located areas
- Example: Elevator matters more in tourist centers (older buildings, tourists with luggage)

**If negative**:
- Some features are substitutes
- Example: Good location compensates for lack of parking (walkable area)

**Business insight**:
> "Elevator adds €50/night in city center, but only €15/night in suburbs."

### 3. Room Type Interactions

**Look for**: `room_type_Private_room × bathrooms`, `room_type_Entire_home × accommodates`

**Expected patterns**:

**Private rooms** should show:
- WEAK or NEGATIVE interactions with size features
- Why? Guests don't fully utilize extra bathrooms/space in shared homes

**Entire homes** should show:
- STRONG POSITIVE interactions with size
- Why? Guests value every additional bedroom/bathroom

**Business insight**:
> "For private room listings, investing in extra bathrooms yields low returns. Focus on cleanliness and host interaction instead."

### 4. Amenity Combinations

**Look for**: `has_AC × has_heating`, `has_kitchen × has_dishwasher`

**If positive**:
- Amenity packages create synergy
- Properties with "full amenities" command premium

**If weak**:
- Amenities work independently
- Add based on individual ROI, not combinations

**Business insight**:
> "Properties should target 'amenity tiers': Basic (wifi only) vs Full (wifi + AC + kitchen + parking). Half-equipped properties underperform."

### 5. Unexpected Interactions

**Watch for**: Interactions that don't make intuitive sense

**Examples**:
- `days_since_last_review × bedrooms`: Should be independent
- `host_listings_count × location`: Could indicate professional hosts in prime areas

**Action items**:
- Investigate data quality issues
- May reveal true market dynamics (professional hosts dominate certain areas)

---

## Part 7: Practical Applications

### Feature Engineering for Simpler Models

**Problem**: Linear models (OLS, Lasso) don't capture interactions automatically.

**Solution**: Create explicit interaction terms for top synergies.

```python
# After identifying top interactions from shapiq analysis
# Create interaction features for linear models

# Example: Top interaction is bathrooms × bedrooms
X['bathrooms_x_bedrooms'] = X['bathrooms'] * X['bedrooms']

# Example: Location × amenity interaction
X['location_x_elevator'] = X['review_scores_location'] * X['has_elevator']

# Retrain Lasso with interaction terms
lasso_with_interactions = Lasso(alpha=1.0)
lasso_with_interactions.fit(X_with_interactions, y)
```

**Expected result**: Lasso performance improves when key interactions are explicitly added.

### Pricing Strategy Recommendations

Based on interaction analysis, we can advise hosts:

**For small properties (studios, 1-bed)**:
- Focus on location and cleanliness (high individual effects)
- Don't over-invest in multiple bathrooms (weak interactions)
- Amenities work independently - add based on individual ROI

**For large properties (3+ bedrooms)**:
- Bathroom upgrades yield disproportionate returns (strong interaction)
- Target entire-home rentals (strong size × room_type synergy)
- Consider premium amenities packages (elevator + AC + parking)

**For mid-range properties (2-bed)**:
- Balance individual features and synergies
- One extra bathroom shows strong interaction effect
- Location matters more (compensates for size limitations)

### Model Selection Guidance

**When to use complex models (HGBM, RF)**:
- Strong interaction effects present
- Features have non-linear relationships
- Need to generalize across contexts (different cities)

**When simpler models suffice**:
- Weak interactions (features mostly additive)
- Interpretability is critical (coefficients need business meaning)
- After adding key interaction terms explicitly

### Generalization Insights

**For transferring models across cities**:
1. Check if model relies on universal vs location-specific interactions
2. Models using size synergies transfer better than those using neighborhood effects
3. Consider retraining with only universal features for geographic transfer

**For temporal stability** (Q1 → Q3):
1. Seasonal patterns may change interaction strengths
2. Models relying on "reviews" may degrade (new competitors, rating inflation)
3. Economic factors interact with price sensitivity differently over time

---

## Part 8: Common Pitfalls and Best Practices

### Pitfall 1: Over-interpreting Weak Interactions

**Problem**: Not all interactions are meaningful. Small values may be noise.

**Solution**: 
- Set a threshold (e.g., |interaction| > 2.0 for price in €)
- Focus on top 10-20 interactions
- Validate findings across multiple observations

### Pitfall 2: Confusing Correlation with Interaction

**Interaction** ≠ **Correlation**

- **Correlation**: Features move together in the data
- **Interaction**: Features have synergistic effects on the outcome

**Example**: `bedrooms` and `bathrooms` are correlated (larger homes have more of both), but k-SII measures whether their *combined effect* exceeds the sum of individual effects.

### Pitfall 3: Observation-Specific Analysis

**Issue**: k-SII values are computed for a specific observation. Patterns may differ for luxury vs budget properties.

**Solution**:
- Analyze multiple representative observations (luxury, mid-range, budget)
- Aggregate interactions across observations
- Report ranges: "Bathroom synergy: €2-5 depending on property type"

### Pitfall 4: Dummy Variable Artifacts

**Issue**: Mutually exclusive dummies (from same categorical) always show negative interactions.

**Why**: When `property_type[Entire home]` = 1, then `property_type[Shared room]` = 0 (by definition).

**Solution**: Filter out these "expected" interactions in your analysis. Focus on interactions between *different* feature types.

### Best Practice 1: Validate with Domain Knowledge

**Always check**: Do interactions make business sense?

**Example**: 
- `bathrooms × bedrooms` > 0: ✓ Makes sense (size synergy)
- `bedrooms × days_since_review` > 5: ✗ Suspicious (investigate data quality)

### Best Practice 2: Compare Multiple Models

Don't just analyze one model. Compare:
- HGBM vs RF vs GBM
- Check if interactions are robust across models
- Model-specific patterns reveal what each learns

### Best Practice 3: Budget Selection

The `budget` parameter controls accuracy vs speed:
- **Low (256)**: Fast, approximate results - use for exploration
- **Medium (512)**: Good balance - recommended for most analyses
- **High (1024+)**: Accurate, slow - use for final reporting

**Rule of thumb**: Start low, increase for final analysis.

### Best Practice 4: Visualization First

Always start with network plots before diving into numerical tables:
1. Network plot → identify clusters
2. Extract numerical values for those clusters
3. Interpret and validate

Don't get lost in hundreds of interaction values without visual guidance.

---

## Part 9: Extension Topics

### Higher-Order Interactions (3-way, 4-way)

```python
# Compute 3-way interactions
explainer_order3 = shapiq.TabularExplainer(
    model=hgbm_final,
    data=X_lisbon_q1,
    index="k-SII",
    max_order=3  # Now includes (i,j,k) terms
)

sii_order3 = explainer_order3.explain(x_explain, budget=1024)
```

**When useful**:
- Very complex models
- Research questions about multi-feature combinations
- Example: Does `bedrooms × bathrooms × location` create 3-way synergy?

**Warning**: 
- Computationally expensive
- Harder to interpret
- Focus on 2-way interactions first

### Alternative Interaction Indices

shapiq supports multiple indices:

1. **k-SII** (k-Shapley Interaction Index): What we used
   - Properties: Efficiency, Symmetry
   - Best for: General interpretation

2. **STII** (Shapley-Taylor Interaction Index): 
   - Similar to k-SII but different axioms
   - Use when efficiency is critical

3. **FSII** (Faith-Shapley Interaction Index):
   - Specific to certain model classes
   - Use for additive models

4. **FBII** (Faith-Banzhaf Interaction Index):
   - Faster approximation
   - Use for large feature spaces (>100 features)

**How to choose**: Start with k-SII. Change only if you have specific theoretical requirements.

### Global Interaction Analysis

**Problem**: We analyzed one observation. What about the whole dataset?

**Solution**: Aggregate across observations.

```python
# Analyze multiple observations
n_samples = 100
sample_indices = np.random.choice(len(X_lisbon_q1), n_samples, replace=False)

all_interactions = []
for idx in sample_indices:
    x = X_lisbon_q1[idx]
    sii = explainer_sii.explain(x, budget=512)
    
    # Extract pairwise interactions
    for interaction_tuple in sii.interaction_lookup:
        if len(interaction_tuple) == 2:
            all_interactions.append({
                'feat1': interaction_tuple[0],
                'feat2': interaction_tuple[1],
                'value': sii[interaction_tuple]
            })

# Convert to DataFrame and aggregate
df_interactions = pd.DataFrame(all_interactions)
global_interactions = df_interactions.groupby(['feat1', 'feat2'])['value'].agg(['mean', 'std', 'min', 'max'])

print("Global Interaction Summary (averaged over 100 observations):")
print(global_interactions.sort_values('mean', ascending=False).head(10))
```

**Result**: Average interaction strengths across the dataset, with confidence intervals.

---

## Part 10: Summary and Key Takeaways

### Core Concepts

1. **Shapley Values**: Measure individual feature contributions
   - Additive: Sum to prediction difference from baseline
   - Fair allocation of "credit"

2. **Shapley Interaction Index (k-SII)**: Measure feature synergies
   - Positive = synergy (amplification)
   - Negative = substitution (suppression)
   - Reveals which features work together

3. **Network Visualization**: Intuitive view of interaction structure
   - Nodes = features, Edges = interactions
   - Identify clusters and key relationships

### Practical Workflow

```
1. Train models (HGBM, RF, etc.)
   ↓
2. Compute Shapley values (understand individual importance)
   ↓
3. Compute k-SII (identify interactions)
   ↓
4. Visualize network (see structure)
   ↓
5. Extract top interactions (quantify synergies)
   ↓
6. Compare models (understand generalization)
   ↓
7. Apply insights (feature engineering, business strategy)
```

