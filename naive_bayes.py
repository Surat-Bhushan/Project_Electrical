import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# =====================================================
# Part 1: Dataset Loading and Inspection
# =====================================================

df = pd.read_csv("PlayTennis.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df["play"].value_counts())

target_col = "play"
features = ["outlook", "temp", "humidity", "windy"]

print("\nTarget Column:")
print(target_col)

print("\nPredictor Features:")
print(features)

# =====================================================
# Part 2: Prior Probabilities
# =====================================================

def compute_priors(df, target_col):
    return df[target_col].value_counts(normalize=True).to_dict()

priors = compute_priors(df, target_col)

print("\nPrior Probabilities:")
print(priors)

# =====================================================
# Part 3: Query Sample
# =====================================================

query = {
    "outlook": "sunny",
    "temp": "cool",
    "humidity": "high",
    "windy": True
}

print("\nQuery Sample:")
print(query)

# =====================================================
# Part 4: Conditional Probabilities
# =====================================================

def compute_conditionals(df, features, target_col, query):

    conditionals = {}

    for cls in df[target_col].unique():

        subset = df[df[target_col] == cls]

        conditionals[cls] = {}

        for feature in features:

            prob = (
                subset[feature] == query[feature]
            ).sum() / len(subset)

            conditionals[cls][feature] = prob

    return conditionals


conditionals = compute_conditionals(
    df,
    features,
    target_col,
    query
)

print("\nConditional Probabilities:")
print(conditionals)

# =====================================================
# Part 5: Posterior Scores and Prediction
# =====================================================

def compute_posteriors(priors, conditionals):

    scores = {}

    for cls in priors:

        score = priors[cls]

        for prob in conditionals[cls].values():
            score *= prob

        scores[cls] = score

    return scores


scores = compute_posteriors(
    priors,
    conditionals
)

print("\nPosterior Scores:")
print(scores)

total = sum(scores.values())

normalized_scores = {
    k: v / total
    for k, v in scores.items()
}

print("\nNormalized Posterior Probabilities:")
print(normalized_scores)

print("\nSum:")
print(sum(normalized_scores.values()))

prediction = max(scores, key=scores.get)

print("\nManual Prediction:")
print(prediction)

# =====================================================
# Part 7: Verification Using Scikit-Learn
# =====================================================

X = df[features].copy()

encoders = {}

for col in X.columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    encoders[col] = le

y_encoder = LabelEncoder()

y = y_encoder.fit_transform(df[target_col])

model = CategoricalNB()

model.fit(X, y)

query_encoded = [[
    encoders["outlook"].transform(["sunny"])[0],
    encoders["temp"].transform(["cool"])[0],
    encoders["humidity"].transform(["high"])[0],
    encoders["windy"].transform([True])[0]
]]

prediction_sklearn = model.predict(query_encoded)

print("\nScikit-Learn Prediction:")
print(
    y_encoder.inverse_transform(prediction_sklearn)[0]
)

# =====================================================
# Part 8: Reflection
# =====================================================

print("\nReflection:")
print("Prior: Probability of a class before observing features.")
print("Conditional: Probability of a feature given a class.")
print("Posterior: Probability of a class after observing features.")
print("Naive Bayes assumes conditional independence among features.")