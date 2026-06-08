import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# Part A: Dataset Inspection
# =====================================================

df = pd.read_csv("PlayTennis.csv")

print("Dataset:")
print(df)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

target_col = "play"
features = ["outlook", "temp", "humidity", "windy"]

print("\nTarget Column:")
print(target_col)

for feature in features:

    print(f"\nUnique values in {feature}:")
    print(df[feature].unique())

print("\nClass Distribution:")
print(df[target_col].value_counts())

# =====================================================
# Encode Dataset
# =====================================================

X = df[features].copy()

encoders = {}

for col in X.columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    encoders[col] = le

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    df[target_col]
)

print("\nEncoded Feature Matrix:")
print(X.head())

print("\nEncoded Target:")
print(y)

# =====================================================
# Part B: Bootstrap Sampling
# =====================================================

print("\n============================")
print("BOOTSTRAP SAMPLES")
print("============================")

bootstrap_samples = []

for seed in [1, 2, 3]:

    sample = df.sample(
        n=len(df),
        replace=True,
        random_state=seed
    )

    bootstrap_samples.append(sample)

    print(f"\nBootstrap Sample {seed}")

    print(
        "Rows Sampled:",
        len(sample)
    )

    duplicated_rows = sample[
        sample.duplicated()
    ]

    print(
        "\nRepeated Rows:"
    )

    print(duplicated_rows)

    oob_indices = list(
        set(df.index) -
        set(sample.index)
    )

    print(
        "\nOut-of-Bag Indices:"
    )

    print(oob_indices)

# =====================================================
# Train 3 Manual Trees
# =====================================================

print("\n============================")
print("MANUAL RANDOM FOREST")
print("============================")

feature_sets = [
    ["outlook", "humidity"],
    ["temp", "windy"],
    ["outlook", "windy"]
]

manual_trees = []

for i in range(3):

    sample = bootstrap_samples[i]

    X_sample = sample[
        feature_sets[i]
    ].copy()

    for col in X_sample.columns:

        le = LabelEncoder()

        X_sample[col] = le.fit_transform(
            X_sample[col]
        )

    y_sample = target_encoder.fit_transform(
        sample[target_col]
    )

    tree = DecisionTreeClassifier(
        max_depth=2,
        random_state=i
    )

    tree.fit(
        X_sample,
        y_sample
    )

    manual_trees.append(tree)

    print(
        f"\nTree {i+1} trained using features:"
    )

    print(feature_sets[i])

# =====================================================
# Query Sample Prediction
# =====================================================

query = pd.DataFrame({
    "outlook": ["sunny"],
    "temp": ["cool"],
    "humidity": ["high"],
    "windy": [True]
})

votes = []

for i in range(3):

    subset_features = feature_sets[i]

    query_subset = query[
        subset_features
    ].copy()

    for col in query_subset.columns:

        le = LabelEncoder()

        le.fit(df[col])

        query_subset[col] = le.transform(
            query_subset[col]
        )

    pred = manual_trees[i].predict(
        query_subset
    )[0]

    votes.append(pred)

    print(
        f"\nTree {i+1} Prediction:",
        target_encoder.inverse_transform(
            [pred]
        )[0]
    )

majority_vote = max(
    set(votes),
    key=votes.count
)

print(
    "\nManual Forest Prediction:"
)

print(
    target_encoder.inverse_transform(
        [majority_vote]
    )[0]
)

# =====================================================
# Part C: Single Tree vs Random Forest
# =====================================================

print("\n============================")
print("SINGLE TREE VS RANDOM FOREST")
print("============================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Single Tree

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

# Random Forest

rf = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

# Accuracies

dt_train_acc = accuracy_score(
    y_train,
    dt.predict(X_train)
)

dt_test_acc = accuracy_score(
    y_test,
    dt.predict(X_test)
)

rf_train_acc = accuracy_score(
    y_train,
    rf.predict(X_train)
)

rf_test_acc = accuracy_score(
    y_test,
    rf.predict(X_test)
)

print(
    "\nDecision Tree Train Accuracy:",
    dt_train_acc
)

print(
    "Decision Tree Test Accuracy:",
    dt_test_acc
)

print(
    "\nRandom Forest Train Accuracy:",
    rf_train_acc
)

print(
    "Random Forest Test Accuracy:",
    rf_test_acc
)

print("\nPredicted Labels (Tree):")

print(
    dt.predict(X_test)
)

print("\nPredicted Labels (Forest):")

print(
    rf.predict(X_test)
)

print("\nTrue Labels:")

print(y_test)

# =====================================================
# Display Tree Rules
# =====================================================

print("\nDecision Tree Rules:")

print(
    export_text(
        dt,
        feature_names=list(X.columns)
    )
)

print("\nRandom Forest Parameters")

print(
    "n_estimators =",
    rf.n_estimators
)

print(
    "max_depth =",
    rf.max_depth
)

print(
    "random_state =",
    rf.random_state
)

# =====================================================
# Part D: Manual Ensemble vs Inbuilt Forest
# =====================================================

print("\n============================")
print("COMPARISON TABLE")
print("============================")

for idx in range(
    min(3, len(X_test))
):

    sample = X_test.iloc[[idx]]

    tree_predictions = []

    for tree_no in range(3):

        selected_features = (
            feature_sets[tree_no]
        )

        pred = manual_trees[
            tree_no
        ].predict(
            sample[selected_features]
        )[0]

        tree_predictions.append(pred)

    majority = max(
        set(tree_predictions),
        key=tree_predictions.count
    )

    rf_pred = rf.predict(sample)[0]

    print("\nSample Index:", idx)

    print(
        "Tree 1:",
        target_encoder.inverse_transform(
            [tree_predictions[0]]
        )[0]
    )

    print(
        "Tree 2:",
        target_encoder.inverse_transform(
            [tree_predictions[1]]
        )[0]
    )

    print(
        "Tree 3:",
        target_encoder.inverse_transform(
            [tree_predictions[2]]
        )[0]
    )

    print(
        "Majority Vote:",
        target_encoder.inverse_transform(
            [majority]
        )[0]
    )

    print(
        "Random Forest:",
        target_encoder.inverse_transform(
            [rf_pred]
        )[0]
    )

    print(
        "True Class:",
        target_encoder.inverse_transform(
            [y_test[idx]]
        )[0]
    )

# =====================================================
# Part E: Effect of Number of Trees
# =====================================================

print("\n============================")
print("NUMBER OF TREES EXPERIMENT")
print("============================")

tree_counts = [1, 3, 5, 10, 20]

train_accuracy = []
test_accuracy = []

for n in tree_counts:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    train_acc = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    train_accuracy.append(
        train_acc
    )

    test_accuracy.append(
        test_acc
    )

    print(
        f"Trees={n} | Train={train_acc:.3f} | Test={test_acc:.3f}"
    )

# Plot

plt.figure(figsize=(8,5))

plt.plot(
    tree_counts,
    train_accuracy,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    tree_counts,
    test_accuracy,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel("Number of Trees")

plt.ylabel("Accuracy")

plt.title(
    "Random Forest Performance"
)

plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# Part F: Reflection
# =====================================================

print("\nReflection")

print(
    "\n1. Three main ideas:"
)

print(
    "- Bootstrap Sampling"
)

print(
    "- Random Feature Selection"
)

print(
    "- Majority Voting"
)

print(
    "\n2. Random Forest combines multiple trees whereas a Decision Tree uses only one tree."
)

print(
    "\n3. Random Forest is an ensemble because multiple models work together."
)

print(
    "\n4. Majority voting determines the final class prediction."
)

print(
    "\n5. Comparing manual and library implementations helps verify understanding of the algorithm."
)
