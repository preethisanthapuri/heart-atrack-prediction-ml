"""Baseline classifier definitions for the discovered heart dataset."""
from __future__ import annotations
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def build_baseline_models() -> dict[str, object]:
    """Return diverse, classical classifiers suitable for a small tabular dataset."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Support Vector Classifier (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }
