"""
Treina um RandomForestClassifier no dataset sintetico e salva o modelo
para inferencia rapida (substituindo a chamada ao LLM apenas para a etapa
de classificacao).

O LLM continua sendo usado downstream, mas so para gerar a justificativa
em linguagem natural - diminuindo drasticamente o tempo de resposta.

Uso:
    python data/treinar_rf.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "triagem_dataset.csv"
MODEL_PATH = DATA_DIR / "rf_model.pkl"


def carregar():
    df = pd.read_csv(CSV_PATH)
    features = [
        "idade", "frequencia_cardiaca", "spo2", "temperatura",
        "pa_sistolica", "pa_diastolica", "sexo_M", "sexo_F",
    ] + [c for c in df.columns if c.startswith("flag_")]
    return df, features


def treinar():
    df, features = carregar()
    X, y = df[features].fillna(0), df["classificacao"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("=" * 60)
    print("Performance no holdout (20%)")
    print("=" * 60)
    print(classification_report(y_test, preds, digits=3))

    print("=" * 60)
    print("Matriz de confusao")
    print("=" * 60)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    header = "real / pred"
    print(f"{header:<12}" + "".join(f"{l:>10}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<12}" + "".join(f"{cm[i,j]:>10}" for j in range(len(labels))))

    print()
    print("=" * 60)
    print("Validacao cruzada (5-fold)")
    print("=" * 60)
    scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted", n_jobs=-1)
    print(f"F1 ponderado: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Persistencia (modelo + lista de features para reusar na inferencia)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"model": clf, "features": features}, f)
    print()
    print(f"[OK] Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    treinar()
