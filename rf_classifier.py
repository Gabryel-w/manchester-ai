"""
Wrapper do classificador Random Forest treinado em data/triagem_dataset.csv.

Usado pelo agent.py quando o backend escolhido e Ollama (local). Substitui
a etapa de classificacao do LLM por inferencia em milissegundos, deixando
o LLM responsavel apenas por gerar a justificativa em linguagem natural.

A inferencia carrega o modelo uma unica vez (singleton) e mantem em memoria
durante o ciclo de vida do processo.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from features import extrair_features_para_rf

MODEL_PATH = Path(__file__).parent / "data" / "rf_model.pkl"


class RFNaoTreinadoError(RuntimeError):
    """Disparado quando o pickle do modelo nao foi encontrado."""


class RFClassifier:
    """Singleton do RandomForestClassifier treinado."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise RFNaoTreinadoError(
                f"Modelo Random Forest nao encontrado em {MODEL_PATH}. "
                f"Rode: python data/treinar_rf.py"
            )
        with MODEL_PATH.open("rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        self.model = payload["model"]
        self.features: list[str] = payload["features"]
        # Lista de classes na ordem usada pelo modelo (predict_proba devolve
        # probabilidades nessa ordem).
        self.classes: list[str] = list(self.model.classes_)

    def predict(self, dados: dict) -> dict:
        """
        Classifica um paciente e devolve diagnostico estatistico.

        Returns:
            {
              "classificacao": "VERMELHO" | ...,
              "confianca":     "ALTA" | "MEDIA" | "BAIXA",
              "probabilidade": float (0-1) da classe vencedora,
              "distribuicao":  {cor: prob} para todas as 5 classes,
              "top_features":  [(nome_feature, valor, importancia_global), ...] top 5
            }
        """
        vetor = extrair_features_para_rf(dados, self.features)

        # Empacota como DataFrame com feature names para silenciar o
        # UserWarning do sklearn ("X does not have valid feature names").
        # O RF foi treinado a partir de um DataFrame, ele espera o mesmo
        # formato na inferencia.
        X = pd.DataFrame([vetor], columns=self.features)
        probs = self.model.predict_proba(X)[0]
        idx_winner = int(probs.argmax())
        cor = self.classes[idx_winner]
        max_prob = float(probs[idx_winner])

        # Calibracao simples de confianca
        if max_prob >= 0.70:
            confianca = "ALTA"
        elif max_prob >= 0.50:
            confianca = "MEDIA"
        else:
            confianca = "BAIXA"

        distribuicao = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

        # Top 5 features mais influentes para esta predicao
        # (importancia global do RF, nao SHAP por instancia, mas suficiente
        # para a justificativa).
        importancias = list(zip(self.features, self.model.feature_importances_))
        # Filtra so as features que estao "ativas" (valor > 0) e ordena por importancia
        ativas = [(f, vetor[i], imp) for i, (f, imp) in enumerate(importancias) if vetor[i] > 0]
        ativas.sort(key=lambda x: -x[2])
        top_features = ativas[:5]

        return {
            "classificacao": cor,
            "confianca": confianca,
            "probabilidade": max_prob,
            "distribuicao": distribuicao,
            "top_features": top_features,
        }


# ---------------------------------------------------------------------------
# Singleton lazy-loaded
# ---------------------------------------------------------------------------
_instance: RFClassifier | None = None


def get_rf() -> RFClassifier:
    """Carrega o modelo na primeira chamada e reusa nas seguintes."""
    global _instance
    if _instance is None:
        _instance = RFClassifier()
    return _instance


def rf_disponivel() -> bool:
    """Retorna True se o pickle do modelo existe (sem carregar)."""
    return MODEL_PATH.exists()
