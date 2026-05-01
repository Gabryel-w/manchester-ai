"""
Runner de avaliação do golden set contra os modelos Groq.

Executa cada vinheta do golden_set.json em todos os modelos Groq do catálogo
de agent.py, mede acurácia (cor exata e ± 1 cor), latência, e gera matriz de
confusão. Salva relatório em eval/results_<timestamp>.json para versionamento.

Uso (com venv ativo, GROQ_API_KEY configurada no .env):
    python eval/runner.py

Para limitar a um único modelo:
    python eval/runner.py --modelo llama-3.3-70b-versatile

Para incluir os modelos Ollama (precisa do serviço local rodando):
    python eval/runner.py --com-ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Permite rodar como `python eval/runner.py` a partir da raiz do projeto.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402  (import depois do sys.path)
    MODELOS_GROQ,
    MODELOS_OLLAMA,
    GroqBackend,
    OllamaBackend,
    classificar,
)

CORES_ORDEM = ["VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL"]
GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def _dist_cor(esperado: str, obtido: str) -> int | None:
    """Distância em níveis Manchester (0 = acerto exato, 1 = adjacente, ...)."""
    if esperado not in CORES_ORDEM or obtido not in CORES_ORDEM:
        return None
    return abs(CORES_ORDEM.index(esperado) - CORES_ORDEM.index(obtido))


def _carregar_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden set não encontrado em {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _avaliar_modelo(modelo: str, backend_factory, casos: list[dict]) -> dict:
    """Roda todas as vinhetas em um modelo e devolve o relatório agregado."""
    print(f"\n=== {modelo} ===")
    resultados: list[dict] = []
    matriz: dict[str, dict[str, int]] = {c: {d: 0 for d in CORES_ORDEM} for c in CORES_ORDEM}
    acertos_exatos = 0
    acertos_adjacentes = 0
    erros_graves = 0  # distância >= 2
    latencias: list[float] = []

    backend = backend_factory(modelo)

    for caso in casos:
        esperado = caso["esperado"]
        t0 = time.perf_counter()
        try:
            res = classificar(caso["dados"], backend)
            obtido = res.classificacao
            erro_msg = res.erro
            inconsistencia = res.inconsistencia
            cor_regra = res.cor_regra
        except Exception as e:  # falha de rede / API
            obtido = "ERRO"
            erro_msg = str(e)
            inconsistencia = False
            cor_regra = None
        latencia = time.perf_counter() - t0
        latencias.append(latencia)

        dist = _dist_cor(esperado, obtido)
        if dist == 0:
            acertos_exatos += 1
            acertos_adjacentes += 1
            simbolo = "OK"
        elif dist == 1:
            acertos_adjacentes += 1
            simbolo = "~"
        elif dist is None:
            erros_graves += 1
            simbolo = "X"
        else:
            erros_graves += 1
            simbolo = "XX"

        if esperado in matriz and obtido in matriz[esperado]:
            matriz[esperado][obtido] += 1

        print(
            f"  [{simbolo:2}] {caso['id']:9} esp={esperado:8} got={obtido:8} "
            f"({latencia:.2f}s)"
            + (f"  trava→{cor_regra}" if inconsistencia else "")
            + (f"  ERR: {erro_msg}" if erro_msg else "")
        )

        resultados.append(
            {
                "id": caso["id"],
                "descricao_curta": caso["descricao_curta"],
                "esperado": esperado,
                "obtido": obtido,
                "distancia": dist,
                "latencia_s": round(latencia, 3),
                "inconsistencia": inconsistencia,
                "cor_regra": cor_regra,
                "erro": erro_msg,
            }
        )

    n = len(casos)
    relatorio = {
        "modelo": modelo,
        "n_casos": n,
        "acuracia_exata": round(acertos_exatos / n, 4) if n else 0,
        "acuracia_adjacente": round(acertos_adjacentes / n, 4) if n else 0,
        "erros_graves": erros_graves,
        "latencia_media_s": round(sum(latencias) / n, 3) if n else 0,
        "latencia_max_s": round(max(latencias), 3) if latencias else 0,
        "matriz_confusao": matriz,
        "casos": resultados,
    }

    print(
        f"  → acerto exato: {acertos_exatos}/{n} ({relatorio['acuracia_exata']:.0%}) · "
        f"± 1 cor: {acertos_adjacentes}/{n} ({relatorio['acuracia_adjacente']:.0%}) · "
        f"erros graves: {erros_graves} · latência média: {relatorio['latencia_media_s']}s"
    )
    return relatorio


def _imprimir_matriz(modelo: str, matriz: dict) -> None:
    """Renderiza matriz de confusão 5×5 como tabela ASCII."""
    print(f"\nMatriz de confusão — {modelo}")
    cab = "esperado \\ obtido".ljust(20) + " ".join(c[:6].rjust(7) for c in CORES_ORDEM)
    print(cab)
    print("-" * len(cab))
    for linha in CORES_ORDEM:
        cells = " ".join(str(matriz[linha][col]).rjust(7) for col in CORES_ORDEM)
        print(linha.ljust(20) + cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", help="Roda apenas o modelo indicado (ex.: llama-3.1-8b-instant)")
    parser.add_argument("--com-ollama", action="store_true", help="Inclui modelos Ollama no run")
    args = parser.parse_args()

    casos = _carregar_golden()
    print(f"Golden set carregado: {len(casos)} vinhetas (de {GOLDEN_PATH})")

    alvos: list[tuple[str, callable]] = []
    if args.modelo:
        # Usuário informou um modelo só. Decide pelo nome se é Groq ou Ollama.
        if args.modelo in MODELOS_GROQ:
            alvos.append((args.modelo, GroqBackend))
        elif args.modelo in MODELOS_OLLAMA:
            alvos.append((args.modelo, OllamaBackend))
        else:
            print(f"Modelo '{args.modelo}' não está em nenhum catálogo conhecido.")
            return 1
    else:
        for m in MODELOS_GROQ:
            alvos.append((m, GroqBackend))
        if args.com_ollama:
            for m in MODELOS_OLLAMA:
                alvos.append((m, OllamaBackend))

    relatorios: list[dict] = []
    for modelo, factory in alvos:
        try:
            relatorios.append(_avaliar_modelo(modelo, factory, casos))
        except Exception as e:
            print(f"Falha ao avaliar {modelo}: {e}")

    # Resumo comparativo + matrizes.
    print("\n" + "=" * 60)
    print("RESUMO COMPARATIVO")
    print("=" * 60)
    print(f"{'modelo':<32} {'exato':>8} {'± 1 cor':>10} {'erro grave':>12} {'lat. med.':>10}")
    for r in relatorios:
        print(
            f"{r['modelo']:<32} "
            f"{r['acuracia_exata']:>8.0%} "
            f"{r['acuracia_adjacente']:>10.0%} "
            f"{r['erros_graves']:>12} "
            f"{r['latencia_media_s']:>9}s"
        )

    for r in relatorios:
        _imprimir_matriz(r["modelo"], r["matriz_confusao"])

    # Salva o relatório bruto em JSON para versionamento.
    saida = {
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_casos": len(casos),
        "relatorios": relatorios,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"results_{timestamp}.json"
    out_path.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório completo salvo em: {out_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
