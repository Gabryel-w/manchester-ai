"""
Camada de persistência SQLite para a fila viva de triagens.

Uma única tabela `triagens` guarda cada classificação feita no sistema, com
o status de atendimento. A fila viva consome `listar_fila()`, ordenada por
gravidade Manchester e ordem de chegada. O cronômetro do tempo máximo é
calculado no frontend a partir de `criado_em` + `tempo_max_min`.

sqlite3 é built-in do Python — não há dependência nova.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "triagens.db"

# Ordem usada no SQL para classificar gravidade. SQLite não tem ENUM, então
# fazemos um CASE inline. VERMELHO=0 (mais grave) → AZUL=4.
_ORDEM_SQL = """
    CASE classificacao
        WHEN 'VERMELHO' THEN 0
        WHEN 'LARANJA'  THEN 1
        WHEN 'AMARELO'  THEN 2
        WHEN 'VERDE'    THEN 3
        WHEN 'AZUL'     THEN 4
        ELSE 5
    END
"""

STATUS_VALIDOS = ("aguardando", "atendido", "dispensado")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Cria a tabela se não existir. Idempotente: pode ser chamado a cada startup."""
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triagens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                criado_em TEXT NOT NULL,
                enfermeiro TEXT NOT NULL,
                paciente_nome TEXT,
                idade INTEGER,
                sexo TEXT,
                pressao TEXT,
                frequencia_cardiaca INTEGER,
                spo2 INTEGER,
                temperatura REAL,
                sintomas TEXT,
                historico TEXT,
                classificacao TEXT NOT NULL,
                classificacao_llm TEXT,
                justificativa TEXT,
                sinais_alerta TEXT,
                perguntas_adicionais TEXT,
                confianca TEXT,
                inconsistencia INTEGER NOT NULL DEFAULT 0,
                cor_regra TEXT,
                motivos_regra TEXT,
                backend_usado TEXT,
                status TEXT NOT NULL DEFAULT 'aguardando',
                atualizado_em TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_triagens_status ON triagens(status)"
        )
        # Migração in-place: adiciona paciente_nome em bancos antigos (criados
        # antes da feature). ALTER TABLE ADD COLUMN é não destrutivo no SQLite.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(triagens)").fetchall()}
        if "paciente_nome" not in cols:
            conn.execute("ALTER TABLE triagens ADD COLUMN paciente_nome TEXT")


def inserir_triagem(payload: dict[str, Any]) -> int:
    """Insere uma triagem e devolve o id gerado.

    Espera um dict já com todos os campos do paciente + resultado serializados.
    Listas e objetos são serializados como JSON nas colunas TEXT.
    """
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO triagens (
                criado_em, enfermeiro, paciente_nome,
                idade, sexo, pressao, frequencia_cardiaca, spo2, temperatura,
                sintomas, historico,
                classificacao, classificacao_llm, justificativa,
                sinais_alerta, perguntas_adicionais, confianca,
                inconsistencia, cor_regra, motivos_regra,
                backend_usado, status, atualizado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'aguardando', ?)
            """,
            (
                agora,
                payload["enfermeiro"],
                payload.get("paciente_nome") or None,
                payload.get("idade"),
                payload.get("sexo"),
                payload.get("pressao"),
                payload.get("frequencia_cardiaca"),
                payload.get("spo2"),
                payload.get("temperatura"),
                payload.get("sintomas"),
                payload.get("historico"),
                payload["classificacao"],
                payload.get("classificacao_llm"),
                payload.get("justificativa"),
                json.dumps(payload.get("sinais_alerta") or [], ensure_ascii=False),
                json.dumps(payload.get("perguntas_adicionais") or [], ensure_ascii=False),
                payload.get("confianca"),
                1 if payload.get("inconsistencia") else 0,
                payload.get("cor_regra"),
                json.dumps(payload.get("motivos_regra") or [], ensure_ascii=False),
                payload.get("backend_usado"),
                agora,
            ),
        )
        return int(cursor.lastrowid)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for col in ("sinais_alerta", "perguntas_adicionais", "motivos_regra"):
        if d.get(col):
            try:
                d[col] = json.loads(d[col])
            except (TypeError, ValueError):
                d[col] = []
        else:
            d[col] = []
    d["inconsistencia"] = bool(d.get("inconsistencia"))
    return d


def listar_fila(incluir_finalizados: bool = False) -> list[dict[str, Any]]:
    """Lista pacientes da fila ordenados por gravidade e ordem de chegada.

    Por padrão devolve só os 'aguardando' (a fila propriamente dita). Com
    incluir_finalizados=True devolve tudo, útil para dashboards de histórico.
    """
    where = "" if incluir_finalizados else "WHERE status = 'aguardando'"
    sql = f"""
        SELECT * FROM triagens
        {where}
        ORDER BY {_ORDEM_SQL}, criado_em ASC
    """
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [_row_to_dict(r) for r in rows]


def atualizar_status(triagem_id: int, novo_status: str) -> bool:
    """Atualiza o status de uma triagem. Devolve False se id não existe ou status é inválido."""
    if novo_status not in STATUS_VALIDOS:
        return False
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as conn:
        cursor = conn.execute(
            "UPDATE triagens SET status = ?, atualizado_em = ? WHERE id = ?",
            (novo_status, agora, triagem_id),
        )
        return cursor.rowcount > 0
