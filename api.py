"""
API FastAPI para o Sistema de Triagem com IA.

Endpoints:
    GET  /                  -> serve a interface (frontend/index.html)
    GET  /api/health        -> health check
    GET  /api/models        -> lista modelos disponíveis por backend
    POST /api/triagem       -> executa a triagem com base nos dados do paciente

Execute com:
    uvicorn api:app --reload
ou (para acesso da rede local):
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
from agent import (
    MODELOS_GROQ,
    MODELOS_OLLAMA,
    classificar,
    get_backend,
)
from protocols import CORES

load_dotenv()

# Cria a tabela da fila viva no import. sqlite3.CREATE TABLE IF NOT EXISTS é
# idempotente, então roda a cada uvicorn reload sem custo perceptível.
db.init_db()

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Sistema de Triagem com IA",
    description="Apoio à decisão clínica usando LLM + Protocolo de Manchester",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# CORS aberto — útil em dev. Em produção, restringir aos hosts específicos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas (validação automática das requisições)
# ---------------------------------------------------------------------------
class TriagemRequest(BaseModel):
    paciente_nome: str = Field("", max_length=120)
    idade: int = Field(..., ge=0, le=120)
    sexo: Literal["Masculino", "Feminino", "Não informado"] = "Não informado"
    pressao: str = Field("", max_length=20)
    frequencia_cardiaca: int | None = Field(None, ge=20, le=250)
    spo2: int | None = Field(None, ge=50, le=100)
    temperatura: float | None = Field(None, ge=30.0, le=43.0)
    sintomas: str = Field(..., min_length=3, max_length=4000)
    historico: str = Field("", max_length=4000)
    backend: Literal["Groq", "Ollama"] = "Groq"
    modelo: str = Field(..., min_length=1)
    enfermeiro: str = Field(..., min_length=2, max_length=80)


class TriagemResponse(BaseModel):
    triagem_id: int
    classificacao: str
    classificacao_llm: str = ""
    justificativa: str
    sinais_alerta: list[str]
    perguntas_adicionais: list[str]
    confianca: str
    backend_usado: str
    cor_info: dict
    inconsistencia: bool = False
    cor_regra: str | None = None
    motivos_regra: list[str] = []
    erro: str = ""


class ModelosResponse(BaseModel):
    groq: list[str]
    ollama: list[str]
    cores: dict


class FilaItem(BaseModel):
    id: int
    criado_em: str
    enfermeiro: str
    paciente_nome: str | None = None
    idade: int | None = None
    sexo: str | None = None
    pressao: str | None = None
    frequencia_cardiaca: int | None = None
    spo2: int | None = None
    temperatura: float | None = None
    sintomas: str | None = None
    historico: str | None = None
    classificacao: str
    classificacao_llm: str | None = None
    justificativa: str | None = None
    sinais_alerta: list[str] = []
    perguntas_adicionais: list[str] = []
    confianca: str | None = None
    inconsistencia: bool = False
    cor_regra: str | None = None
    motivos_regra: list[str] = []
    backend_usado: str | None = None
    status: str
    tempo_max_min: int
    cor_info: dict


class FilaUpdateRequest(BaseModel):
    status: Literal["aguardando", "atendido", "dispensado"]


# ---------------------------------------------------------------------------
# Endpoints da API
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models", response_model=ModelosResponse)
def models():
    """Lista modelos disponíveis e metadados das cores do Manchester."""
    return ModelosResponse(
        groq=MODELOS_GROQ,
        ollama=MODELOS_OLLAMA,
        cores=CORES,
    )


@app.post("/api/triagem", response_model=TriagemResponse)
def triagem(req: TriagemRequest):
    """Executa a triagem do paciente."""
    try:
        backend = get_backend(req.backend, req.modelo)
    except RuntimeError as e:
        # GROQ_API_KEY ausente, etc.
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dados = {
        "idade": req.idade,
        "sexo": req.sexo,
        "pressao": req.pressao,
        "frequencia_cardiaca": req.frequencia_cardiaca,
        "spo2": req.spo2,
        "temperatura": req.temperatura,
        "sintomas": req.sintomas,
        "historico": req.historico,
    }

    resultado = classificar(dados, backend)

    cor_info = CORES.get(
        resultado.classificacao,
        {
            "tempo_max": -1,
            "descricao": "Classificação não reconhecida.",
            "hex_fundo": "#6C757D",
            "hex_texto": "#FFFFFF",
            "icone": "⚪",
        },
    )

    # Persiste a triagem para a fila viva. Mesmo classificações com erro são
    # gravadas — assim o operador vê na fila um item em estado degradado em vez
    # do registro sumir silenciosamente.
    triagem_id = db.inserir_triagem(
        {
            **dados,
            "enfermeiro": req.enfermeiro,
            "paciente_nome": req.paciente_nome,
            "classificacao": resultado.classificacao,
            "classificacao_llm": resultado.classificacao_llm,
            "justificativa": resultado.justificativa,
            "sinais_alerta": resultado.sinais_alerta,
            "perguntas_adicionais": resultado.perguntas_adicionais,
            "confianca": resultado.confianca,
            "inconsistencia": resultado.inconsistencia,
            "cor_regra": resultado.cor_regra,
            "motivos_regra": resultado.motivos_regra,
            "backend_usado": backend.name,
        }
    )

    return TriagemResponse(
        triagem_id=triagem_id,
        classificacao=resultado.classificacao,
        classificacao_llm=resultado.classificacao_llm,
        justificativa=resultado.justificativa,
        sinais_alerta=resultado.sinais_alerta,
        perguntas_adicionais=resultado.perguntas_adicionais,
        confianca=resultado.confianca,
        backend_usado=backend.name,
        cor_info=cor_info,
        inconsistencia=resultado.inconsistencia,
        cor_regra=resultado.cor_regra,
        motivos_regra=resultado.motivos_regra,
        erro=resultado.erro,
    )


# ---------------------------------------------------------------------------
# Fila viva
# ---------------------------------------------------------------------------
def _to_fila_item(row: dict) -> FilaItem:
    cor_info = CORES.get(
        row["classificacao"],
        {
            "tempo_max": -1,
            "descricao": "Classificação não reconhecida.",
            "hex_fundo": "#6C757D",
            "hex_texto": "#FFFFFF",
            "icone": "⚪",
        },
    )
    return FilaItem(
        id=row["id"],
        criado_em=row["criado_em"],
        enfermeiro=row["enfermeiro"],
        paciente_nome=row.get("paciente_nome"),
        idade=row.get("idade"),
        sexo=row.get("sexo"),
        pressao=row.get("pressao"),
        frequencia_cardiaca=row.get("frequencia_cardiaca"),
        spo2=row.get("spo2"),
        temperatura=row.get("temperatura"),
        sintomas=row.get("sintomas"),
        historico=row.get("historico"),
        classificacao=row["classificacao"],
        classificacao_llm=row.get("classificacao_llm"),
        justificativa=row.get("justificativa"),
        sinais_alerta=row.get("sinais_alerta") or [],
        perguntas_adicionais=row.get("perguntas_adicionais") or [],
        confianca=row.get("confianca"),
        inconsistencia=bool(row.get("inconsistencia")),
        cor_regra=row.get("cor_regra"),
        motivos_regra=row.get("motivos_regra") or [],
        backend_usado=row.get("backend_usado"),
        status=row["status"],
        tempo_max_min=int(cor_info.get("tempo_max", -1)),
        cor_info=cor_info,
    )


@app.get("/api/fila", response_model=list[FilaItem])
def listar_fila(incluir_finalizados: bool = False):
    """Lista pacientes na fila ordenados por gravidade e ordem de chegada."""
    rows = db.listar_fila(incluir_finalizados=incluir_finalizados)
    return [_to_fila_item(r) for r in rows]


@app.patch("/api/fila/{triagem_id}", response_model=FilaItem)
def atualizar_fila(triagem_id: int, req: FilaUpdateRequest):
    """Muda o status de uma triagem (atendido / dispensado / aguardando)."""
    ok = db.atualizar_status(triagem_id, req.status)
    if not ok:
        raise HTTPException(status_code=404, detail="Triagem não encontrada.")
    rows = [r for r in db.listar_fila(incluir_finalizados=True) if r["id"] == triagem_id]
    if not rows:
        raise HTTPException(status_code=404, detail="Triagem não encontrada após update.")
    return _to_fila_item(rows[0])


# ---------------------------------------------------------------------------
# Frontend estático
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Serve a interface principal."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Arquivo frontend/index.html não encontrado. "
                "Verifique a estrutura do projeto."
            ),
        )
    return FileResponse(index)


# Serve qualquer outro asset estático em /static (futuro: imagens, fontes locais, etc.)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
