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

from agent import (
    MODELOS_GROQ,
    MODELOS_OLLAMA,
    classificar,
    get_backend,
)
from protocols import CORES

load_dotenv()

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


class TriagemResponse(BaseModel):
    classificacao: str
    justificativa: str
    sinais_alerta: list[str]
    perguntas_adicionais: list[str]
    confianca: str
    backend_usado: str
    cor_info: dict
    erro: str = ""


class ModelosResponse(BaseModel):
    groq: list[str]
    ollama: list[str]
    cores: dict


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

    return TriagemResponse(
        classificacao=resultado.classificacao,
        justificativa=resultado.justificativa,
        sinais_alerta=resultado.sinais_alerta,
        perguntas_adicionais=resultado.perguntas_adicionais,
        confianca=resultado.confianca,
        backend_usado=backend.name,
        cor_info=cor_info,
        erro=resultado.erro,
    )


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
