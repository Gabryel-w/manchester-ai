"""
Agente de Triagem Médica - núcleo da decisão.

Implementa abstração de backend LLM para alternar entre Groq (cloud) e
Ollama (local) com uma única troca de configuração. A saída é estruturada
em JSON para garantir parse robusto e explicabilidade (XAI) do raciocínio
clínico do modelo.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# System prompt — núcleo da explicabilidade (XAI)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um assistente de triagem médica treinado no Protocolo de Manchester (Manchester Triage System), o sistema oficial de classificação de risco adotado em UPAs e prontos-socorros do Brasil.

Sua tarefa: classificar a prioridade de atendimento de um paciente em UMA das 5 cores e justificar seu raciocínio clínico de forma clara para um enfermeiro triador.

CORES E TEMPO MÁXIMO DE ESPERA:
- VERMELHO (0 min): emergência, risco iminente de vida. Ex: parada cardiorrespiratória, choque, inconsciência, hemorragia maciça, sepse grave, IAM com instabilidade.
- LARANJA (10 min): muito urgente. Ex: dor torácica suspeita de SCA, dispneia severa, AVC suspeito, dor 8-10/10, alteração de consciência, hipotensão grave (PA sistólica < 90), SpO2 < 90%.
- AMARELO (60 min): urgente. Ex: dor moderada (4-7/10), febre alta (≥ 39°C) sem outros sinais graves, vômitos persistentes, dor abdominal moderada, dispneia leve.
- VERDE (120 min): pouco urgente. Ex: sintomas leves a moderados, dor 1-3/10, febre baixa sem outros sintomas, queixas crônicas agudizadas levemente.
- AZUL (240 min): não urgente. Ex: queixas administrativas, sintomas crônicos sem agudização, retorno para receita, queixas mínimas, sinais vitais normais.

REGRAS OBRIGATÓRIAS:
1. Sempre escolha UMA das 5 cores acima. Nunca invente cores intermediárias.
2. Trabalhe APENAS com os dados informados. Não invente sintomas que não foram descritos.
3. Se a descrição for ambígua, classifique de forma conservadora (cor mais grave) e formule perguntas adicionais que ajudariam a refinar.
4. Justifique sempre combinando sintomas E sinais vitais quando ambos forem informados.
5. Cite explicitamente o(s) fator(es) decisivo(s) que pesaram mais para a classificação escolhida.
6. Seja objetivo: a justificativa deve caber em 2 a 4 frases, em linguagem que um enfermeiro leigo entenda.

FORMATO DE SAÍDA OBRIGATÓRIO — apenas JSON puro, sem markdown, sem comentários, sem texto antes ou depois:
{
  "classificacao": "VERMELHO" | "LARANJA" | "AMARELO" | "VERDE" | "AZUL",
  "justificativa": "Explicação clínica em 2-4 frases, citando os fatores decisivos.",
  "sinais_alerta": ["lista de sinais de alerta identificados, ou lista vazia"],
  "perguntas_adicionais": ["perguntas que o triador deveria fazer para refinar a avaliação, ou lista vazia"],
  "confianca": "ALTA" | "MEDIA" | "BAIXA"
}
"""


# ---------------------------------------------------------------------------
# Resultado tipado da triagem
# ---------------------------------------------------------------------------
@dataclass
class TriagemResult:
    classificacao: str
    justificativa: str
    sinais_alerta: list = field(default_factory=list)
    perguntas_adicionais: list = field(default_factory=list)
    confianca: str = "N/A"
    raw_response: str = ""
    erro: str = ""

    @property
    def sucesso(self) -> bool:
        return self.classificacao in ("VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL")


# ---------------------------------------------------------------------------
# Backends LLM
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    """Interface comum para qualquer provedor de LLM."""

    @abstractmethod
    def chat(self, system: str, user: str) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class GroqBackend(LLMBackend):
    """Backend que usa a API gratuita da Groq (cloud, ultrarrápida)."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.startswith("cole_sua_chave"):
            raise RuntimeError(
                "GROQ_API_KEY não configurada. Crie um arquivo .env (copie de .env.example) "
                "e cole sua chave gratuita de https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class OllamaBackend(LLMBackend):
    """Backend que usa Ollama rodando local (ou em outra máquina da rede)."""

    def __init__(self, model: str = "mistral:7b"):
        import ollama

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)
        self.model = model
        self.host = host

    @property
    def name(self) -> str:
        return f"Ollama ({self.model}) @ {self.host}"

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": 0.2, "num_predict": 1024},
        )
        return resp["message"]["content"]


# ---------------------------------------------------------------------------
# Construção do prompt e parse da resposta
# ---------------------------------------------------------------------------
def build_prompt(dados: dict) -> str:
    """Formata os dados do paciente em um prompt estruturado para o LLM."""
    return f"""Avalie o seguinte paciente para triagem segundo o Protocolo de Manchester.

DADOS DO PACIENTE
- Idade: {dados.get('idade', 'não informada')} anos
- Sexo: {dados.get('sexo', 'não informado')}
- Pressão arterial: {dados.get('pressao', 'não medida')} mmHg
- Frequência cardíaca: {dados.get('frequencia_cardiaca', 'não medida')} bpm
- Saturação de oxigênio (SpO2): {dados.get('spo2', 'não medida')}%
- Temperatura corporal: {dados.get('temperatura', 'não medida')} °C

QUEIXA E SINTOMAS RELATADOS
{dados.get('sintomas', 'Não informado.')}

HISTÓRICO CLÍNICO RELEVANTE
{dados.get('historico') or 'Nenhum informado.'}

Classifique a prioridade segundo o Protocolo de Manchester e responda APENAS no formato JSON especificado nas suas instruções."""


def _parse_response(raw: str) -> dict:
    """Extrai o objeto JSON da resposta do LLM, tolerando markdown e texto extra."""
    text = raw.strip()

    # Remove fences markdown (```json ... ``` ou ``` ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    # Recorta do primeiro { ao último }, caso ainda haja texto fora do JSON
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def classificar(dados: dict, backend: LLMBackend) -> TriagemResult:
    """Executa a triagem completa: chama o LLM e devolve resultado estruturado."""
    user_prompt = build_prompt(dados)

    # Etapa 1: chamada ao LLM
    try:
        raw = backend.chat(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return TriagemResult(
            classificacao="ERRO",
            justificativa=(
                f"Falha ao contatar o backend ({backend.name}). "
                f"Verifique conexão / chave de API / se o Ollama está rodando.\n\n"
                f"Detalhe técnico: {e}"
            ),
            erro=str(e),
        )

    # Etapa 2: parse do JSON
    try:
        parsed = _parse_response(raw)
    except json.JSONDecodeError as e:
        return TriagemResult(
            classificacao="ERRO",
            justificativa=(
                "O modelo retornou uma resposta que não é JSON válido. "
                "Isso pode acontecer com modelos pequenos. Tente um modelo maior ou execute novamente."
            ),
            raw_response=raw,
            erro=f"JSONDecodeError: {e}",
        )

    # Etapa 3: normalização da classificação
    classificacao_raw = str(parsed.get("classificacao", "")).upper().strip()
    classificacao = "ERRO"
    for cor in ("VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL"):
        if cor in classificacao_raw:
            classificacao = cor
            break

    return TriagemResult(
        classificacao=classificacao,
        justificativa=parsed.get("justificativa", "Sem justificativa fornecida."),
        sinais_alerta=parsed.get("sinais_alerta") or [],
        perguntas_adicionais=parsed.get("perguntas_adicionais") or [],
        confianca=str(parsed.get("confianca", "N/A")).upper(),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Catálogo de modelos disponíveis e factory
# ---------------------------------------------------------------------------
MODELOS_GROQ = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]

MODELOS_OLLAMA = [
    "mistral:7b",
    "llama3.1:8b",
    "llama3.2:3b",
    "gemma2:2b",
    "phi4:14b",
    "deepseek-r1:8b",
]


def get_backend(tipo: str, modelo: str) -> LLMBackend:
    """Factory: instancia o backend correto a partir do tipo e modelo escolhidos."""
    if tipo == "Groq":
        return GroqBackend(model=modelo)
    if tipo == "Ollama":
        return OllamaBackend(model=modelo)
    raise ValueError(f"Backend desconhecido: {tipo}")
