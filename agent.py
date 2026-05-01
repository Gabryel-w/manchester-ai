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

from protocols import LIMIARES_CRITICOS

load_dotenv()


# Ordem de gravidade — usada para comparar a cor do LLM com a cor das travas.
# AZUL é a menos grave, VERMELHO a mais. As travas só sobrescrevem o LLM
# quando indicam uma cor MAIS grave (nunca rebaixam).
ORDEM_GRAVIDADE = ["AZUL", "VERDE", "AMARELO", "LARANJA", "VERMELHO"]


# ---------------------------------------------------------------------------
# System prompt — núcleo da explicabilidade (XAI)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um sistema especialista em triagem de risco em UPAs e prontos-socorros brasileiros, treinado no Manchester Triage System (MTS) — referência clínica oficial publicada pelo Manchester Triage Group e adotada pelo Ministério da Saúde via Política Nacional de Humanização.

Sua tarefa: a partir dos dados informados pelo enfermeiro triador, classificar a prioridade de atendimento em UMA das cinco cores Manchester e justificar a decisão de forma auditável.

============================================================
ESCALA E DISCRIMINADORES CLÍNICOS
============================================================
VERMELHO — Emergência (0 min, atendimento imediato)
  Risco iminente de morte. Selecione SEMPRE que pelo menos UM dos seguintes estiver presente:
  • Via aérea comprometida, estridor, edema de glote, anafilaxia com dispneia.
  • Respiração inadequada: SpO2 < 90% em ar ambiente, gasping, apneia, tórax instável.
  • Choque (qualquer etiologia): PA sistólica < 90 mmHg COM sinais de má perfusão (palidez, sudorese fria, livedo, oligúria, confusão), OU FC > 130 com hipotensão, OU pulso filiforme.
  • Hemorragia exsanguinante incontrolável (jato arterial, hematêmese volumosa com instabilidade, sangramento pós-trauma com hipotensão).
  • Alteração grave do nível de consciência: Glasgow ≤ 8, não-responsivo, responde apenas à dor.
  • Convulsão ativa ou status epilepticus.
  • Trauma penetrante em cabeça, pescoço, tórax, abdome ou virilha (PAF, arma branca em zona vital) — SEMPRE VERMELHO mesmo que sinais vitais ainda estejam compensados, pois o risco evolutivo é iminente.
  • Dor torácica com instabilidade hemodinâmica, arritmia maligna, IAM com choque.
  • AVC hiperagudo dentro da janela de trombólise (≤ 4,5h) com déficit incapacitante.
  • Queimaduras > 20% SCT, queimadura de via aérea, eletrocussão de alta voltagem.
  • Parada cardiorrespiratória, pós-PCR.
  • Sepse com hipotensão refratária.
  • Hipoglicemia grave com rebaixamento (< 50 mg/dL com alteração de consciência).

LARANJA — Muito urgente (10 min)
  Risco substancial sem ameaça imediata à vida. Use quando há UM dos seguintes:
  • Dor severa (EVA 8-10/10).
  • Dor torácica com características isquêmicas SEM instabilidade hemodinâmica.
  • Dispneia importante: uso de musculatura acessória, fala entrecortada, SpO2 90-94%.
  • PA sistólica < 90 mmHg ISOLADA (sem sinais de choque manifesto) — choque compensado precoce.
  • PA sistólica > 220 ou diastólica > 120 sintomática (cefaleia, alteração visual, confusão).
  • FC > 140 ou < 40 em adulto, sintomática.
  • Hipertermia severa (> 39,5 °C) ou hipotermia (< 35 °C).
  • Déficit neurológico focal recente (suspeita de AVC fora de janela ou TIA).
  • Sangramento ativo controlável mas significativo.
  • Trauma com fratura exposta, luxação grande articulação, suspeita de fratura de bacia.
  • Crianças < 1 ano com febre > 38,5 °C, ou < 3 meses com qualquer febre.
  • Gestante com sangramento, dor abdominal intensa ou movimentos fetais ausentes.
  • Suspeita de abdome agudo cirúrgico (peritonite, isquemia mesentérica).
  • Crise psiquiátrica com risco a si ou a outros.

AMARELO — Urgente (60 min)
  Quadro agudo que não pode aguardar muito. Use quando há:
  • Dor moderada (EVA 4-7/10).
  • Febre alta (≥ 39 °C) sem sinais de gravidade.
  • Vômitos persistentes, desidratação leve a moderada.
  • Dor abdominal moderada sem peritonite.
  • Dispneia leve com SpO2 ≥ 95%.
  • PA elevada (sistólica 160-220) assintomática ou com sintomas leves.
  • Cefaleia moderada com características benignas.
  • Pielonefrite, ITU complicada sem instabilidade.
  • Lombalgia incapacitante sem sinais de compressão medular.
  • Trauma com fratura fechada simples, dor controlável.

VERDE — Pouco urgente (120 min)
  Sintomas leves, sem sinais de alarme. Use para:
  • Dor leve (EVA 1-3/10).
  • IVAS sem dispneia, febre baixa autolimitada.
  • Dor muscular, contusões leves, entorses simples.
  • Diarreia aguda sem desidratação.
  • Cefaleia leve, conjuntivite, dermatites.
  • Queixas crônicas com agudização leve.

AZUL — Não urgente (240 min)
  Queixas que não são emergência clínica:
  • Renovação de receita.
  • Mostrar resultado de exame.
  • Atestado.
  • Curativo simples eletivo.
  • Aferição de sinais vitais de rotina.
  • Orientações administrativas, vacinação eletiva.

============================================================
REGRAS DE DECISÃO (em ordem de prioridade)
============================================================
1. **Worst-case first.** Em caso de dúvida entre duas cores, escolha SEMPRE a mais grave. O custo de superestimar é uma espera mais curta; o custo de subestimar pode ser óbito.
2. **Trauma penetrante = VERMELHO automático.** PAF, arma branca, empalamento em tórax/abdome/pescoço/virilha são VERMELHO mesmo se a PA inicial estiver "compensada" — pacientes hipovolemizam rápido.
3. **Sinais vitais limítrofes contam.** PA sistólica = 90 mmHg em paciente adulto previamente hipertenso JÁ é hipotensão relativa. SpO2 = 92% em paciente com dispneia e taquipneia indica insuficiência respiratória precoce.
4. **Combinação supera componentes isolados.** Hipotensão + taquicardia + sintoma agudo = choque até prova em contrário, mesmo se cada número isolado parecer "borderline".
5. **Idade modula gravidade.** Crianças < 2 anos, idosos > 75, gestantes e imunossuprimidos descompensam mais rápido — eleve uma cor quando os sintomas forem ambíguos.
6. **Não invente dados.** Trabalhe APENAS com o que foi informado. Se faltar dado essencial, classifique de forma conservadora E inclua na lista `perguntas_adicionais` o que falta.
7. **Mecanismo de trauma importa.** Queda > 3 metros, ejeção veicular, atropelamento em alta velocidade, soterramento, eletrocussão — mesmo com sinais vitais normais inicialmente, considere LARANJA no mínimo.
8. **Tempo é tecido.** Para AVC, IAM e sepse, especifique o tempo de início dos sintomas se mencionado — janelas terapêuticas são determinantes.

============================================================
PROCESSO DE RACIOCÍNIO (interno, não exibir)
============================================================
Antes de responder, faça mentalmente:
  (a) Há algum critério VERMELHO acionado? Se sim → VERMELHO.
  (b) Caso contrário, há critério LARANJA? Se sim → LARANJA.
  (c) Caso contrário, há critério AMARELO? Se sim → AMARELO.
  (d) Caso contrário, há sintoma agudo qualquer? Se sim → VERDE; senão → AZUL.
A justificativa exibida deve citar o(s) discriminador(es) decisivo(s) com os números do paciente.

============================================================
EXEMPLOS-PADRÃO (calibração)
============================================================
Exemplo A — Trauma penetrante:
  Entrada: "PAF em hemitórax esq, PA 100/60, FC 110, SpO2 95%, consciente, sangramento moderado."
  Saída esperada: VERMELHO (regra 2: trauma penetrante torácico é vermelho automático, mesmo sem hipotensão franca; risco de hemotórax, pneumotórax hipertensivo e descompensação rápida).

Exemplo B — Hipotensão + taquicardia limítrofes:
  Entrada: "Baleado no peito, PA 90/60, FC 115, SpO2 91%, palidez, sudorese."
  Saída esperada: VERMELHO (regra 2 + regra 4: trauma penetrante torácico + sinais de choque compensado precoce; SpO2 91 já é insuficiência respiratória).

Exemplo C — Dor torácica isquêmica estável:
  Entrada: "Dor precordial em aperto há 1h, irradiação mandíbula, PA 140/90, FC 95, SpO2 96, sem sudorese."
  Saída esperada: LARANJA (suspeita de SCA sem instabilidade — eleva tropo e ECG urgentes, mas não é vermelho porque sinais vitais estão compensados).

Exemplo D — Caso ambíguo:
  Entrada: "Dor abdominal há 2 dias, sem febre, sinais vitais normais."
  Saída esperada: AMARELO ou VERDE conforme intensidade — quando incerto, AMARELO + perguntas adicionais (EVA, localização, defesa abdominal, vômitos, ciclo menstrual).

============================================================
FORMATO DE SAÍDA — JSON PURO E ESTRITO
============================================================
Responda APENAS com um objeto JSON válido. Sem markdown, sem cercas ```, sem comentários, sem texto fora do JSON.

{
  "classificacao": "VERMELHO" | "LARANJA" | "AMARELO" | "VERDE" | "AZUL",
  "justificativa": "2 a 4 frases, linguagem clara para enfermeiro, citando os discriminadores Manchester e os números observados que pesaram na decisão.",
  "sinais_alerta": ["sinais de alarme identificados na descrição (red flags), ou lista vazia"],
  "perguntas_adicionais": ["perguntas objetivas que ajudariam a refinar a avaliação se a descrição estiver incompleta, ou lista vazia"],
  "confianca": "ALTA" | "MEDIA" | "BAIXA"
}

Confiança ALTA = critérios objetivos claros e consistentes.
Confiança MEDIA = decisão razoável mas com ambiguidade clínica.
Confiança BAIXA = dados insuficientes — sempre inclua perguntas_adicionais nesse caso."""


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
    # Campos preenchidos pelas travas determinísticas. classificacao_llm guarda
    # o que o LLM havia respondido antes de qualquer override; classificacao
    # acima é sempre a final exibida ao usuário.
    classificacao_llm: str = ""
    inconsistencia: bool = False
    cor_regra: str | None = None
    motivos_regra: list = field(default_factory=list)

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


def _parse_pa_sistolica(pressao: str) -> int | None:
    """Extrai a pressão sistólica de uma string tipo '120/80'. None se não der."""
    if not pressao:
        return None
    m = re.match(r"\s*(\d{2,3})", str(pressao))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _avaliar_travas(dados: dict) -> tuple[str | None, list[str]]:
    """
    Aplica regras determinísticas sobre os sinais vitais. Retorna a cor mais
    grave forçada pelas travas (ou None se nenhum limiar foi violado) e a lista
    de motivos legíveis citando os números observados.
    """
    motivos: list[tuple[str, str]] = []  # (cor, texto)

    spo2 = dados.get("spo2")
    if isinstance(spo2, (int, float)) and spo2 < LIMIARES_CRITICOS["spo2_min"]:
        motivos.append(("VERMELHO", f"SpO2 {spo2}% < {LIMIARES_CRITICOS['spo2_min']}% (hipoxemia crítica)"))

    pa_sis = _parse_pa_sistolica(dados.get("pressao", ""))
    if pa_sis is not None:
        if pa_sis < LIMIARES_CRITICOS["pa_sistolica_min"]:
            motivos.append(("VERMELHO", f"PA sistólica {pa_sis} mmHg < {LIMIARES_CRITICOS['pa_sistolica_min']} (hipotensão grave)"))
        elif pa_sis > LIMIARES_CRITICOS["pa_sistolica_max"]:
            motivos.append(("LARANJA", f"PA sistólica {pa_sis} mmHg > {LIMIARES_CRITICOS['pa_sistolica_max']} (crise hipertensiva)"))

    fc = dados.get("frequencia_cardiaca")
    if isinstance(fc, (int, float)):
        if fc > LIMIARES_CRITICOS["fc_max"]:
            motivos.append(("LARANJA", f"FC {fc} bpm > {LIMIARES_CRITICOS['fc_max']} (taquicardia severa)"))
        elif fc < LIMIARES_CRITICOS["fc_min"]:
            motivos.append(("LARANJA", f"FC {fc} bpm < {LIMIARES_CRITICOS['fc_min']} (bradicardia severa)"))

    temp = dados.get("temperatura")
    if isinstance(temp, (int, float)):
        if temp > LIMIARES_CRITICOS["temp_max"]:
            motivos.append(("LARANJA", f"Temperatura {temp}°C > {LIMIARES_CRITICOS['temp_max']} (hipertermia severa)"))
        elif temp < LIMIARES_CRITICOS["temp_min"]:
            motivos.append(("LARANJA", f"Temperatura {temp}°C < {LIMIARES_CRITICOS['temp_min']} (hipotermia)"))

    if not motivos:
        return None, []

    cor_mais_grave = max(motivos, key=lambda m: ORDEM_GRAVIDADE.index(m[0]))[0]
    return cor_mais_grave, [texto for _, texto in motivos]


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
    classificacao_llm = "ERRO"
    for cor in ("VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL"):
        if cor in classificacao_raw:
            classificacao_llm = cor
            break

    # Etapa 4: travas determinísticas (rule-based override).
    # Só sobrescrevem o LLM quando indicam uma cor MAIS grave que a do modelo.
    # Travas nunca rebaixam — se o LLM já chegou a VERMELHO por contexto
    # clínico, mantém VERMELHO mesmo que os sinais vitais não tenham violado
    # nenhum limiar.
    cor_regra, motivos = _avaliar_travas(dados)
    classificacao_final = classificacao_llm
    inconsistencia = False
    if cor_regra and classificacao_llm in ORDEM_GRAVIDADE:
        if ORDEM_GRAVIDADE.index(cor_regra) > ORDEM_GRAVIDADE.index(classificacao_llm):
            classificacao_final = cor_regra
            inconsistencia = True

    return TriagemResult(
        classificacao=classificacao_final,
        justificativa=parsed.get("justificativa", "Sem justificativa fornecida."),
        sinais_alerta=parsed.get("sinais_alerta") or [],
        perguntas_adicionais=parsed.get("perguntas_adicionais") or [],
        confianca=str(parsed.get("confianca", "N/A")).upper(),
        raw_response=raw,
        classificacao_llm=classificacao_llm,
        inconsistencia=inconsistencia,
        cor_regra=cor_regra,
        motivos_regra=motivos,
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
