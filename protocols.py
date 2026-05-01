"""
Protocolo de Manchester - Constantes de classificação de risco.

Referência:
    Sistema Manchester de Classificação de Risco (Manchester Triage System).
    Adotado oficialmente em diversas UPAs e prontos-socorros do Brasil.
"""

# Cores do Protocolo de Manchester com tempo máximo de espera (em minutos),
# descrição clínica, e cores hex para a interface.
CORES = {
    "VERMELHO": {
        "tempo_max": 0,
        "descricao": "Emergência — atendimento imediato. Risco iminente de vida.",
        "hex_fundo": "#FF4444",
        "hex_texto": "#FFFFFF",
        "icone": "🔴",
    },
    "LARANJA": {
        "tempo_max": 10,
        "descricao": "Muito urgente — atendimento em até 10 minutos.",
        "hex_fundo": "#FF8C00",
        "hex_texto": "#FFFFFF",
        "icone": "🟠",
    },
    "AMARELO": {
        "tempo_max": 60,
        "descricao": "Urgente — atendimento em até 60 minutos.",
        "hex_fundo": "#FFD700",
        "hex_texto": "#000000",
        "icone": "🟡",
    },
    "VERDE": {
        "tempo_max": 120,
        "descricao": "Pouco urgente — atendimento em até 120 minutos.",
        "hex_fundo": "#28A745",
        "hex_texto": "#FFFFFF",
        "icone": "🟢",
    },
    "AZUL": {
        "tempo_max": 240,
        "descricao": "Não urgente — atendimento em até 240 minutos.",
        "hex_fundo": "#17A2B8",
        "hex_texto": "#FFFFFF",
        "icone": "🔵",
    },
}

# Lista usada como referência pelo agente para identificar sinais de alerta
# que tipicamente elevam a prioridade. Não é exaustiva — o LLM pode identificar
# outros sinais a partir da descrição livre dos sintomas.
SINAIS_ALARME = [
    "dor no peito",
    "dor torácica",
    "falta de ar",
    "dispneia",
    "perda de consciência",
    "síncope",
    "convulsão",
    "paralisia",
    "fraqueza unilateral",
    "sangramento intenso",
    "hemorragia",
    "febre acima de 39.5",
    "hipertermia severa",
    "pressão muito baixa",
    "hipotensão severa",
    "saturação abaixo de 90",
    "cianose",
    "trauma grave",
    "queimadura extensa",
    "intoxicação",
    "dor abdominal intensa",
    "alteração súbita do estado mental",
]

# Faixas de referência para sinais vitais usados na avaliação.
# Servem como guia para o LLM justificar decisões.
FAIXAS_REFERENCIA = {
    "frequencia_cardiaca": {
        "normal": (60, 100),
        "alerta": (50, 120),
        "critico": "fora de 40-140 bpm",
    },
    "saturacao_oxigenio": {
        "normal": ">= 95",
        "alerta": "90-94",
        "critico": "< 90",
    },
    "temperatura": {
        "normal": (36.0, 37.5),
        "alerta": (37.6, 38.5),
        "critico": "> 38.5 ou < 35.0",
    },
    "pressao_sistolica": {
        "normal": (110, 130),
        "alerta": (90, 109),
        "critico": "< 90 ou > 180",
    },
}


def get_cor_info(classificacao: str) -> dict:
    """Retorna dict com info da cor, ou um dict default se classificação for inválida."""
    return CORES.get(
        classificacao.upper(),
        {
            "tempo_max": -1,
            "descricao": "Classificação não reconhecida.",
            "hex_fundo": "#6C757D",
            "hex_texto": "#FFFFFF",
            "icone": "⚪",
        },
    )
