"""
Extracao de features clinicas a partir dos dados brutos do paciente.

Este modulo e a fonte da verdade para qual representacao numerica/binaria
e fornecida ao classificador Random Forest. Tem que ficar SINCRONIZADO com
data/gerar_dataset.py - se alterar regex aqui, regenere o dataset e
retreine o modelo, senao a inferencia em producao usara features
diferentes do treino e a precisao despenca.

Uso:
    from features import extrair_features_para_rf
    vetor = extrair_features_para_rf(dados, lista_features)
"""
from __future__ import annotations

import re

# Mesmo dicionario regex de data/gerar_dataset.py.
# IMPORTANTE: manter sincronizado.
TERMOS = {
    "flag_dor_toracica":     [r"\bdor torácica\b", r"\bdor no peito\b", r"\bprecordial\b", r"\bretroesternal\b"],
    "flag_dispneia":         [r"\bdispneia\b", r"\bfalta de ar\b", r"\bdificuldade respirat", r"\btiragem\b", r"\bsibil"],
    "flag_alt_consciencia":  [r"\bglasgow\b", r"\binconsciente\b", r"\bsonolento\b", r"\brebaixamento\b", r"\bconfus", r"\bcoma\b"],
    "flag_convulsao":        [r"\bconvuls", r"\bcrise tônico\b", r"\bstatus epilepticus\b", r"\bepiléptic"],
    "flag_hemorragia":       [r"\bhemorragia\b", r"\bsangramento\b", r"\bhematêmese\b", r"\bepistax"],
    "flag_trauma_penetrante":[r"\barma de fogo\b", r"\bPAF\b", r"\barma branca\b", r"\bempalamento\b", r"\bferimento por\b"],
    "flag_trauma_outro":     [r"\btrauma\b", r"\bcontusão\b", r"\bentorse\b", r"\bfratura\b", r"\bqueda\b"],
    "flag_febre":            [r"\bfebre\b", r"\bfebril\b", r"\bhipertermia\b"],
    "flag_dor_abdominal":    [r"\bdor abdominal\b", r"\babdome\b", r"\bcólica\b", r"\bperitonite\b"],
    "flag_dor_cabeca":       [r"\bcefaleia\b", r"\bdor de cabeça\b"],
    "flag_dor_intensa":      [r"\beva (?:8|9|10)/10\b", r"\bintens", r"\bsevera\b", r"\bem trovão\b", r"\bpior dor\b"],
    "flag_gestante":         [r"\bgestante\b", r"\bgestação\b", r"\bgrávida\b"],
    "flag_anafilaxia":       [r"\banafila", r"\bedema de glote\b", r"\bestridor\b", r"\bbroncoespasmo\b"],
    "flag_pcr":              [r"\bparada cardiorrespiratór", r"\bPCR\b", r"\bRCP\b", r"\bassistolia\b"],
    "flag_sepse":            [r"\bsepse\b", r"\bchoque séptico\b", r"\binfecção generalizada\b"],
    "flag_avc":              [r"\bAVC\b", r"\bhemiparesia\b", r"\bhemiplegia\b", r"\bdisartria\b", r"\bafasia\b", r"\bNIHSS\b"],
    "flag_iam":              [r"\bIAM\b", r"\binfarto\b", r"\bsupra de ST\b", r"\bSCA\b"],
    "flag_psiquiatrico":     [r"\bsuicid", r"\bautoextermínio\b", r"\bsurto\b", r"\bpsicótic", r"\bagressividade\b", r"\bpânico\b"],
    "flag_vomitos":          [r"\bvômito", r"\bêmese\b"],
    "flag_diarreia":         [r"\bdiarreia\b"],
    "flag_administrativo":   [r"\breceita\b", r"\batestado\b", r"\bdeclaração\b", r"\bcurativo\b", r"\bvacin", r"\bresultado de exame\b", r"\baferir\b", r"\borientação\b", r"\bcomparece\b"],
}


def parse_pa(pres: str) -> tuple[int, int]:
    """Extrai (sistolica, diastolica) de uma string '120/80'. Retorna (0, 0) se invalido."""
    if not pres or "/" not in str(pres):
        return 0, 0
    try:
        sis, dia = str(pres).split("/", 1)
        return int(sis.strip()), int(dia.strip())
    except (ValueError, TypeError):
        return 0, 0


def extrair_flags(sintomas: str, historico: str) -> dict[str, int]:
    """Aplica os regex de TERMOS na concatenacao sintomas+historico."""
    text = f"{sintomas or ''} {historico or ''}".lower()
    return {
        col: int(any(re.search(p, text) for p in padroes))
        for col, padroes in TERMOS.items()
    }


def extrair_features_para_rf(dados: dict, ordem_colunas: list[str]) -> list[float]:
    """
    Constroi o vetor de features na MESMA ordem usada no treino, lendo
    do dict de dados do paciente.

    Args:
        dados: dict com chaves idade, sexo, pressao, frequencia_cardiaca,
               spo2, temperatura, sintomas, historico.
        ordem_colunas: lista das colunas na ordem que o RF espera (vem
                       persistida junto com o pickle do modelo).
    """
    pa_sis, pa_dia = parse_pa(dados.get("pressao", ""))
    flags = extrair_flags(dados.get("sintomas", ""), dados.get("historico", ""))

    # Mapa unificado das features disponiveis.
    valores = {
        "idade":                dados.get("idade") or 0,
        "frequencia_cardiaca":  dados.get("frequencia_cardiaca") or 0,
        "spo2":                 dados.get("spo2") or 0,
        "temperatura":          dados.get("temperatura") or 0,
        "pa_sistolica":         pa_sis,
        "pa_diastolica":        pa_dia,
        "sexo_M":               int(dados.get("sexo") == "Masculino"),
        "sexo_F":               int(dados.get("sexo") == "Feminino"),
        **flags,
    }

    return [float(valores.get(col, 0)) for col in ordem_colunas]
