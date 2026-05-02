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
    # ===== Sintomas / queixa principal =====
    "flag_dor_toracica":     [r"\bdor toracica\b", r"\bdor torácica\b", r"\bdor no peito\b", r"\bprecordial\b", r"\bretroesternal\b"],
    "flag_dispneia":         [r"\bdispneia\b", r"\bfalta de ar\b", r"\bdificuldade respirat", r"\btiragem\b", r"\bsibil"],
    "flag_alt_consciencia":  [r"\bglasgow\b", r"\binconsciente\b", r"\bsonolento\b", r"\brebaixamento\b", r"\bconfus", r"\bcoma\b", r"\btorpor"],
    "flag_convulsao":        [r"\bconvuls", r"\bcrise tonic", r"\bcrise tônico", r"\bstatus epilepticus\b", r"\bepileptic", r"\bepiléptic"],
    "flag_hemorragia":       [r"\bhemorragia\b", r"\bsangramento\b", r"\bhematemese\b", r"\bhematêmese\b", r"\bepistax", r"\bmelena\b", r"\benterorragia\b"],
    "flag_trauma_penetrante":[r"\barma de fogo\b", r"\bPAF\b", r"\barma branca\b", r"\bempalamento\b", r"\bferimento por\b"],
    "flag_trauma_outro":     [r"\btrauma\b", r"\bcontusao\b", r"\bcontusão\b", r"\bentorse\b", r"\bfratura\b", r"\bqueda\b", r"\batropel"],
    "flag_febre":            [r"\bfebre\b", r"\bfebril\b", r"\bhipertermia\b"],
    "flag_dor_abdominal":    [r"\bdor abdominal\b", r"\babdome\b", r"\bcolica\b", r"\bcólica\b", r"\bperitonite\b"],
    "flag_dor_cabeca":       [r"\bcefaleia\b", r"\bdor de cabeca\b", r"\bdor de cabeça\b"],
    "flag_dor_intensa":      [r"\beva (?:8|9|10)/10\b", r"\bintens", r"\bsevera\b", r"\bem trovao\b", r"\bem trovão\b", r"\bpior dor\b"],
    "flag_gestante":         [r"\bgestante\b", r"\bgestacao\b", r"\bgestação\b", r"\bgravida\b", r"\bgrávida\b"],
    "flag_anafilaxia":       [r"\banafila", r"\bedema de glote\b", r"\bestridor\b", r"\bbroncoespasmo\b"],
    "flag_pcr":              [r"\bparada cardiorrespiratori", r"\bparada cardiorrespiratór", r"\bPCR\b", r"\bRCP\b", r"\bassistolia\b"],
    "flag_sepse":            [r"\bsepse\b", r"\bchoque septico\b", r"\bchoque séptico\b", r"\binfeccao generalizada\b", r"\binfecção generalizada\b"],
    "flag_avc":              [r"\bAVC\b", r"\bhemiparesia\b", r"\bhemiplegia\b", r"\bdisartria\b", r"\bafasia\b", r"\bNIHSS\b"],
    "flag_iam":              [r"\bIAM\b", r"\binfarto\b", r"\bsupra de ST\b", r"\bSCA\b"],
    "flag_psiquiatrico":     [r"\bsuicid", r"\bautoexterminio\b", r"\bautoextermínio\b", r"\bsurto\b", r"\bpsicotic", r"\bpsicótic", r"\bagressividade\b", r"\bpanico\b", r"\bpânico\b"],
    "flag_vomitos":          [r"\bvomito", r"\bvômito", r"\bemese\b", r"\bêmese\b"],
    "flag_diarreia":         [r"\bdiarreia\b"],
    "flag_administrativo":   [r"\breceita\b", r"\batestado\b", r"\bdeclaracao\b", r"\bdeclaração\b", r"\bcurativo\b", r"\bvacin", r"\bresultado de exame\b", r"\baferir\b", r"\borientacao\b", r"\borientação\b", r"\bcomparece\b", r"\bencaminhamento\b", r"\brotina\b"],
    "flag_intoxicacao":      [r"\bintoxica", r"\benvenenament", r"\bingestao de medicament", r"\bingestão de medicament", r"\babuso de subst"],
    "flag_queimadura":       [r"\bqueimadura\b", r"\bqueimad", r"\bSCT\b"],
    # ===== Historico clinico =====
    "flag_imunossuprimido":  [r"\bimunossuprim", r"\bquimioterap", r"\bcorticoide cron", r"\bHIV\b", r"\btransplant", r"\bneutropenic", r"\bneutropênic"],
    "flag_cardiopata_hist":  [r"\bcardiopat", r"\bcoronariopat", r"\bIAM previo\b", r"\bIAM prévio\b", r"\binfarto previo\b", r"\binfarto prévio\b", r"\binsuficiencia cardiaca\b", r"\binsuficiência cardíaca\b", r"\bIC CF\b", r"\bFA\b", r"\bfibrilacao atrial\b", r"\bfibrilação atrial\b", r"\bmarca-passo\b"],
    "flag_diabetico_hist":   [r"\bdiabet", r"\bdiabét", r"\bDM[12]\b", r"\bDM\b", r"\binsulinodepend", r"\bglicemia"],
    "flag_hipertenso_hist":  [r"\bhipertens", r"\bHAS\b", r"\bPA elevada\b", r"\blosartana\b", r"\banlodipino\b", r"\bIECA\b"],
    "flag_renal_hist":       [r"\binsuficiencia renal\b", r"\binsuficiência renal\b", r"\bIRC\b", r"\bnefropat", r"\bhemodialitic", r"\bhemodialític", r"\btransplante renal\b"],
}


def parse_pa(pres):
    if not pres or "/" not in str(pres):
        return 0, 0
    try:
        sis, dia = str(pres).split("/", 1)
        return int(sis.strip()), int(dia.strip())
    except (ValueError, TypeError):
        return 0, 0


def extrair_flags(sintomas, historico):
    text = (str(sintomas or "") + " " + str(historico or "")).lower()
    return {
        col: int(any(re.search(p, text) for p in padroes))
        for col, padroes in TERMOS.items()
    }


def faixa_etaria(idade):
    try:
        i = int(idade or 0)
    except (ValueError, TypeError):
        return 1
    if i < 18:
        return 0
    if i >= 65:
        return 2
    return 1


def extrair_features_para_rf(dados, ordem_colunas):
    pa_sis, pa_dia = parse_pa(dados.get("pressao", ""))
    flags = extrair_flags(dados.get("sintomas", ""), dados.get("historico", ""))
    idade_int = int(dados.get("idade") or 0)
    fx = faixa_etaria(idade_int)

    valores = {
        "idade":                idade_int,
        "frequencia_cardiaca":  dados.get("frequencia_cardiaca") or 0,
        "spo2":                 dados.get("spo2") or 0,
        "temperatura":          dados.get("temperatura") or 0,
        "pa_sistolica":         pa_sis,
        "pa_diastolica":        pa_dia,
        "pulse_pressure":       max(0, pa_sis - pa_dia),
        "sexo_M":               int(dados.get("sexo") == "Masculino"),
        "sexo_F":               int(dados.get("sexo") == "Feminino"),
        "idade_faixa":          fx,
        "flag_pediatrico":      int(fx == 0),
        "flag_idoso":           int(fx == 2),
    }
    valores.update(flags)
    return [float(valores.get(col, 0)) for col in ordem_colunas]
