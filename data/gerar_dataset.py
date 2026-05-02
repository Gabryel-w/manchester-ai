"""
Gerador de dataset sintético para treino do classificador Random Forest
de triagem Manchester.

Filosofia:
  - Cada cor Manchester tem múltiplos cenários clínicos típicos.
  - Cada cenário gera dezenas de variações com sinais vitais coerentes,
    histórico plausível e descrição livre dos sintomas variada.
  - As features pré-extraídas (flags binárias) são derivadas
    automaticamente do texto + sinais vitais — assim o dataset fica
    pronto para treinar um RandomForestClassifier sem preprocessing.

Saída: data/triagem_dataset.csv com ~2000 linhas e 30+ colunas.

Uso:
    python data/gerar_dataset.py
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Callable

random.seed(42)  # reprodutibilidade

# ---------------------------------------------------------------------------
# Helpers numéricos
# ---------------------------------------------------------------------------
def pa(sis_min: int, sis_max: int, dia_offset: tuple[int, int] = (-50, -30)) -> tuple[int, int, str]:
    sis = random.randint(sis_min, sis_max)
    dia = max(40, min(120, sis + random.randint(*dia_offset)))
    return sis, dia, f"{sis}/{dia}"

def fc(low: int, high: int) -> int:
    return random.randint(low, high)

def spo2(low: int, high: int) -> int:
    return random.randint(low, high)

def temp(low: float, high: float) -> float:
    return round(random.uniform(low, high), 1)

def pick(*opts: str) -> str:
    return random.choice(opts)

def maybe(p: float) -> bool:
    return random.random() < p


# ---------------------------------------------------------------------------
# Cenários — VERMELHO (Emergência, 0 min)
# ---------------------------------------------------------------------------
def _pcr() -> dict:
    sis, _, pres = pa(50, 80)
    return dict(
        idade=random.randint(45, 90),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(20, 50),
        spo2=spo2(60, 82),
        temperatura=temp(34.5, 36.5),
        sintomas=pick(
            "Paciente em parada cardiorrespiratória, sem pulso central, sem respiração espontânea. RCP em curso desde a chegada.",
            "PCR presenciada há 5 minutos, em assistolia ao monitor. Família relata colapso súbito durante refeição.",
            "Encontrado inconsciente pela família, sem pulso e sem respiração. RCP iniciada por leigo, equipe assumiu na ambulância.",
            "Parada cardiorrespiratória após desconforto torácico de 30 minutos. RCP avançada em curso, intubado.",
        ),
        historico=pick(
            "HAS, DM2, IAM prévio há 2 anos, em uso de AAS, atorvastatina, losartana.",
            "Cardiopata isquêmico, fibrilação atrial crônica, anticoagulado.",
            "Sem comorbidades conhecidas. Tabagista 40 maços/ano.",
            "Insuficiência cardíaca CF III, ex-tabagista.",
        ),
        classificacao="VERMELHO",
    )


def _choque_septico() -> dict:
    sis, _, pres = pa(60, 88)
    return dict(
        idade=random.randint(50, 90),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(120, 160),
        spo2=spo2(82, 92),
        temperatura=temp(38.5, 40.5) if maybe(0.7) else temp(34.5, 35.5),
        sintomas=pick(
            "Paciente febril há 3 dias, agora sonolento, com extremidades frias e marmóreas. PA persistentemente baixa, taquicárdico, oligúrico nas últimas horas.",
            "Sepse de foco urinário, hipotensão refratária a hidratação, confuso, livedo em membros inferiores.",
            "Quadro de pneumonia evoluindo com hipotensão, taquicardia, alteração do nível de consciência. Sudorese fria.",
            "Foco abdominal de sepse, peritonite difusa, choque séptico instalado, anúrico.",
        ),
        historico=pick(
            "DM2 descompensado, ITU de repetição, em quimioterapia para neoplasia de mama.",
            "Imunossuprimido por uso crônico de corticoide.",
            "Cirrose hepática Child C, ascite refratária.",
            "Diabético, nefropata, hemodialítico.",
        ),
        classificacao="VERMELHO",
    )


def _iam_choque() -> dict:
    sis, _, pres = pa(60, 88)
    return dict(
        idade=random.randint(55, 88),
        sexo=pick("Masculino", "Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(110, 150) if maybe(0.6) else fc(35, 50),
        spo2=spo2(82, 92),
        temperatura=temp(35.8, 36.8),
        sintomas=pick(
            "Dor torácica retroesternal intensa irradiando para o braço esquerdo iniciada há 40 minutos. Sudorese fria profusa, palidez, hipotensão. Refere náusea e vômito. Paciente apreensivo.",
            "IAM com supra de ST anterior extenso, choque cardiogênico, dispneia importante e PA persistentemente baixa.",
            "Dor precordial em aperto, irradiando para mandíbula, com hipotensão e bradicardia. Suspeita de IAM inferior com bloqueio AV.",
            "Síncope após dor torácica intensa, recuperou com hipotensão mantida. Pulsos periféricos finos, sudoreico.",
        ),
        historico=pick(
            "HAS, DM2, dislipidemia, tabagista 30 maços/ano. Pai faleceu de IAM aos 58 anos.",
            "Coronariopata conhecido, angioplastia há 4 anos, em dupla antiagregação.",
            "Hipertenso, obeso (IMC 34), sedentário.",
            "Sem comorbidades conhecidas. Tabagista pesado.",
        ),
        classificacao="VERMELHO",
    )


def _avc_hiperagudo() -> dict:
    sis, _, pres = pa(150, 220)
    return dict(
        idade=random.randint(55, 90),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 110),
        spo2=spo2(92, 98),
        temperatura=temp(36.0, 37.5),
        sintomas=pick(
            "Hemiparesia direita súbita iniciada há 90 minutos, disartria, desvio de comissura labial. Glasgow 13. NIHSS estimado 12. Dentro de janela de trombólise.",
            "Afasia de expressão e hemiplegia esquerda iniciadas há 2 horas, NIHSS 18, paciente alerta. Suspeita de AVCi de ACM direita.",
            "AVC iniciado há 3 horas com plegia em dimídio direito, desvio de olhar conjugado, NIHSS 20. Em janela para trombectomia.",
            "Paciente acordou com hemiparesia e afasia há 1 hora, sem recuperação. Glasgow 14, déficit incapacitante.",
        ),
        historico=pick(
            "HAS de longa data, FA paroxística sem anticoagulação, dislipidemia.",
            "Diabético, hipertenso, AVC isquêmico prévio sem sequelas.",
            "Anticoagulado por FA crônica com varfarina, INR não controlado.",
            "Cardiopata isquêmico, em dupla antiagregação.",
        ),
        classificacao="VERMELHO",
    )


def _trauma_penetrante() -> dict:
    sis, _, pres = pa(80, 130)
    return dict(
        idade=random.randint(15, 55),
        sexo=pick("Masculino", "Masculino", "Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(90, 140),
        spo2=spo2(88, 98),
        temperatura=temp(35.8, 36.8),
        sintomas=pick(
            "Ferimento por arma de fogo em hemitórax esquerdo há 20 minutos. Em uso de máscara não reinalante. Hipotensão progressiva, taquicárdico, palidez cutâneo-mucosa.",
            "Ferimento por arma branca em região epigástrica, abdome em tábua, sinais de irritação peritoneal. Estável hemodinamicamente no momento.",
            "PAF em região cervical, sangramento moderado controlado com compressão. Vias aéreas pérvias, alerta.",
            "Empalamento por barra de ferro em hemitórax direito após acidente de construção. Paciente alerta, dispneico.",
        ),
        historico=pick(
            "Sem comorbidades conhecidas.",
            "Etilista crônico, sem outras comorbidades.",
            "Tabagista, ex-usuário de drogas ilícitas.",
            "Hipertenso em uso irregular de medicação.",
        ),
        classificacao="VERMELHO",
    )


def _convulsao_ativa() -> dict:
    sis, _, pres = pa(110, 160)
    return dict(
        idade=random.randint(2, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(110, 150),
        spo2=spo2(85, 94),
        temperatura=temp(36.5, 39.5),
        sintomas=pick(
            "Status epilepticus, crise tônico-clônica generalizada há 8 minutos sem recuperação de consciência entre crises. Cianose perioral, sialorreia.",
            "Crise convulsiva presenciada, durou 10 minutos, paciente em pós-ictal profundo, ainda não responde aos chamados.",
            "Convulsões repetidas há 30 minutos, três episódios sem recuperar consciência. Glasgow 7 atual.",
            "Crise focal evoluindo para generalização, tremores em hemicorpo esquerdo de início, agora generalizado.",
        ),
        historico=pick(
            "Epilepsia conhecida em uso irregular de fenitoína.",
            "Sem antecedentes neurológicos. Primeira crise da vida.",
            "Etilista crônico em abstinência há 48h.",
            "Tumor cerebral em investigação, em uso de dexametasona.",
        ),
        classificacao="VERMELHO",
    )


def _anafilaxia() -> dict:
    sis, _, pres = pa(70, 95)
    return dict(
        idade=random.randint(5, 70),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(120, 160),
        spo2=spo2(85, 94),
        temperatura=temp(36.0, 37.0),
        sintomas=pick(
            "Anafilaxia após ingestão de amendoim há 15 minutos. Edema de glote, estridor laríngeo, dispneia importante, urticária generalizada, hipotensão.",
            "Quadro de choque anafilático após picada de inseto, edema facial, broncoespasmo grave, hipotensão refratária.",
            "Reação anafilática medicamentosa (cefalosporina), edema de orofaringe, estridor, sibilância difusa, hipotensão.",
            "Anafilaxia idiopática com edema facial e cervical importantes, dificuldade respiratória progressiva, sibilos.",
        ),
        historico=pick(
            "Atopia conhecida, asma desde a infância.",
            "Sem comorbidades. Primeira reação alérgica documentada.",
            "Alergia previamente diagnosticada a penicilina e contraste iodado.",
            "Asmática em uso irregular de budesonida.",
        ),
        classificacao="VERMELHO",
    )


def _alteracao_consciencia_grave() -> dict:
    sis, _, pres = pa(85, 200)
    return dict(
        idade=random.randint(40, 90),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(50, 130),
        spo2=spo2(85, 95),
        temperatura=temp(35.5, 38.5),
        sintomas=pick(
            "Encontrado inconsciente em casa pela família, Glasgow 6, anisocoria à direita. Suspeita de AVC hemorrágico ou TCE.",
            "Rebaixamento progressivo do nível de consciência ao longo de 3 horas, agora Glasgow 7, não responde a estímulos.",
            "Coma de etiologia a esclarecer, Glasgow 5, postura de descerebração ao estímulo doloroso.",
            "Hipoglicemia grave (HGT 28) em paciente diabético, inconsciente, sudorese fria abundante.",
        ),
        historico=pick(
            "HAS grave em uso irregular de medicação. AVC prévio.",
            "DM2 insulinodependente, episódios prévios de hipoglicemia.",
            "Etilista crônico, hepatopata.",
            "Trauma craniano há 4 horas após queda da própria altura, em uso de varfarina.",
        ),
        classificacao="VERMELHO",
    )


# ---------------------------------------------------------------------------
# Cenários — LARANJA (Muito urgente, 10 min)
# ---------------------------------------------------------------------------
def _dor_toracica_estavel() -> dict:
    sis, _, pres = pa(110, 160)
    return dict(
        idade=random.randint(45, 80),
        sexo=pick("Masculino", "Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 110),
        spo2=spo2(94, 98),
        temperatura=temp(36.2, 37.0),
        sintomas=pick(
            "Dor torácica em aperto há 2 horas, irradiando para o braço esquerdo, EVA 7/10. Sem dispneia importante, sem sudorese profusa. Hemodinamicamente estável.",
            "Dor precordial intensa há 1 hora, com irradiação para mandíbula, sem instabilidade. ECG aguardando.",
            "Desconforto retroesternal de início súbito após esforço, dor 6/10, melhorou parcialmente com repouso. Suspeita de SCA.",
            "Dor torácica atípica, EVA 8/10, mas sem alterações hemodinâmicas. Paciente ansiosa, sem fatores de risco maiores.",
        ),
        historico=pick(
            "HAS, DM2, dislipidemia, ex-tabagista. Pai com IAM precoce.",
            "Hipertenso em uso de losartana, dislipidêmico.",
            "Sem fatores de risco conhecidos.",
            "Coronariopata, angioplastia há 6 anos, em uso de AAS e estatina.",
        ),
        classificacao="LARANJA",
    )


def _dispneia_moderada() -> dict:
    sis, _, pres = pa(100, 160)
    return dict(
        idade=random.randint(50, 90),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 130),
        spo2=spo2(90, 94),
        temperatura=temp(36.5, 38.5),
        sintomas=pick(
            "Dispneia progressiva há 2 dias, hoje fala entrecortada, uso de musculatura acessória. Sibilos difusos à ausculta. Saturação caindo.",
            "Falta de ar há 4 horas em paciente com DPOC, tiragem intercostal, conseguindo formular frases curtas.",
            "Dispneia importante, taquipneico (28 irpm), saturação 92%, edema de MMII bilateral. Suspeita de IC descompensada.",
            "Crise asmática moderada, sibilância audível à distância, uso de musculatura acessória, saturação 91%.",
        ),
        historico=pick(
            "DPOC GOLD III em uso de tiotrópio e formoterol. Tabagista 50 maços/ano.",
            "Asma desde a infância, várias internações prévias.",
            "Insuficiência cardíaca CF II em uso de carvedilol, espironolactona, furosemida.",
            "Hipertenso, ex-tabagista.",
        ),
        classificacao="LARANJA",
    )


def _dor_severa() -> dict:
    sis, _, pres = pa(115, 165)
    return dict(
        idade=random.randint(20, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 125),
        spo2=spo2(95, 99),
        temperatura=temp(36.5, 37.5),
        sintomas=pick(
            "Cólica nefrética intensa, EVA 10/10, irradiando para flanco e região inguinal. Náuseas e vômitos. Paciente inquieto, não consegue ficar parado.",
            "Lombalgia aguda intensa após esforço, EVA 9/10, sem irradiação para MMII, sem déficit neurológico.",
            "Dor abdominal em FID intensa há 6 horas, EVA 8/10, com defesa local. Suspeita de apendicite aguda.",
            "Cefaleia intensa em trovão, EVA 10/10, de início súbito há 1 hora. Paciente refere ser a pior dor da vida.",
        ),
        historico=pick(
            "Episódio prévio de cálculo renal há 2 anos.",
            "Sem comorbidades.",
            "Hipertensa em uso de hidroclorotiazida.",
            "Histórico de enxaqueca crônica, mas refere ser dor diferente.",
        ),
        classificacao="LARANJA",
    )


def _hipertensao_sintomatica() -> dict:
    sis, _, pres = pa(190, 240, dia_offset=(-70, -55))
    return dict(
        idade=random.randint(50, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(75, 105),
        spo2=spo2(95, 99),
        temperatura=temp(36.4, 37.0),
        sintomas=pick(
            "Cefaleia intensa, alteração visual (escotomas), sensação de formigamento em hemiface direita há 3 horas. PA persistentemente elevada.",
            "Crise hipertensiva sintomática com confusão mental leve, náuseas e cefaleia occipital.",
            "Epistaxe abundante, cefaleia intensa, PA muito elevada na chegada. Sem déficit neurológico focal.",
            "PA muito elevada com dor torácica leve e dispneia aos pequenos esforços, sugerindo emergência hipertensiva.",
        ),
        historico=pick(
            "HAS de longa data, em uso irregular de losartana e anlodipino.",
            "Hipertensa, sem aderência ao tratamento.",
            "Diabético, hipertenso, em uso de múltiplos anti-hipertensivos.",
            "Insuficiência renal crônica em conservador.",
        ),
        classificacao="LARANJA",
    )


def _bradicardia_taquicardia_sintomatica() -> dict:
    sis, _, pres = pa(95, 145)
    extrema_alta = maybe(0.5)
    return dict(
        idade=random.randint(40, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(150, 200) if extrema_alta else fc(28, 38),
        spo2=spo2(93, 98),
        temperatura=temp(36.2, 37.2),
        sintomas=pick(
            "Palpitações intensas há 1 hora, sensação de falta de ar leve, tontura. Paciente refere FC muito elevada ao palpar pulso.",
            "Bradicardia sintomática com tontura, escurecimento visual e fraqueza generalizada. Pré-síncope em pé.",
            "Taquicardia supraventricular sintomática, dispneia leve, dor precordial atípica.",
            "Síncope após sensação de palpitação rápida, recuperou consciência espontaneamente. Mantém FC muito alta no monitor.",
        ),
        historico=pick(
            "FA paroxística conhecida, em uso de propranolol.",
            "Cardiopata isquêmico, marca-passo definitivo há 3 anos.",
            "Sem comorbidades cardiológicas conhecidas.",
            "Hipertireoidismo em investigação.",
        ),
        classificacao="LARANJA",
    )


def _hipertermia_severa() -> dict:
    sis, _, pres = pa(95, 135)
    return dict(
        idade=random.randint(2, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(110, 145),
        spo2=spo2(92, 98),
        temperatura=temp(39.6, 41.5),
        sintomas=pick(
            "Febre alta (40°C) há 6 horas refratária a antitérmicos. Cefaleia intensa, mialgia generalizada, prostração.",
            "Hipertermia severa (39,8°C) em criança de 2 anos, irritada, com gemência.",
            "Febre persistente acima de 40°C há 24 horas, calafrios, sudorese profusa, mialgia importante.",
            "Idosa com febre 39,7°C, confusa, desidratada. Sem foco aparente identificado.",
        ),
        historico=pick(
            "Sem comorbidades conhecidas.",
            "Imunossuprimida em uso de metotrexato por artrite reumatoide.",
            "Diabético tipo 2, em uso de metformina.",
            "Idosa institucionalizada, demência avançada.",
        ),
        classificacao="LARANJA",
    )


def _gestante_alto_risco() -> dict:
    sis, _, pres = pa(95, 165)
    return dict(
        idade=random.randint(16, 42),
        sexo="Feminino",
        pressao=pres,
        frequencia_cardiaca=fc(85, 120),
        spo2=spo2(94, 99),
        temperatura=temp(36.3, 37.5),
        sintomas=pick(
            "Gestante 32 semanas com sangramento vaginal volumoso há 2 horas, sem contrações regulares. Refere movimentos fetais reduzidos hoje.",
            "Gestante a termo com dor abdominal intensa em baixo ventre, sangramento vaginal moderado, refere ausência de movimentação fetal nas últimas 6 horas.",
            "Cefaleia intensa, escotomas e epigastralgia em gestante de 36 semanas com PA elevada. Suspeita de pré-eclâmpsia grave.",
            "Gestante 28 semanas com perda de líquido amniótico há 4 horas, contrações regulares, sem sangramento.",
        ),
        historico=pick(
            "Primigesta, pré-natal regular, sem intercorrências prévias.",
            "G3P2A0, último parto cesárea há 2 anos, gestação atual de risco habitual.",
            "Hipertensão gestacional em uso de metildopa.",
            "Diabetes gestacional em controle dietético.",
        ),
        classificacao="LARANJA",
    )


def _abdome_agudo() -> dict:
    sis, _, pres = pa(95, 145)
    return dict(
        idade=random.randint(30, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 130),
        spo2=spo2(94, 99),
        temperatura=temp(37.5, 38.8),
        sintomas=pick(
            "Dor abdominal difusa intensa há 8 horas, EVA 8/10, com piora progressiva. Abdome em tábua, sinais de peritonite. Náuseas, vômitos biliosos.",
            "Dor abdominal súbita em punhalada em flanco esquerdo, irradiando para dorso, palidez, taquicardia. Suspeita de aneurisma de aorta abdominal.",
            "Quadro de obstrução intestinal: distensão abdominal importante, vômitos fecaloides, ausência de eliminação de gases há 36h.",
            "Isquemia mesentérica suspeita: dor abdominal desproporcional ao exame, idoso com FA não anticoagulada, acidose metabólica laboratorial.",
        ),
        historico=pick(
            "Hipertenso, dislipidêmico, ex-tabagista.",
            "FA crônica não anticoagulada por escolha do paciente.",
            "Cirurgias abdominais prévias (apendicectomia, colecistectomia).",
            "Doença diverticular conhecida.",
        ),
        classificacao="LARANJA",
    )


def _crise_psiquiatrica() -> dict:
    sis, _, pres = pa(110, 165)
    return dict(
        idade=random.randint(16, 60),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 130),
        spo2=spo2(95, 99),
        temperatura=temp(36.5, 37.4),
        sintomas=pick(
            "Surto psicótico agudo com agitação psicomotora intensa, heteroagressividade verbal, alucinações auditivas. Risco para si e para terceiros.",
            "Tentativa de autoextermínio por intoxicação medicamentosa há 1 hora. Paciente sonolento, vias aéreas pérvias.",
            "Crise de pânico severa com despersonalização, dispneia, parestesias em mãos e perilabiais. Refere ideação suicida.",
            "Episódio maníaco com agressividade, fuga de ideias, sem dormir há 3 dias, ameaçando familiares.",
        ),
        historico=pick(
            "Esquizofrenia em uso irregular de risperidona.",
            "Transtorno bipolar tipo I, várias internações psiquiátricas prévias.",
            "Depressão em uso de fluoxetina, com ideação suicida prévia.",
            "Primeiro episódio psiquiátrico documentado.",
        ),
        classificacao="LARANJA",
    )


def _avc_subagudo() -> dict:
    sis, _, pres = pa(150, 210)
    return dict(
        idade=random.randint(55, 88),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 105),
        spo2=spo2(94, 98),
        temperatura=temp(36.2, 37.4),
        sintomas=pick(
            "Hemiparesia direita iniciada há 12 horas, fora de janela para trombólise. Disartria leve, alerta. NIHSS 6.",
            "Paciente acordou com déficit de força em MSE há 8 horas. AVCi em território de ACM esquerda, fora de janela.",
            "TIA com hemiparesia transitória que reverteu em 30 minutos, ocorrida há 5 horas. Sem déficit residual.",
            "Hemiparesia leve em MID iniciada há 18 horas, paciente assintomático no momento, sem novos eventos.",
        ),
        historico=pick(
            "HAS, FA paroxística, dislipidemia, em uso de varfarina.",
            "Diabético, hipertenso, AVC prévio com sequela motora leve.",
            "Tabagista pesado, em uso de AAS profilático.",
            "Hipertensão grave, sem aderência ao tratamento.",
        ),
        classificacao="LARANJA",
    )


# ---------------------------------------------------------------------------
# Cenários — AMARELO (Urgente, 60 min)
# ---------------------------------------------------------------------------
def _dor_moderada() -> dict:
    sis, _, pres = pa(110, 145)
    return dict(
        idade=random.randint(18, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(75, 100),
        spo2=spo2(96, 99),
        temperatura=temp(36.4, 37.3),
        sintomas=pick(
            "Lombalgia moderada após esforço físico há 2 dias, EVA 5/10, sem irradiação para MMII, sem alteração de força.",
            "Cefaleia tensional moderada há 1 dia, EVA 5/10, contínua, sem alterações visuais ou neurológicas.",
            "Dor em joelho direito após queda há 12 horas, edema moderado, dificuldade para deambular, EVA 6/10.",
            "Otalgia intensa em ouvido esquerdo há 24 horas, EVA 6/10, com hipoacusia. Sem febre.",
            "Cervicalgia moderada com limitação de movimento há 3 dias, EVA 5/10, sem irradiação.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Hipertenso em uso de losartana.",
            "Histórico de lombalgia crônica.",
            "Diabético tipo 2 controlado.",
        ),
        classificacao="AMARELO",
    )


def _febre_alta() -> dict:
    sis, _, pres = pa(105, 140)
    return dict(
        idade=random.randint(18, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 120),
        spo2=spo2(95, 99),
        temperatura=temp(39.0, 39.5),
        sintomas=pick(
            "Febre alta (39,2°C) há 24 horas, calafrios, mialgia generalizada, cefaleia. Sem foco aparente. Estado geral preservado.",
            "Quadro gripal com febre 39°C há 2 dias, tosse, coriza, dor de garganta. Sem dispneia.",
            "Febre persistente 39,3°C há 36 horas com disúria e polaciúria. Suspeita de ITU.",
            "Febre 39,1°C com cefaleia retro-orbitária, dor articular intensa, exantema discreto. Paciente vindo de área endêmica.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Hipertenso, diabético tipo 2.",
            "ITU de repetição.",
            "Sem comorbidades. Vacinação em dia.",
        ),
        classificacao="AMARELO",
    )


def _vomitos_persistentes() -> dict:
    sis, _, pres = pa(95, 135)
    return dict(
        idade=random.randint(15, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(85, 115),
        spo2=spo2(96, 99),
        temperatura=temp(36.5, 38.4),
        sintomas=pick(
            "Vômitos persistentes há 12 horas, mais de 10 episódios, associados a diarreia. Desidratação leve a moderada, mucosas secas.",
            "Êmese persistente há 24 horas em paciente diabético, dor abdominal moderada, refere sede intensa.",
            "Vômitos incoercíveis em gestante de 1° trimestre, perda ponderal de 3 kg em 1 semana, sinais de desidratação.",
            "Quadro de gastroenterite com vômitos volumosos há 8 horas, diarreia aquosa, dor abdominal em cólica.",
        ),
        historico=pick(
            "DM2 em uso de insulina.",
            "Sem comorbidades.",
            "Gastrite crônica em uso eventual de omeprazol.",
            "Gestante de 10 semanas, primeira gestação.",
        ),
        classificacao="AMARELO",
    )


def _dor_abdominal_moderada() -> dict:
    sis, _, pres = pa(105, 140)
    return dict(
        idade=random.randint(18, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(80, 110),
        spo2=spo2(96, 99),
        temperatura=temp(36.5, 38.0),
        sintomas=pick(
            "Dor abdominal em mesogástrio há 18 horas, contínua, EVA 6/10, sem sinais de peritonite. Náuseas, sem vômitos.",
            "Dor em hipocôndrio direito há 6 horas em pós-prandial gorduroso, EVA 6/10, irradiação para dorso. Sem icterícia.",
            "Dor pélvica em mulher jovem, moderada, há 24 horas, ciclo menstrual atrasado. Sem febre.",
            "Disúria, dor lombar leve, polaciúria há 2 dias, urina turva e fétida. Sem febre alta.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Colelitíase conhecida em fila de cirurgia.",
            "Sexualmente ativa, sem método contraceptivo.",
            "ITU de repetição.",
        ),
        classificacao="AMARELO",
    )


def _dispneia_leve() -> dict:
    sis, _, pres = pa(110, 145)
    return dict(
        idade=random.randint(40, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(85, 110),
        spo2=spo2(95, 97),
        temperatura=temp(36.5, 38.0),
        sintomas=pick(
            "Dispneia aos médios esforços há 2 dias, sem dispneia em repouso. Tosse seca leve. Sem dor torácica.",
            "Falta de ar leve com sibilância intermitente, paciente conhecido asmático em descompensação leve.",
            "Cansaço aos esforços, ortopneia há 3 dias, edema discreto em MMII. Suspeita de IC descompensada.",
            "Tosse produtiva há 1 semana com expectoração amarelada, dispneia leve, febre baixa.",
        ),
        historico=pick(
            "DPOC GOLD II, ex-tabagista.",
            "Asma intermitente em uso eventual de salbutamol.",
            "IC CF II em uso de losartana, espironolactona.",
            "Hipertenso em uso de IECA.",
        ),
        classificacao="AMARELO",
    )


def _hipertensao_assintomatica_alta() -> dict:
    sis, _, pres = pa(165, 215)
    return dict(
        idade=random.randint(45, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(75, 100),
        spo2=spo2(96, 99),
        temperatura=temp(36.4, 37.0),
        sintomas=pick(
            "PA aferida em casa elevada (180/110) há 2 horas, refere apenas leve dor de cabeça. Sem alterações visuais ou neurológicas.",
            "Hipertensão sistólica isolada (sis 195) detectada em consulta de rotina, paciente assintomático.",
            "Cefaleia leve há 1 dia, PA aferida 175/105 em casa, paciente fez uso de AINE recentemente.",
            "Tontura ao mudar de posição, PA 185/100 ao chegar. Sem outros sintomas.",
        ),
        historico=pick(
            "HAS em uso irregular de losartana.",
            "Hipertenso recém-diagnosticado, sem tratamento.",
            "DM2, hipertenso, em uso de múltiplas medicações.",
            "Hipertensa em uso de hidroclorotiazida há anos.",
        ),
        classificacao="AMARELO",
    )


def _cefaleia_moderada() -> dict:
    sis, _, pres = pa(110, 150)
    return dict(
        idade=random.randint(15, 70),
        sexo=pick("Masculino", "Feminino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(75, 100),
        spo2=spo2(96, 99),
        temperatura=temp(36.4, 37.3),
        sintomas=pick(
            "Cefaleia hemicraniana há 6 horas, pulsátil, EVA 6/10, com fotofobia e fonofobia. Padrão habitual de enxaqueca.",
            "Dor de cabeça em peso há 2 dias, contínua, EVA 5/10, alivia parcialmente com analgésico simples.",
            "Cefaleia frontal e periorbitária há 3 dias com obstrução nasal e dor à palpação dos seios da face. Suspeita de sinusite.",
            "Cefaleia tensional moderada há 24 horas, EVA 5/10, sem alterações neurológicas.",
        ),
        historico=pick(
            "Histórico de enxaqueca crônica em uso de propranolol profilático.",
            "Sem comorbidades.",
            "Sinusopatia crônica.",
            "Hipertensa em uso de losartana.",
        ),
        classificacao="AMARELO",
    )


def _ITU_complicada() -> dict:
    sis, _, pres = pa(105, 140)
    return dict(
        idade=random.randint(18, 80),
        sexo=pick("Feminino", "Feminino", "Masculino"),
        pressao=pres,
        frequencia_cardiaca=fc(85, 115),
        spo2=spo2(96, 99),
        temperatura=temp(38.0, 39.0),
        sintomas=pick(
            "Disúria, polaciúria, dor lombar à direita, febre 38,5°C há 24 horas. Suspeita de pielonefrite, sem instabilidade.",
            "Dor em flanco direito há 2 dias, urgência miccional, urina turva, febre 38,3°C. Sem náuseas importantes.",
            "Pielonefrite recorrente, queixas urinárias típicas + febre 38°C, sem confusão ou hipotensão.",
            "ITU complicada em paciente diabético com queixas urinárias e febre 38,7°C, hidratação preservada.",
        ),
        historico=pick(
            "ITU de repetição, último episódio há 3 meses.",
            "DM2 descompensado, ITU recorrente.",
            "Cálculo renal direito conhecido.",
            "Cateter vesical de demora há 1 mês.",
        ),
        classificacao="AMARELO",
    )


def _trauma_fechado_simples() -> dict:
    sis, _, pres = pa(110, 145)
    return dict(
        idade=random.randint(15, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(80, 110),
        spo2=spo2(96, 99),
        temperatura=temp(36.4, 37.0),
        sintomas=pick(
            "Trauma em punho esquerdo após queda da própria altura há 2 horas. Edema moderado, dor 6/10, deformidade discreta. Sem outros sinais de fratura.",
            "Entorse de tornozelo direito há 4 horas após acidente desportivo, edema importante, dor à palpação maleolar lateral.",
            "Contusão em hemitórax esquerdo após trauma em pancada lateral, dor à respiração profunda, sem dispneia ou enfisema.",
            "Fratura de antebraço fechada após queda, deformidade evidente, dor controlada com analgésico oral.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Osteoporose em uso de cálcio e alendronato.",
            "Sem comorbidades. Atleta amador.",
            "Hipertenso em uso de IECA.",
        ),
        classificacao="AMARELO",
    )


def _hiperglicemia_moderada() -> dict:
    sis, _, pres = pa(105, 145)
    return dict(
        idade=random.randint(35, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(85, 115),
        spo2=spo2(96, 99),
        temperatura=temp(36.5, 37.3),
        sintomas=pick(
            "Diabético referindo poliúria, polidipsia, fraqueza há 24 horas. HGT em casa 380 mg/dL. Sem náuseas, sem dispneia.",
            "Glicemia muito elevada (HGT > 400) detectada na chegada, paciente lúcido, sem sinais de cetoacidose franca.",
            "Hiperglicemia sintomática com poliúria intensa e perda ponderal de 4 kg em 1 mês.",
            "Diabético descompensado com queixas inespecíficas, glicemia em torno de 350 mg/dL, sem alteração de consciência.",
        ),
        historico=pick(
            "DM2 em uso irregular de metformina.",
            "DM1 desde a infância, em uso de insulina basal-bolus.",
            "DM recém-diagnosticado, sem tratamento iniciado.",
            "DM2 em uso de glibenclamida e metformina, mau controle.",
        ),
        classificacao="AMARELO",
    )


def _crise_asmatica_leve() -> dict:
    sis, _, pres = pa(105, 140)
    return dict(
        idade=random.randint(8, 60),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 120),
        spo2=spo2(94, 97),
        temperatura=temp(36.4, 37.5),
        sintomas=pick(
            "Crise de asma leve há 2 horas, sibilância intermitente, dispneia leve, conseguindo formular frases longas.",
            "Sibilância e tosse seca há 1 dia em paciente asmático, sem uso de musculatura acessória, saturação 96%.",
            "Quadro de bronquite aguda em paciente asmático, dispneia leve a moderada, expectoração mucoide.",
            "Asma com piora há 3 dias, broncodilatador de resgate sem efeito completo, sem instabilidade.",
        ),
        historico=pick(
            "Asma persistente leve em uso de budesonida.",
            "Asma episódica desde a infância.",
            "Asma e rinite alérgica.",
            "Asma em uso de salbutamol PRN.",
        ),
        classificacao="AMARELO",
    )


def _diarreia_desidratacao() -> dict:
    sis, _, pres = pa(95, 130)
    return dict(
        idade=random.randint(2, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(95, 120),
        spo2=spo2(96, 99),
        temperatura=temp(36.5, 38.5),
        sintomas=pick(
            "Diarreia aquosa há 24 horas, mais de 10 evacuações, vômitos associados, mucosas secas, oligúrico.",
            "Quadro de gastroenterite com diarreia profusa há 2 dias, dor abdominal em cólica, desidratação moderada.",
            "Criança de 5 anos com diarreia há 3 dias, perdeu peso visivelmente, mucosas ressecadas, choro sem lágrimas.",
            "Idoso com diarreia e vômitos há 24 horas, hipotensão postural, fraqueza importante.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Idoso institucionalizado.",
            "Gastrite crônica.",
            "Imunossuprimido leve por uso de corticoide inalatório.",
        ),
        classificacao="AMARELO",
    )


# ---------------------------------------------------------------------------
# Cenários — VERDE (Pouco urgente, 120 min)
# ---------------------------------------------------------------------------
def _dor_leve() -> dict:
    sis, _, pres = pa(110, 140)
    return dict(
        idade=random.randint(15, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 37.0),
        sintomas=pick(
            "Lombalgia leve há 2 dias após esforço físico, EVA 3/10, melhora com analgésico simples. Sem irradiação.",
            "Cefaleia leve há 1 dia, EVA 2/10, contínua, paciente conseguiu trabalhar normalmente.",
            "Dor em punho direito após pequena torção há 1 dia, edema discreto, EVA 3/10. Movimentação preservada.",
            "Mialgia em panturrilhas após atividade física intensa há 2 dias, EVA 2/10, sem outros sintomas.",
            "Dor leve em cotovelo direito por uso repetitivo (epicondilite), EVA 3/10, há 1 semana.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Sedentário.",
            "Pratica musculação regularmente.",
            "Hipertenso controlado.",
        ),
        classificacao="VERDE",
    )


def _IVAS() -> dict:
    sis, _, pres = pa(108, 135)
    return dict(
        idade=random.randint(2, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.5, 38.4),
        sintomas=pick(
            "Resfriado há 3 dias com coriza, espirros, dor de garganta leve. Febre baixa (37,8°C). Sem dispneia.",
            "Tosse seca há 5 dias com odinofagia leve, sem febre alta, sem falta de ar.",
            "Quadro gripal com mialgia leve, mal-estar e febre baixa, há 2 dias. Apetite preservado.",
            "Faringite viral, dor para deglutir leve, hiperemia de orofaringe, sem placas. Febre 37,8°C.",
            "Rinorreia clara abundante, espirros em salva, congestão nasal há 2 dias. Sem febre.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Rinite alérgica.",
            "Sem comorbidades. Vacinação em dia.",
            "Histórico de quadros gripais frequentes.",
        ),
        classificacao="VERDE",
    )


def _dor_muscular() -> dict:
    sis, _, pres = pa(110, 140)
    return dict(
        idade=random.randint(15, 70),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 37.0),
        sintomas=pick(
            "Contusão em coxa direita após pancada em prática esportiva há 6 horas. Edema discreto, hematoma em formação. Deambulação preservada.",
            "Entorse leve de tornozelo direito há 1 dia, edema discreto, dor à movimentação, deambula com leve claudicação.",
            "Cervicalgia leve por má postura no trabalho há 1 semana, sem irradiação, sem alterações neurológicas.",
            "Tendinite no ombro direito por uso repetitivo, dor leve à movimentação, sem limitação funcional importante.",
        ),
        historico=pick(
            "Sem comorbidades. Pratica esportes regularmente.",
            "Sedentário.",
            "Trabalho braçal.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


def _diarreia_leve() -> dict:
    sis, _, pres = pa(110, 135)
    return dict(
        idade=random.randint(5, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.4, 37.5),
        sintomas=pick(
            "Diarreia 3 a 4 vezes ao dia há 24 horas, sem sangue, sem febre. Hidratação preservada, ingestão hídrica adequada.",
            "Episódios diarreicos há 2 dias, autolimitados, sem desidratação. Refere ter comido alimento suspeito.",
            "Diarreia leve com cólica abdominal de baixa intensidade há 36 horas. Continua se alimentando.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Síndrome do intestino irritável conhecida.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


def _conjuntivite() -> dict:
    sis, _, pres = pa(110, 135)
    return dict(
        idade=random.randint(5, 70),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 92),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 37.0),
        sintomas=pick(
            "Olho vermelho bilateral há 2 dias com secreção amarelada matinal, prurido leve. Sem alterações visuais.",
            "Hiperemia conjuntival e secreção em olho direito há 3 dias, lacrimejamento abundante. Visão preservada.",
            "Conjuntivite viral suspeita, olhos vermelhos, prurido intenso, secreção aquosa, contato com caso similar em casa.",
        ),
        historico=pick(
            "Sem comorbidades. Conjuntivites recorrentes.",
            "Rinite alérgica.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


def _dermatite_alergica() -> dict:
    sis, _, pres = pa(110, 140)
    return dict(
        idade=random.randint(5, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.4, 37.0),
        sintomas=pick(
            "Lesões eritematosas e prurido em região cervical há 3 dias após uso de bijuteria nova. Sem sintomas sistêmicos.",
            "Erupção cutânea em tronco e MMSS, prurido moderado, sem dispneia, sem edema facial. Início após uso de antibiótico há 2 dias.",
            "Urticária recorrente leve em MMII há 1 semana, sem outros sintomas. Pruriginosa.",
            "Eczema em mãos por contato com produto de limpeza, descamação e prurido leves há 5 dias.",
        ),
        historico=pick(
            "Atopia conhecida.",
            "Sem comorbidades. Possível alergia em investigação.",
            "Asma e rinite alérgica.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


def _odontalgia() -> dict:
    sis, _, pres = pa(110, 140)
    return dict(
        idade=random.randint(15, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(75, 98),
        spo2=spo2(97, 99),
        temperatura=temp(36.4, 37.3),
        sintomas=pick(
            "Dor de dente intensa há 2 dias, EVA 6/10, em molar inferior esquerdo. Sem febre, sem trismo importante.",
            "Odontalgia em incisivo superior direito, sensibilidade ao frio, há 4 dias, EVA 5/10.",
            "Dor dentária após procedimento odontológico há 24 horas, EVA 5/10, edema gengival local.",
            "Pericoronarite em terceiro molar inferior, dor moderada, dificuldade para abrir a boca.",
        ),
        historico=pick(
            "Sem acompanhamento odontológico regular há anos.",
            "Sem comorbidades.",
            "Bruxismo conhecido.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


def _ITU_simples() -> dict:
    sis, _, pres = pa(108, 135)
    return dict(
        idade=random.randint(18, 70),
        sexo=pick("Feminino", "Feminino", "Feminino", "Masculino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.4, 37.5),
        sintomas=pick(
            "Disúria e polaciúria há 24 horas, sem febre, sem dor lombar. Urina sem alteração macroscópica importante.",
            "Queixas urinárias há 2 dias (ardência ao urinar, urgência), sem febre, sem dor lombar. Sem náuseas.",
            "Cistite simples, disúria leve, polaciúria, urina turva. Sem sintomas sistêmicos.",
        ),
        historico=pick(
            "ITU de repetição leve.",
            "Sem comorbidades.",
            "Diabética compensada.",
        ),
        classificacao="VERDE",
    )


def _quadro_gastrite() -> dict:
    sis, _, pres = pa(108, 138)
    return dict(
        idade=random.randint(18, 75),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 37.0),
        sintomas=pick(
            "Dor epigástrica em queimação há 3 dias, piora com alimentação, EVA 4/10. Sem vômitos, sem sangramento.",
            "Pirose e regurgitação ácida há 1 semana, alívio parcial com antiácido. Sem disfagia.",
            "Dor em queimação retroesternal pós-prandial há vários dias, sem dor torácica isquêmica.",
            "Quadro dispéptico com plenitude pós-prandial, eructações, náuseas leves há 5 dias.",
        ),
        historico=pick(
            "Gastrite crônica em uso eventual de omeprazol.",
            "Etilista social. Tabagista.",
            "Sem comorbidades.",
            "DRGE conhecida.",
        ),
        classificacao="VERDE",
    )


def _quadro_dermato_leve() -> dict:
    sis, _, pres = pa(108, 138)
    return dict(
        idade=random.randint(2, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 92),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 37.4),
        sintomas=pick(
            "Lesão circular descamativa pruriginosa em antebraço direito há 1 semana, suspeita de tinea corporis.",
            "Acne inflamatória moderada em face, com algumas lesões pustulosas, sem sinais de infecção sistêmica.",
            "Pequena ferida em pé direito há 5 dias, lenta cicatrização, sem sinais flogísticos importantes.",
            "Ressecamento e prurido em pernas há 2 semanas, suspeita de eczema atópico.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Diabético tipo 2 com cuidado moderado dos pés.",
            "Sem comorbidades.",
            "Atopia conhecida.",
        ),
        classificacao="VERDE",
    )


def _agudizacao_cronica_leve() -> dict:
    sis, _, pres = pa(110, 145)
    return dict(
        idade=random.randint(35, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 98),
        spo2=spo2(96, 99),
        temperatura=temp(36.4, 37.2),
        sintomas=pick(
            "Piora discreta da dor crônica em joelho direito há 3 dias, EVA 4/10. Paciente com osteoartrose conhecida.",
            "Cefaleia tensional habitual mais frequente nos últimos dias, intensidade leve, EVA 3/10.",
            "Constipação intestinal habitual com piora há 1 semana, sem sangramento, sem dor importante.",
            "Tontura postural leve em hipertensa que ajustou medicação recentemente.",
        ),
        historico=pick(
            "Osteoartrose poliarticular.",
            "Cefaleia tensional crônica.",
            "Constipação crônica.",
            "Hipertensa em ajuste de medicação.",
        ),
        classificacao="VERDE",
    )


def _otalgia_leve() -> dict:
    sis, _, pres = pa(108, 138)
    return dict(
        idade=random.randint(2, 70),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(72, 95),
        spo2=spo2(97, 99),
        temperatura=temp(36.4, 37.6),
        sintomas=pick(
            "Otalgia leve à direita há 2 dias, sem febre alta, sem otorreia. Histórico recente de IVAS.",
            "Sensação de ouvido tampado há 3 dias após resfriado, hipoacusia leve, sem dor importante.",
            "Otite externa por contato com água de piscina há 4 dias, prurido e dor leve.",
            "Tampão de cera com sensação de plenitude e leve hipoacusia há 1 semana.",
        ),
        historico=pick(
            "Otites recorrentes na infância.",
            "Sem comorbidades.",
            "Frequentador de piscina.",
            "Sem comorbidades.",
        ),
        classificacao="VERDE",
    )


# ---------------------------------------------------------------------------
# Cenários — AZUL (Não urgente, 240 min)
# ---------------------------------------------------------------------------
def _renovacao_receita() -> dict:
    sis, _, pres = pa(115, 140)
    return dict(
        idade=random.randint(40, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(68, 88),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Paciente comparece para renovação de receita de medicação contínua de uso crônico. Assintomático no momento.",
            "Pedido de receita para anti-hipertensivo de uso habitual. Refere medicação acabou ontem. Sem sintomas.",
            "Solicita renovação de receita controlada de antidepressivo de uso há 3 anos, em acompanhamento estável.",
            "Comparece para receita de medicação para diabetes, em uso regular, assintomático.",
        ),
        historico=pick(
            "HAS em uso de losartana.",
            "DM2 em uso de metformina.",
            "Depressão em uso de sertralina.",
            "Hipotireoidismo em uso de levotiroxina.",
        ),
        classificacao="AZUL",
    )


def _resultado_exame() -> dict:
    sis, _, pres = pa(115, 140)
    return dict(
        idade=random.randint(20, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(68, 90),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Comparece para mostrar resultado de exames laboratoriais solicitados na última consulta. Assintomático.",
            "Trouxe resultado de ultrassom para avaliação. Sem queixas atuais.",
            "Solicita interpretação de exame de imagem realizado há 2 dias. Sem sintomas agudos.",
            "Veio buscar resultado de biópsia. Sem queixas no momento.",
        ),
        historico=pick(
            "Acompanhamento para investigação de massa abdominal.",
            "Cardiopata em acompanhamento ambulatorial.",
            "Diabético em acompanhamento.",
            "Sem comorbidades.",
        ),
        classificacao="AZUL",
    )


def _atestado() -> dict:
    sis, _, pres = pa(115, 138)
    return dict(
        idade=random.randint(18, 65),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(68, 88),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Comparece solicitando atestado médico para justificar falta no trabalho de ontem. Assintomático no momento.",
            "Pede declaração de comparecimento. Sem queixas clínicas.",
            "Solicita atestado para academia/atividade física. Sem queixas.",
            "Veio buscar atestado de saúde para escola. Sem sintomas.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Hipertensa controlada.",
            "Sem comorbidades.",
            "Diabético compensado.",
        ),
        classificacao="AZUL",
    )


def _curativo_eletivo() -> dict:
    sis, _, pres = pa(115, 138)
    return dict(
        idade=random.randint(20, 85),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 90),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Comparece para troca de curativo programado de ferida operatória de cirurgia há 7 dias. Sem sinais flogísticos.",
            "Curativo de úlcera venosa em MIE, sem piora, sem secreção purulenta. Programado.",
            "Retirada de pontos de sutura realizada há 10 dias. Cicatrização adequada.",
            "Curativo eletivo de cateter venoso central, sem sinais de infecção.",
        ),
        historico=pick(
            "Pós-operatório de colecistectomia.",
            "Insuficiência venosa crônica.",
            "Pós-operatório de pequena cirurgia ambulatorial.",
            "Em quimioterapia, com cateter de longa permanência.",
        ),
        classificacao="AZUL",
    )


def _vacinacao_orientacao() -> dict:
    sis, _, pres = pa(115, 138)
    return dict(
        idade=random.randint(2, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 92),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Veio para vacinação de rotina conforme calendário. Assintomática.",
            "Solicita orientação sobre prevenção de doenças. Sem queixas atuais.",
            "Pede orientação sobre uso de medicação prescrita em outro serviço. Sem sintomas.",
            "Comparece para imunização contra influenza, conforme campanha. Saudável.",
        ),
        historico=pick(
            "Sem comorbidades. Vacinação em dia.",
            "Idoso, dentro de grupo prioritário para vacinação.",
            "Profissional da saúde.",
            "Gestante de 24 semanas, vacinação programada.",
        ),
        classificacao="AZUL",
    )


def _aferir_PA_rotina() -> dict:
    sis, _, pres = pa(115, 145)
    return dict(
        idade=random.randint(40, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 90),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Comparece apenas para aferir pressão arterial, conforme orientação médica anterior. Sem sintomas.",
            "Quer verificar PA porque está sem aparelho em casa. Assintomático no momento.",
            "Aferição de PA de rotina mensal, conforme acompanhamento. Sem queixas.",
            "Veio aferir glicemia capilar de rotina. Sem sintomas.",
        ),
        historico=pick(
            "HAS em uso regular de medicação.",
            "DM2 em controle.",
            "HAS leve em acompanhamento.",
            "Pré-diabético em monitoramento.",
        ),
        classificacao="AZUL",
    )


def _orientacao_medicamento() -> dict:
    sis, _, pres = pa(115, 138)
    return dict(
        idade=random.randint(20, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 90),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Veio pedir orientação sobre como tomar medicação prescrita em consulta anterior. Sem queixas.",
            "Tem dúvida sobre dose de antibiótico recém-prescrito. Assintomático.",
            "Solicita esclarecimento sobre interação entre medicações de uso contínuo.",
            "Quer entender como aplicar insulina recém-prescrita. Sem sintomas atuais.",
        ),
        historico=pick(
            "DM2 recém-diagnosticado, iniciando insulina.",
            "Vários medicamentos em uso por múltiplas comorbidades.",
            "Sem comorbidades.",
            "Hipertenso em ajuste de medicação.",
        ),
        classificacao="AZUL",
    )


def _comparece_visita_familiar() -> dict:
    sis, _, pres = pa(115, 140)
    return dict(
        idade=random.randint(20, 80),
        sexo=pick("Masculino", "Feminino"),
        pressao=pres,
        frequencia_cardiaca=fc(70, 90),
        spo2=spo2(97, 99),
        temperatura=temp(36.3, 36.9),
        sintomas=pick(
            "Pede declaração de acompanhante para idoso internado. Assintomático.",
            "Comparece a pedido administrativo para preenchimento de formulário. Sem queixas.",
            "Quer informações sobre internação de familiar. Sem sintomas.",
            "Solicita declaração de saúde para fins burocráticos. Sem queixas atuais.",
        ),
        historico=pick(
            "Sem comorbidades.",
            "Hipertensa controlada.",
            "Sem comorbidades.",
            "Diabético compensado.",
        ),
        classificacao="AZUL",
    )


# ---------------------------------------------------------------------------
# Mapa de cenários por cor
# ---------------------------------------------------------------------------
CENARIOS: dict[str, list[Callable[[], dict]]] = {
    "VERMELHO": [
        _pcr, _choque_septico, _iam_choque, _avc_hiperagudo, _trauma_penetrante,
        _convulsao_ativa, _anafilaxia, _alteracao_consciencia_grave,
    ],
    "LARANJA": [
        _dor_toracica_estavel, _dispneia_moderada, _dor_severa, _hipertensao_sintomatica,
        _bradicardia_taquicardia_sintomatica, _hipertermia_severa, _gestante_alto_risco,
        _abdome_agudo, _crise_psiquiatrica, _avc_subagudo,
    ],
    "AMARELO": [
        _dor_moderada, _febre_alta, _vomitos_persistentes, _dor_abdominal_moderada,
        _dispneia_leve, _hipertensao_assintomatica_alta, _cefaleia_moderada,
        _ITU_complicada, _trauma_fechado_simples, _hiperglicemia_moderada,
        _crise_asmatica_leve, _diarreia_desidratacao,
    ],
    "VERDE": [
        _dor_leve, _IVAS, _dor_muscular, _diarreia_leve, _conjuntivite,
        _dermatite_alergica, _odontalgia, _ITU_simples, _quadro_gastrite,
        _quadro_dermato_leve, _agudizacao_cronica_leve, _otalgia_leve,
    ],
    "AZUL": [
        _renovacao_receita, _resultado_exame, _atestado, _curativo_eletivo,
        _vacinacao_orientacao, _aferir_PA_rotina, _orientacao_medicamento,
        _comparece_visita_familiar,
    ],
}

# Distribuição alvo (~ 2000 linhas, levemente enriquecendo classes raras).
DISTRIBUICAO = {
    "VERMELHO": 200,   # 10%
    "LARANJA":  300,   # 15%
    "AMARELO":  500,   # 25%
    "VERDE":    700,   # 35%
    "AZUL":     300,   # 15%
}


# ---------------------------------------------------------------------------
# Extração de features (flags binárias) a partir do texto
# ---------------------------------------------------------------------------
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


def extrair_flags(sintomas: str, historico: str) -> dict[str, int]:
    text = f"{sintomas} {historico}".lower()
    out: dict[str, int] = {}
    for col, padroes in TERMOS.items():
        out[col] = int(any(re.search(p, text) for p in padroes))
    return out


def parse_pa(pres: str) -> tuple[int, int]:
    if "/" in pres:
        try:
            sis, dia = pres.split("/", 1)
            return int(sis), int(dia)
        except (ValueError, TypeError):
            pass
    return 0, 0


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def gerar_dataset() -> list[dict]:
    rows: list[dict] = []
    for cor, n_alvo in DISTRIBUICAO.items():
        cenarios = CENARIOS[cor]
        # Distribui o total entre os cenários (com sobra para alguns)
        por_cenario = n_alvo // len(cenarios)
        sobra = n_alvo - por_cenario * len(cenarios)
        for i, cenario in enumerate(cenarios):
            n = por_cenario + (1 if i < sobra else 0)
            for _ in range(n):
                row = cenario()
                pa_sis, pa_dia = parse_pa(row["pressao"])
                row["pa_sistolica"] = pa_sis
                row["pa_diastolica"] = pa_dia
                row["sexo_M"] = int(row["sexo"] == "Masculino")
                row["sexo_F"] = int(row["sexo"] == "Feminino")
                row.update(extrair_flags(row["sintomas"], row["historico"]))
                rows.append(row)
    random.shuffle(rows)
    return rows


def salvar_csv(rows: list[dict], path: Path) -> None:
    # Ordem das colunas: campos originais → features extraídas → label
    campos_originais = [
        "idade", "sexo", "pressao", "frequencia_cardiaca", "spo2",
        "temperatura", "sintomas", "historico",
    ]
    features = [
        "pa_sistolica", "pa_diastolica", "sexo_M", "sexo_F",
        *sorted(TERMOS.keys()),
    ]
    colunas = campos_originais + features + ["classificacao"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    out_path = Path(__file__).parent / "triagem_dataset.csv"
    dataset = gerar_dataset()
    salvar_csv(dataset, out_path)

    # Resumo
    from collections import Counter
    counts = Counter(r["classificacao"] for r in dataset)
    print(f"✓ Dataset gerado em: {out_path}")
    print(f"✓ Total de linhas: {len(dataset)}")
    print(f"✓ Distribuição:")
    for cor in ["VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL"]:
        n = counts.get(cor, 0)
        pct = 100 * n / len(dataset)
        print(f"    {cor:10s}: {n:4d} ({pct:.1f}%)")
