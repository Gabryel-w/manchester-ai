"""
Gerador de dataset sintetico para treino do RF de triagem Manchester.

Cada cenario clinico e descrito como um dict de TEMPLATES (faixas de
sinais vitais + listas de variacoes textuais para sintomas/historico).
O gerador amostra aleatoriamente dentro das faixas e dos templates,
produzindo dezenas de variacoes por cenario.

Uso:
    python data/gerar_dataset.py
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from features import TERMOS, extrair_flags, faixa_etaria, parse_pa  # noqa: E402

random.seed(42)


# ---------------------------------------------------------------------------
# Templates: cada cenario e um dict com faixas e variacoes
# ---------------------------------------------------------------------------
# Schema:
#   "vit": (idade_min, idade_max, pa_sis_min, pa_sis_max, fc_min, fc_max,
#           spo2_min, spo2_max, temp_min, temp_max)
#   "sex": "M" | "F" | "X" (qualquer)
#   "sintomas": [str, ...]
#   "historico": [str, ...]
# ---------------------------------------------------------------------------

VERMELHO = [
    # PCR
    {"vit": (45,90, 50,80, 20,50, 60,82, 34.5,36.5), "sex":"X",
     "sintomas":[
        "Paciente em parada cardiorrespiratoria, sem pulso central, sem respiracao espontanea. RCP em curso.",
        "PCR presenciada ha 5 minutos, em assistolia ao monitor. Familia relata colapso subito.",
        "Encontrado inconsciente pela familia, sem pulso. RCP iniciada por leigo, equipe assumiu na ambulancia.",
        "Parada cardiorrespiratoria apos desconforto toracico. RCP avancada em curso, intubado.",
        "PCR em fibrilacao ventricular. Choque desfibrilador aplicado, ainda sem retorno de circulacao."],
     "historico":["HAS, DM2, IAM previo, em uso de AAS e estatina.",
                  "Cardiopata isquemico, FA cronica, anticoagulado.",
                  "Sem comorbidades. Tabagista 40 macos/ano.",
                  "Insuficiencia cardiaca CF III, ex-tabagista."]},
    # Choque septico
    {"vit": (50,90, 60,88, 120,160, 82,92, 38.5,40.5), "sex":"X",
     "sintomas":[
        "Paciente febril ha 3 dias, sonolento, extremidades frias e marmoreas. PA persistentemente baixa, taquicardico, oliguric o.",
        "Sepse de foco urinario, hipotensao refrataria a hidratacao, confuso, livedo em MMII.",
        "Quadro de pneumonia evoluindo com hipotensao, taquicardia, alteracao do nivel de consciencia. Sudorese fria.",
        "Foco abdominal de sepse, peritonite difusa, choque septico, anurico.",
        "Sepse de foco pulmonar, paciente confuso, lactato 6.2, PA mantida apenas com volume."],
     "historico":["DM2, ITU de repeticao, em quimioterapia.",
                  "Imunossuprimido por uso cronico de corticoide.",
                  "Cirrose hepatica Child C.",
                  "Diabetico, nefropata, hemodialitico."]},
    # IAM com choque
    {"vit": (55,88, 60,88, 110,150, 82,92, 35.8,36.8), "sex":"M",
     "sintomas":[
        "Dor toracica retroesternal intensa irradiando para braco esquerdo ha 40 minutos. Sudorese fria profusa, palidez, hipotensao.",
        "IAM com supra de ST anterior extenso, choque cardiogenico, dispneia importante e PA persistentemente baixa.",
        "Dor precordial em aperto, irradiando para mandibula, hipotensao e bradicardia. Suspeita de IAM inferior.",
        "Sincope apos dor toracica intensa, recuperou com hipotensao mantida. Pulsos perifericos finos, sudoreico.",
        "Infarto extenso com edema agudo de pulmao, dispneia severa, sibilos difusos, PA 75/50."],
     "historico":["HAS, DM2, dislipidemia, tabagista 30 macos/ano. Pai com IAM precoce.",
                  "Coronariopata, angioplastia ha 4 anos, dupla antiagregacao.",
                  "Hipertenso, obeso, sedentario.",
                  "Sem comorbidades. Tabagista pesado."]},
    # AVC hiperagudo
    {"vit": (55,90, 150,220, 70,110, 92,98, 36.0,37.5), "sex":"X",
     "sintomas":[
        "Hemiparesia direita subita ha 90 minutos, disartria, desvio de comissura labial. Glasgow 13. NIHSS 12.",
        "Afasia de expressao e hemiplegia esquerda ha 2 horas, NIHSS 18. AVCi de ACM direita.",
        "AVC ha 3 horas com plegia em dimidio direito, desvio de olhar conjugado, NIHSS 20. Em janela para trombectomia.",
        "Paciente acordou com hemiparesia e afasia ha 1 hora, sem recuperacao. Glasgow 14, deficit incapacitante."],
     "historico":["HAS, FA paroxistica sem anticoagulacao, dislipidemia.",
                  "Diabetico, hipertenso, AVC isquemico previo.",
                  "Anticoagulado por FA com varfarina, INR nao controlado.",
                  "Cardiopata isquemico, dupla antiagregacao."]},
    # Trauma penetrante
    {"vit": (15,55, 80,130, 90,140, 88,98, 35.8,36.8), "sex":"M",
     "sintomas":[
        "Ferimento por arma de fogo em hemitorax esquerdo ha 20 minutos. Hipotensao progressiva, taquicardico, palidez.",
        "Ferimento por arma branca em regiao epigastrica, abdome em tabua, sinais de irritacao peritoneal.",
        "PAF em regiao cervical, sangramento moderado controlado com compressao. Vias aereas pervias.",
        "Empalamento por barra de ferro em hemitorax direito apos acidente. Paciente alerta, dispneico.",
        "Multiplos ferimentos por arma branca em torax e abdome, sangramento abundante, sudoreico."],
     "historico":["Sem comorbidades.",
                  "Etilista cronico.",
                  "Tabagista, ex-usuario de drogas iliticitas.",
                  "Hipertenso em uso irregular."]},
    # Convulsao ativa
    {"vit": (2,80, 110,160, 110,150, 85,94, 36.5,39.5), "sex":"X",
     "sintomas":[
        "Status epilepticus, crise tonico-clonica generalizada ha 8 minutos sem recuperacao. Cianose perioral, sialorreia.",
        "Crise convulsiva presenciada, durou 10 minutos, paciente em pos-ictal profundo, nao responde aos chamados.",
        "Convulsoes repetidas ha 30 minutos, tres episodios sem recuperar consciencia. Glasgow 7.",
        "Crise focal evoluindo para generalizacao, tremores em hemicorpo esquerdo, agora generalizado."],
     "historico":["Epilepsia em uso irregular de fenitoina.",
                  "Sem antecedentes neurologicos. Primeira crise da vida.",
                  "Etilista cronico em abstinencia ha 48h.",
                  "Tumor cerebral em investigacao, em uso de dexametasona."]},
    # Anafilaxia
    {"vit": (5,70, 70,95, 120,160, 85,94, 36.0,37.0), "sex":"X",
     "sintomas":[
        "Anafilaxia apos ingestao de amendoim ha 15 minutos. Edema de glote, estridor laringeo, dispneia, urticaria, hipotensao.",
        "Choque anafilatico apos picada de inseto, edema facial, broncoespasmo grave, hipotensao refrataria.",
        "Reacao anafilatica medicamentosa (cefalosporina), edema de orofaringe, estridor, sibilancia difusa, hipotensao.",
        "Anafilaxia idiopatica com edema facial e cervical, dificuldade respiratoria progressiva, sibilos."],
     "historico":["Atopia, asma desde a infancia.",
                  "Sem comorbidades. Primeira reacao alergica.",
                  "Alergia previa a penicilina e contraste iodado.",
                  "Asmatica em uso irregular de budesonida."]},
    # Alteracao de consciencia grave
    {"vit": (40,90, 85,200, 50,130, 85,95, 35.5,38.5), "sex":"X",
     "sintomas":[
        "Encontrado inconsciente em casa pela familia, Glasgow 6, anisocoria a direita. Suspeita de AVC hemorragico.",
        "Rebaixamento progressivo ao longo de 3 horas, agora Glasgow 7, nao responde a estimulos.",
        "Coma de etiologia a esclarecer, Glasgow 5, postura de descerebracao ao estimulo doloroso.",
        "Hipoglicemia grave (HGT 28) em diabetico, inconsciente, sudorese fria abundante."],
     "historico":["HAS grave em uso irregular. AVC previo.",
                  "DM2 insulinodependente, episodios previos de hipoglicemia.",
                  "Etilista cronico, hepatopata.",
                  "TCE ha 4 horas apos queda, em uso de varfarina."]},
    # AVC hemorragico (NOVO)
    {"vit": (50,85, 180,240, 50,90, 88,96, 36.5,37.5), "sex":"X",
     "sintomas":[
        "Cefaleia subita em trovao ha 2 horas, EVA 10/10, evoluindo para rebaixamento de consciencia. Glasgow caindo, anisocoria.",
        "AVC hemorragico em TC: hematoma intraparenquimatoso temporal direito 60ml. Sonolento, hemiparesia esquerda, vomitos.",
        "Cefaleia explosiva apos esforco, perda de consciencia inicial, agora confusa, rigidez de nuca."],
     "historico":["HAS grave nao tratada ha anos.",
                  "Anticoagulada por FA, INR supraterapeutico.",
                  "Hipertensa, dislipidemica, sem aderencia."]},
    # Hipoglicemia grave (NOVO)
    {"vit": (30,85, 95,140, 95,130, 94,99, 35.5,36.5), "sex":"X",
     "sintomas":[
        "Hipoglicemia grave HGT 32 em diabetico, inconsciente, sudorese fria profusa. Familia administrou acucar oral sem resposta.",
        "Diabetico encontrado em coma hipoglicemico no domicilio, HGT 28, glasgow 8, palidez intensa.",
        "Hipoglicemia HGT 38 com convulsoes generalizadas em diabetico tipo 1 que pulou refeicao apos dose de insulina."],
     "historico":["DM1 ha 15 anos, multiplas internacoes por hipoglicemia.",
                  "DM2 insulinodependente, mau controle glicemico.",
                  "DM em uso de glibenclamida, idoso com insuficiencia renal."]},
    # HDA com choque (NOVO)
    {"vit": (45,85, 70,95, 115,150, 88,96, 35.8,36.8), "sex":"X",
     "sintomas":[
        "Hematemese volumosa (3 episodios em 1 hora) com instabilidade, palidez, sudorese, taquicardia. Suspeita de varizes esofagianas rotas.",
        "Melena abundante ha 6 horas com hipotensao, taquicardia, lipotimia. Hb 5.2 na chegada.",
        "Hemorragia digestiva alta com choque hipovolemico, palido, sudoreico, pulsos finos. Cirrotico."],
     "historico":["Cirrose Child B, varizes esofagianas conhecidas.",
                  "Etilista cronico, hepatopata, ascite.",
                  "Em uso cronico de AINE por artrose."]},
    # Queimadura grave (NOVO)
    {"vit": (15,65, 85,130, 110,150, 85,96, 36.0,38.5), "sex":"X",
     "sintomas":[
        "Queimadura de 35% SCT por chama (face, torax, MMSS). Dor EVA 10, edema facial progressivo, rouquidao - suspeita de queimadura de via aerea.",
        "Queimadura eletrica de alta voltagem, pontos de entrada e saida visiveis, paciente confuso, urina escura.",
        "Queimadura por agua fervente em 25% SCT em crianca de 4 anos, choro intenso, edema progressivo."],
     "historico":["Sem comorbidades.",
                  "Eletricista de profissao.",
                  "Sem comorbidades. Pediatrico previamente higido."]},
    # TCE grave (NOVO)
    {"vit": (15,80, 100,180, 50,110, 85,96, 36.0,37.5), "sex":"M",
     "sintomas":[
        "TCE grave por queda de altura (4 metros), glasgow 6, anisocoria direita, vomitos em jato. Otorragia. Suspeita de fratura de base de cranio.",
        "Acidente motociclistico em alta velocidade, sem capacete, comatoso, Glasgow 5, postura de descerebracao.",
        "TCE por agressao com objeto contundente, perda de consciencia, agora confuso, anisocoria, hemiparesia direita."],
     "historico":["Sem comorbidades.",
                  "Etilista, em estado de embriaguez no momento do trauma.",
                  "Anticoagulado por FA cronica."]},
]

LARANJA = [
    # Dor toracica estavel
    {"vit": (45,80, 110,160, 70,110, 94,98, 36.2,37.0), "sex":"M",
     "sintomas":[
        "Dor toracica em aperto ha 2 horas, irradiando para braco esquerdo, EVA 7/10. Sem dispneia importante, sem sudorese profusa. Estavel.",
        "Dor precordial intensa ha 1 hora, irradiacao para mandibula, sem instabilidade. ECG aguardando.",
        "Desconforto retroesternal de inicio subito apos esforco, dor 6/10. Suspeita de SCA.",
        "Dor toracica atipica, EVA 8/10, sem alteracoes hemodinamicas. Paciente ansiosa.",
        "Dor precordial ha 3 horas, intermitente, em queimacao, com piora aos esforcos. PA estavel."],
     "historico":["HAS, DM2, dislipidemia, ex-tabagista. Pai com IAM precoce.",
                  "Hipertenso em uso de losartana, dislipidemico.",
                  "Sem fatores de risco conhecidos.",
                  "Coronariopata, angioplastia ha 6 anos, em uso de AAS e estatina."]},
    # Dispneia moderada
    {"vit": (50,90, 100,160, 95,130, 90,94, 36.5,38.5), "sex":"X",
     "sintomas":[
        "Dispneia progressiva ha 2 dias, hoje fala entrecortada, uso de musculatura acessoria. Sibilos difusos. Saturacao caindo.",
        "Falta de ar ha 4 horas em paciente DPOC, tiragem intercostal, conseguindo formular frases curtas.",
        "Dispneia importante, taquipneico (28 irpm), saturacao 92%, edema de MMII bilateral. Suspeita de IC descompensada.",
        "Crise asmatica moderada, sibilancia audivel a distancia, uso de musculatura acessoria, saturacao 91%."],
     "historico":["DPOC GOLD III em uso de tiotropio. Tabagista 50 macos/ano.",
                  "Asma desde a infancia, varias internacoes previas.",
                  "Insuficiencia cardiaca CF II em uso de carvedilol, espironolactona, furosemida.",
                  "Hipertenso, ex-tabagista."]},
    # Dor severa
    {"vit": (20,75, 115,165, 95,125, 95,99, 36.5,37.5), "sex":"X",
     "sintomas":[
        "Colica nefretica intensa, EVA 10/10, irradiando para flanco e regiao inguinal. Nauseas e vomitos. Inquieto.",
        "Lombalgia aguda intensa apos esforco, EVA 9/10, sem irradiacao para MMII, sem deficit neurologico.",
        "Dor abdominal em FID intensa ha 6 horas, EVA 8/10, com defesa local. Suspeita de apendicite aguda.",
        "Cefaleia intensa em trovao, EVA 10/10, de inicio subito ha 1 hora. Pior dor da vida."],
     "historico":["Episodio previo de calculo renal ha 2 anos.",
                  "Sem comorbidades.",
                  "Hipertensa em uso de hidroclorotiazida.",
                  "Historico de enxaqueca, mas refere ser dor diferente."]},
    # Hipertensao sintomatica
    {"vit": (50,85, 190,240, 75,105, 95,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Cefaleia intensa, alteracao visual (escotomas), formigamento em hemiface direita ha 3 horas. PA persistentemente elevada.",
        "Crise hipertensiva sintomatica com confusao mental leve, nauseas e cefaleia occipital.",
        "Epistaxe abundante, cefaleia intensa, PA muito elevada na chegada. Sem deficit neurologico focal.",
        "PA muito elevada com dor toracica leve e dispneia aos pequenos esforcos, sugerindo emergencia hipertensiva."],
     "historico":["HAS de longa data, em uso irregular de losartana e anlodipino.",
                  "Hipertensa, sem aderencia ao tratamento.",
                  "Diabetico, hipertenso, em uso de multiplos anti-hipertensivos.",
                  "Insuficiencia renal cronica em conservador."]},
    # Bradicardia/taquicardia
    {"vit": (40,85, 95,145, 150,200, 93,98, 36.2,37.2), "sex":"X",
     "sintomas":[
        "Palpitacoes intensas ha 1 hora, sensacao de falta de ar leve, tontura. Refere FC muito elevada ao palpar pulso.",
        "Taquicardia supraventricular sintomatica, dispneia leve, dor precordial atipica.",
        "Sincope apos sensacao de palpitacao rapida, recuperou consciencia. Mantem FC muito alta no monitor.",
        "Episodio de palpitacoes ha 2 horas, fraqueza generalizada, FC 180 ao monitor."],
     "historico":["FA paroxistica conhecida, em uso de propranolol.",
                  "Cardiopata isquemico, marca-passo definitivo ha 3 anos.",
                  "Sem comorbidades cardiologicas conhecidas.",
                  "Hipertireoidismo em investigacao."]},
    # Hipertermia severa
    {"vit": (2,85, 95,135, 110,145, 92,98, 39.6,41.5), "sex":"X",
     "sintomas":[
        "Febre alta (40C) ha 6 horas refrataria a antitermicos. Cefaleia intensa, mialgia generalizada, prostracao.",
        "Hipertermia severa (39.8C) em crianca de 2 anos, irritada, com gemencia.",
        "Febre persistente acima de 40C ha 24 horas, calafrios, sudorese profusa, mialgia importante.",
        "Idosa com febre 39.7C, confusa, desidratada. Sem foco aparente identificado."],
     "historico":["Sem comorbidades conhecidas.",
                  "Imunossuprimida em uso de metotrexato por artrite reumatoide.",
                  "Diabetico tipo 2, em uso de metformina.",
                  "Idosa institucionalizada, demencia avancada."]},
    # Gestante alto risco
    {"vit": (16,42, 95,165, 85,120, 94,99, 36.3,37.5), "sex":"F",
     "sintomas":[
        "Gestante 32 semanas com sangramento vaginal volumoso ha 2 horas. Refere movimentos fetais reduzidos hoje.",
        "Gestante a termo com dor abdominal intensa em baixo ventre, sangramento vaginal moderado, ausencia de movimentacao fetal nas ultimas 6 horas.",
        "Cefaleia intensa, escotomas e epigastralgia em gestante de 36 semanas com PA elevada. Suspeita de pre-eclampsia grave.",
        "Gestante 28 semanas com perda de liquido amniotico ha 4 horas, contracoes regulares."],
     "historico":["Primigesta, pre-natal regular, sem intercorrencias previas.",
                  "G3P2A0, ultimo parto cesarea ha 2 anos.",
                  "Hipertensao gestacional em uso de metildopa.",
                  "Diabetes gestacional em controle dietetico."]},
    # Abdome agudo
    {"vit": (30,80, 95,145, 95,130, 94,99, 37.5,38.8), "sex":"X",
     "sintomas":[
        "Dor abdominal difusa intensa ha 8 horas, EVA 8/10, com piora progressiva. Abdome em tabua, peritonite. Vomitos biliosos.",
        "Dor abdominal subita em punhalada em flanco esquerdo, irradiando para dorso, palidez, taquicardia.",
        "Obstrucao intestinal: distensao abdominal, vomitos fecaloides, ausencia de eliminacao de gases ha 36h.",
        "Isquemia mesenterica suspeita: dor abdominal desproporcional ao exame, idoso com FA nao anticoagulada."],
     "historico":["Hipertenso, dislipidemico, ex-tabagista.",
                  "FA cronica nao anticoagulada por escolha do paciente.",
                  "Cirurgias abdominais previas (apendicectomia, colecistectomia).",
                  "Doenca diverticular conhecida."]},
    # Crise psiquiatrica
    {"vit": (16,60, 110,165, 95,130, 95,99, 36.5,37.4), "sex":"X",
     "sintomas":[
        "Surto psicotico agudo com agitacao psicomotora intensa, heteroagressividade, alucinacoes auditivas. Risco para si e terceiros.",
        "Tentativa de autoexterminio por intoxicacao medicamentosa ha 1 hora. Sonolento, vias aereas pervias.",
        "Crise de panico severa com despersonalizacao, dispneia, parestesias. Refere ideacao suicida.",
        "Episodio maniaco com agressividade, fuga de ideias, sem dormir ha 3 dias, ameacando familiares."],
     "historico":["Esquizofrenia em uso irregular de risperidona.",
                  "Transtorno bipolar tipo I, varias internacoes psiquiatricas.",
                  "Depressao em uso de fluoxetina, com ideacao suicida previa.",
                  "Primeiro episodio psiquiatrico documentado."]},
    # AVC subagudo
    {"vit": (55,88, 150,210, 70,105, 94,98, 36.2,37.4), "sex":"X",
     "sintomas":[
        "Hemiparesia direita iniciada ha 12 horas, fora de janela para tromboliise. Disartria leve, alerta. NIHSS 6.",
        "Paciente acordou com deficit de forca em MSE ha 8 horas. AVCi em territorio de ACM esquerda, fora de janela.",
        "TIA com hemiparesia transitoria que reverteu em 30 minutos, ocorrida ha 5 horas. Sem deficit residual.",
        "Hemiparesia leve em MID iniciada ha 18 horas, paciente assintomatico, sem novos eventos."],
     "historico":["HAS, FA paroxistica, dislipidemia, em uso de varfarina.",
                  "Diabetico, hipertenso, AVC previo com sequela motora leve.",
                  "Tabagista pesado, em uso de AAS profilatico.",
                  "Hipertensao grave, sem aderencia."]},
    # Fratura exposta (NOVO)
    {"vit": (15,75, 105,150, 95,125, 95,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Fratura exposta de tibia direita apos queda de moto ha 30 minutos, deformidade evidente, sangramento ativo, dor EVA 9/10.",
        "Luxacao de quadril esquerdo apos atropelamento, dor severa EVA 10/10, deformidade visivel, paciente nao consegue mover o membro.",
        "Fratura exposta de antebraco com exposicao ossea, sangramento ativo, dor severa, palido e sudoreico."],
     "historico":["Sem comorbidades. Motociclista profissional.",
                  "Osteoporose em uso de bisfosfonato.",
                  "Etilista cronico."]},
    # Gravidez ectopica (NOVO)
    {"vit": (18,42, 95,130, 95,130, 95,99, 36.4,37.4), "sex":"F",
     "sintomas":[
        "Mulher em idade fertil com dor pelvica severa ha 4 horas, EVA 9/10, sangramento vaginal moderado, atraso menstrual de 6 semanas. Suspeita de gravidez ectopica rota.",
        "Dor abdominal aguda em fossa iliaca direita em mulher jovem, sangramento vaginal escuro, lipotimia. Beta-HCG positivo previamente.",
        "Mulher com dor pelvica intensa, palidez, hipotensao postural, refere atraso menstrual."],
     "historico":["DIP previa, sexualmente ativa sem metodo contraceptivo.",
                  "Cirurgia de trompas ha 3 anos.",
                  "Sem comorbidades. G2P0A2."]},
    # Intoxicacao grave (NOVO)
    {"vit": (15,65, 85,140, 50,130, 88,97, 35.5,38.0), "sex":"X",
     "sintomas":[
        "Intoxicacao por organofosforado ha 2 horas (tentativa de auto-exterminio), miose, sialorreia, sudorese, fasciculacoes, bradicardia.",
        "Intoxicacao por benzodiazepinicos, sonolento mas responsivo, depressao respiratoria leve, hipotensao.",
        "Abuso de cocaina ha 1 hora com dor toracica, taquicardia, hipertensao, agitacao psicomotora."],
     "historico":["Depressao em tratamento, multiplas tentativas previas.",
                  "Trabalhador rural, contato com agrotoxicos.",
                  "Usuario de drogas iliticitas conhecido."]},
    # Pediatrico febre alta (NOVO)
    {"vit": (0,5, 85,110, 140,180, 92,98, 39.0,40.5), "sex":"X",
     "sintomas":[
        "Lactente de 8 meses com febre 39.5C ha 24 horas, irritabilidade, recusa alimentar, gemencia. Mae refere choro sem lagrimas e fralda seca ha 6 horas.",
        "Crianca de 2 meses com febre 38.8C, hipoativa, peristalse aumentada, vomitos. Idade < 3 meses, qualquer febre exige avaliacao urgente.",
        "Bebe de 18 meses com febre alta sustentada, prostrado, mae refere convulsao febril ha 1 hora, agora pos-ictal."],
     "historico":["Sem comorbidades. Vacinacao em dia.",
                  "Prematuro, RN de 32 semanas.",
                  "Asmatico em tratamento."]},
]

AMARELO = [
    # Dor moderada
    {"vit": (18,75, 110,145, 75,100, 96,99, 36.4,37.3), "sex":"X",
     "sintomas":[
        "Lombalgia moderada apos esforco fisico ha 2 dias, EVA 5/10, sem irradiacao para MMII, sem alteracao de forca.",
        "Cefaleia tensional moderada ha 1 dia, EVA 5/10, continua, sem alteracoes visuais ou neurologicas.",
        "Dor em joelho direito apos queda ha 12 horas, edema moderado, dificuldade para deambular, EVA 6/10.",
        "Otalgia intensa em ouvido esquerdo ha 24 horas, EVA 6/10, com hipoacusia. Sem febre.",
        "Cervicalgia moderada com limitacao de movimento ha 3 dias, EVA 5/10, sem irradiacao."],
     "historico":["Sem comorbidades.",
                  "Hipertenso em uso de losartana.",
                  "Historico de lombalgia cronica.",
                  "Diabetico tipo 2 controlado."]},
    # Febre alta sem gravidade
    {"vit": (18,80, 105,140, 95,120, 95,99, 39.0,39.5), "sex":"X",
     "sintomas":[
        "Febre alta (39.2C) ha 24 horas, calafrios, mialgia, cefaleia. Sem foco aparente. Estado geral preservado.",
        "Quadro gripal com febre 39C ha 2 dias, tosse, coriza, dor de garganta. Sem dispneia.",
        "Febre persistente 39.3C ha 36 horas com disuria e polaciuria. Suspeita de ITU.",
        "Febre 39.1C com cefaleia retro-orbitaria, dor articular intensa, exantema discreto. Vindo de area endemica."],
     "historico":["Sem comorbidades.",
                  "Hipertenso, diabetico tipo 2.",
                  "ITU de repeticao.",
                  "Sem comorbidades. Vacinacao em dia."]},
    # Vomitos persistentes
    {"vit": (15,80, 95,135, 85,115, 96,99, 36.5,38.4), "sex":"X",
     "sintomas":[
        "Vomitos persistentes ha 12 horas, mais de 10 episodios, associados a diarreia. Desidratacao leve a moderada, mucosas secas.",
        "Emese persistente ha 24 horas em diabetico, dor abdominal moderada, refere sede intensa.",
        "Vomitos incoerciveis em gestante de 1o trimestre, perda ponderal de 3 kg em 1 semana, sinais de desidratacao.",
        "Gastroenterite com vomitos volumosos ha 8 horas, diarreia aquosa, dor abdominal em colica."],
     "historico":["DM2 em uso de insulina.",
                  "Sem comorbidades.",
                  "Gastrite cronica em uso eventual de omeprazol.",
                  "Gestante de 10 semanas, primeira gestacao."]},
    # Dor abdominal moderada
    {"vit": (18,75, 105,140, 80,110, 96,99, 36.5,38.0), "sex":"X",
     "sintomas":[
        "Dor abdominal em mesogastrio ha 18 horas, continua, EVA 6/10, sem sinais de peritonite. Nauseas, sem vomitos.",
        "Dor em hipocondrio direito ha 6 horas em pos-prandial gorduroso, EVA 6/10, irradiacao para dorso.",
        "Dor pelvica em mulher jovem, moderada, ha 24 horas, ciclo menstrual atrasado. Sem febre.",
        "Disuria, dor lombar leve, polaciuria ha 2 dias, urina turva e fetida. Sem febre alta."],
     "historico":["Sem comorbidades.",
                  "Colelitiase conhecida em fila de cirurgia.",
                  "Sexualmente ativa, sem metodo contraceptivo.",
                  "ITU de repeticao."]},
    # Dispneia leve
    {"vit": (40,85, 110,145, 85,110, 95,97, 36.5,38.0), "sex":"X",
     "sintomas":[
        "Dispneia aos medios esforcos ha 2 dias, sem dispneia em repouso. Tosse seca leve. Sem dor toracica.",
        "Falta de ar leve com sibilancia intermitente, asmatico em descompensacao leve.",
        "Cansaco aos esforcos, ortopneia ha 3 dias, edema discreto em MMII. Suspeita de IC descompensada.",
        "Tosse produtiva ha 1 semana com expectoracao amarelada, dispneia leve, febre baixa."],
     "historico":["DPOC GOLD II, ex-tabagista.",
                  "Asma intermitente em uso eventual de salbutamol.",
                  "IC CF II em uso de losartana, espironolactona.",
                  "Hipertenso em uso de IECA."]},
    # Hipertensao assintomatica alta
    {"vit": (45,85, 165,215, 75,100, 96,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "PA aferida em casa elevada (180/110) ha 2 horas, refere apenas leve dor de cabeca. Sem alteracoes visuais ou neurologicas.",
        "Hipertensao sistolica isolada (sis 195) detectada em consulta de rotina, assintomatico.",
        "Cefaleia leve ha 1 dia, PA aferida 175/105 em casa, fez uso de AINE recentemente.",
        "Tontura ao mudar de posicao, PA 185/100 ao chegar. Sem outros sintomas."],
     "historico":["HAS em uso irregular de losartana.",
                  "Hipertenso recem-diagnosticado, sem tratamento.",
                  "DM2, hipertenso, em uso de multiplas medicacoes.",
                  "Hipertensa em uso de hidroclorotiazida ha anos."]},
    # Cefaleia moderada
    {"vit": (15,70, 110,150, 75,100, 96,99, 36.4,37.3), "sex":"F",
     "sintomas":[
        "Cefaleia hemicraniana ha 6 horas, pulsatil, EVA 6/10, com fotofobia e fonofobia. Padrao habitual de enxaqueca.",
        "Dor de cabeca em peso ha 2 dias, continua, EVA 5/10, alivia parcialmente com analgesico simples.",
        "Cefaleia frontal e periorbitaria ha 3 dias com obstrucao nasal e dor a palpacao dos seios da face. Suspeita de sinusite.",
        "Cefaleia tensional moderada ha 24 horas, EVA 5/10, sem alteracoes neurologicas."],
     "historico":["Historico de enxaqueca em uso de propranolol profilatico.",
                  "Sem comorbidades.",
                  "Sinusopatia cronica.",
                  "Hipertensa em uso de losartana."]},
    # ITU complicada
    {"vit": (18,80, 105,140, 85,115, 96,99, 38.0,39.0), "sex":"F",
     "sintomas":[
        "Disuria, polaciuria, dor lombar a direita, febre 38.5C ha 24 horas. Suspeita de pielonefrite, sem instabilidade.",
        "Dor em flanco direito ha 2 dias, urgencia miccional, urina turva, febre 38.3C.",
        "Pielonefrite recorrente, queixas urinarias tipicas + febre 38C, sem confusao ou hipotensao.",
        "ITU complicada em diabetico com queixas urinarias e febre 38.7C, hidratacao preservada."],
     "historico":["ITU de repeticao, ultimo episodio ha 3 meses.",
                  "DM2 descompensado, ITU recorrente.",
                  "Calculo renal direito conhecido.",
                  "Cateter vesical de demora ha 1 mes."]},
    # Trauma fechado simples
    {"vit": (15,75, 110,145, 80,110, 96,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Trauma em punho esquerdo apos queda da propria altura ha 2 horas. Edema moderado, dor 6/10, deformidade discreta.",
        "Entorse de tornozelo direito ha 4 horas apos acidente desportivo, edema importante, dor a palpacao maleolar lateral.",
        "Contusao em hemitorax esquerdo apos trauma em pancada lateral, dor a respiracao profunda, sem dispneia.",
        "Fratura de antebraco fechada apos queda, deformidade evidente, dor controlada com analgesico oral."],
     "historico":["Sem comorbidades.",
                  "Osteoporose em uso de calcio e alendronato.",
                  "Sem comorbidades. Atleta amador.",
                  "Hipertenso em uso de IECA."]},
    # Hiperglicemia moderada
    {"vit": (35,80, 105,145, 85,115, 96,99, 36.5,37.3), "sex":"X",
     "sintomas":[
        "Diabetico referindo poliuria, polidipsia, fraqueza ha 24 horas. HGT em casa 380 mg/dL.",
        "Glicemia muito elevada (HGT > 400) detectada na chegada, lucido, sem sinais de cetoacidose franca.",
        "Hiperglicemia sintomatica com poliuria intensa e perda ponderal de 4 kg em 1 mes.",
        "Diabetico descompensado com queixas inespecificas, glicemia em torno de 350 mg/dL."],
     "historico":["DM2 em uso irregular de metformina.",
                  "DM1 desde a infancia, em uso de insulina basal-bolus.",
                  "DM recem-diagnosticado, sem tratamento iniciado.",
                  "DM2 em uso de glibenclamida e metformina, mau controle."]},
    # Crise asmatica leve
    {"vit": (8,60, 105,140, 95,120, 94,97, 36.4,37.5), "sex":"X",
     "sintomas":[
        "Crise de asma leve ha 2 horas, sibilancia intermitente, dispneia leve, conseguindo formular frases longas.",
        "Sibilancia e tosse seca ha 1 dia em asmatico, sem uso de musculatura acessoria, saturacao 96%.",
        "Bronquite aguda em asmatico, dispneia leve a moderada, expectoracao mucoide.",
        "Asma com piora ha 3 dias, broncodilatador de resgate sem efeito completo, sem instabilidade."],
     "historico":["Asma persistente leve em uso de budesonida.",
                  "Asma episodica desde a infancia.",
                  "Asma e rinite alergica.",
                  "Asma em uso de salbutamol PRN."]},
    # Diarreia com desidratacao
    {"vit": (2,80, 95,130, 95,120, 96,99, 36.5,38.5), "sex":"X",
     "sintomas":[
        "Diarreia aquosa ha 24 horas, mais de 10 evacuacoes, vomitos associados, mucosas secas, oliguric o.",
        "Gastroenterite com diarreia profusa ha 2 dias, dor abdominal em colica, desidratacao moderada.",
        "Crianca de 5 anos com diarreia ha 3 dias, perdeu peso visivelmente, mucosas ressecadas.",
        "Idoso com diarreia e vomitos ha 24 horas, hipotensao postural, fraqueza importante."],
     "historico":["Sem comorbidades.",
                  "Idoso institucionalizado.",
                  "Gastrite cronica.",
                  "Imunossuprimido leve por uso de corticoide inalatorio."]},
    # Otite media aguda (NOVO)
    {"vit": (2,60, 105,135, 85,115, 96,99, 38.0,38.8), "sex":"X",
     "sintomas":[
        "Otalgia intensa em ouvido direito EVA 7/10 ha 36 horas, febre 38.5C, hipoacusia, otorreia purulenta.",
        "Crianca de 4 anos com otalgia, choro intenso, puxando orelha esquerda, febre 38.3C, otorreia ha 6 horas.",
        "Adulto com otalgia severa pulsatil, febre 38.6C, otorreia purulenta, vertigem leve associada."],
     "historico":["Sem comorbidades. Otite previa ha 6 meses.",
                  "Crianca com otites de repeticao.",
                  "Diabetico tipo 2."]},
    # Crise renal calculo (NOVO)
    {"vit": (25,65, 115,155, 90,115, 96,99, 36.4,37.5), "sex":"X",
     "sintomas":[
        "Colica nefretica moderada ha 2 horas, EVA 7/10, irradiando para regiao inguinal, urina rosada. Sem febre.",
        "Dor lombar em peso a direita ha 6 horas, hematuria macroscopica, polaciuria, sem disuria importante.",
        "Quadro de calculo renal previo, agora com dor moderada EVA 6, nauseas, urina turva."],
     "historico":["Calculo renal previo ha 2 anos, mesmo lado.",
                  "Sem comorbidades.",
                  "Hiperuricemico em tratamento."]},
    # Hemorroidas sangramento (NOVO)
    {"vit": (30,75, 105,140, 80,105, 96,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Sangramento retal vivo ha 3 dias com a evacuacao, hemorroida externa visivel, dor moderada na regiao anal.",
        "Hematoquesia leve a moderada ha 24 horas, dor anal severa EVA 6/10, dificuldade para sentar.",
        "Trombose hemorroidaria com dor intensa EVA 7, paciente nao consegue evacuar."],
     "historico":["Hemorroidas conhecidas, primeiro episodio de sangramento.",
                  "Constipacao cronica.",
                  "Gestante de 32 semanas."]},
    # Laceracao simples (NOVO)
    {"vit": (10,70, 110,145, 80,105, 96,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Laceracao em mao esquerda apos corte com faca de cozinha ha 30 minutos, sangramento controlado, ferida de 4cm com bordas regulares, necessita sutura.",
        "Ferimento corto-contuso em couro cabeludo apos queda, ferida de 6cm sangrando moderadamente.",
        "Laceracao profunda em coxa direita apos acidente domestico, exposicao subcutanea, sangramento moderado."],
     "historico":["Sem comorbidades. Vacinacao antitetanica em dia.",
                  "Hipertenso em uso de losartana.",
                  "Sem comorbidades. Vacina antitetanica vencida."]},
]

VERDE = [
    # Dor leve
    {"vit": (15,75, 110,140, 70,95, 97,99, 36.3,37.0), "sex":"X",
     "sintomas":[
        "Lombalgia leve ha 2 dias apos esforco fisico, EVA 3/10, melhora com analgesico simples. Sem irradiacao.",
        "Cefaleia leve ha 1 dia, EVA 2/10, continua, conseguiu trabalhar normalmente.",
        "Dor em punho direito apos pequena torcao ha 1 dia, edema discreto, EVA 3/10. Movimentacao preservada.",
        "Mialgia em panturrilhas apos atividade fisica intensa ha 2 dias, EVA 2/10, sem outros sintomas.",
        "Dor leve em cotovelo direito por uso repetitivo (epicondilite), EVA 3/10, ha 1 semana."],
     "historico":["Sem comorbidades.","Sedentario.","Pratica musculacao regularmente.","Hipertenso controlado."]},
    # IVAS
    {"vit": (2,80, 108,135, 72,95, 97,99, 36.5,38.4), "sex":"X",
     "sintomas":[
        "Resfriado ha 3 dias com coriza, espirros, dor de garganta leve. Febre baixa (37.8C). Sem dispneia.",
        "Tosse seca ha 5 dias com odinofagia leve, sem febre alta, sem falta de ar.",
        "Quadro gripal com mialgia leve, mal-estar e febre baixa, ha 2 dias. Apetite preservado.",
        "Faringite viral, dor para deglutir leve, hiperemia de orofaringe. Febre 37.8C.",
        "Rinorreia clara abundante, espirros em salva, congestao nasal ha 2 dias. Sem febre."],
     "historico":["Sem comorbidades.","Rinite alergica.","Sem comorbidades. Vacinacao em dia.","Historico de quadros gripais frequentes."]},
    # Dor muscular
    {"vit": (15,70, 110,140, 70,95, 97,99, 36.3,37.0), "sex":"X",
     "sintomas":[
        "Contusao em coxa direita apos pancada em pratica esportiva ha 6 horas. Edema discreto, hematoma em formacao.",
        "Entorse leve de tornozelo direito ha 1 dia, edema discreto, dor a movimentacao, deambula com leve claudicacao.",
        "Cervicalgia leve por ma postura no trabalho ha 1 semana, sem irradiacao, sem alteracoes neurologicas.",
        "Tendinite no ombro direito por uso repetitivo, dor leve a movimentacao, sem limitacao funcional importante."],
     "historico":["Sem comorbidades. Pratica esportes regularmente.","Sedentario.","Trabalho bracal.","Sem comorbidades."]},
    # Diarreia leve
    {"vit": (5,75, 110,135, 72,95, 97,99, 36.4,37.5), "sex":"X",
     "sintomas":[
        "Diarreia 3 a 4 vezes ao dia ha 24 horas, sem sangue, sem febre. Hidratacao preservada.",
        "Episodios diarreicos ha 2 dias, autolimitados, sem desidratacao. Refere ter comido alimento suspeito.",
        "Diarreia leve com colica abdominal de baixa intensidade ha 36 horas. Continua se alimentando."],
     "historico":["Sem comorbidades.","Sindrome do intestino irritavel conhecida.","Sem comorbidades."]},
    # Conjuntivite
    {"vit": (5,70, 110,135, 70,92, 97,99, 36.3,37.0), "sex":"X",
     "sintomas":[
        "Olho vermelho bilateral ha 2 dias com secrecao amarelada matinal, prurido leve. Sem alteracoes visuais.",
        "Hiperemia conjuntival e secrecao em olho direito ha 3 dias, lacrimejamento abundante. Visao preservada.",
        "Conjuntivite viral suspeita, olhos vermelhos, prurido intenso, secrecao aquosa, contato com caso similar em casa."],
     "historico":["Sem comorbidades. Conjuntivites recorrentes.","Rinite alergica.","Sem comorbidades."]},
    # Dermatite alergica
    {"vit": (5,75, 110,140, 72,95, 97,99, 36.4,37.0), "sex":"X",
     "sintomas":[
        "Lesoes eritematosas e prurido em regiao cervical ha 3 dias apos uso de bijuteria nova. Sem sintomas sistemicos.",
        "Erupcao cutanea em tronco e MMSS, prurido moderado, sem dispneia, sem edema facial. Inicio apos uso de antibiotico ha 2 dias.",
        "Urticaria recorrente leve em MMII ha 1 semana, sem outros sintomas. Pruriginosa.",
        "Eczema em maos por contato com produto de limpeza, descamacao e prurido leves ha 5 dias."],
     "historico":["Atopia conhecida.","Sem comorbidades. Possivel alergia em investigacao.","Asma e rinite alergica.","Sem comorbidades."]},
    # Odontalgia
    {"vit": (15,75, 110,140, 75,98, 97,99, 36.4,37.3), "sex":"X",
     "sintomas":[
        "Dor de dente intensa ha 2 dias, EVA 6/10, em molar inferior esquerdo. Sem febre, sem trismo importante.",
        "Odontalgia em incisivo superior direito, sensibilidade ao frio, ha 4 dias, EVA 5/10.",
        "Dor dentaria apos procedimento odontologico ha 24 horas, EVA 5/10, edema gengival local.",
        "Pericoronarite em terceiro molar inferior, dor moderada, dificuldade para abrir a boca."],
     "historico":["Sem acompanhamento odontologico regular ha anos.","Sem comorbidades.","Bruxismo conhecido.","Sem comorbidades."]},
    # ITU simples
    {"vit": (18,70, 108,135, 72,95, 97,99, 36.4,37.5), "sex":"F",
     "sintomas":[
        "Disuria e polaciuria ha 24 horas, sem febre, sem dor lombar. Urina sem alteracao macroscopica importante.",
        "Queixas urinarias ha 2 dias (ardencia ao urinar, urgencia), sem febre, sem dor lombar. Sem nauseas.",
        "Cistite simples, disuria leve, polaciuria, urina turva. Sem sintomas sistemicos."],
     "historico":["ITU de repeticao leve.","Sem comorbidades.","Diabetica compensada."]},
    # Gastrite
    {"vit": (18,75, 108,138, 72,95, 97,99, 36.3,37.0), "sex":"X",
     "sintomas":[
        "Dor epigastrica em queimacao ha 3 dias, piora com alimentacao, EVA 4/10. Sem vomitos, sem sangramento.",
        "Pirose e regurgitacao acida ha 1 semana, alivio parcial com antiacido. Sem disfagia.",
        "Dor em queimacao retroesternal pos-prandial ha varios dias, sem dor toracica isquemica.",
        "Quadro dispeptico com plenitude pos-prandial, eructacoes, nauseas leves ha 5 dias."],
     "historico":["Gastrite cronica em uso eventual de omeprazol.","Etilista social. Tabagista.","Sem comorbidades.","DRGE conhecida."]},
    # Dermato leve
    {"vit": (2,80, 108,138, 72,92, 97,99, 36.3,37.4), "sex":"X",
     "sintomas":[
        "Lesao circular descamativa pruriginosa em antebraco direito ha 1 semana, suspeita de tinea corporis.",
        "Acne inflamatoria moderada em face, com algumas lesoes pustulosas, sem sinais de infeccao sistemica.",
        "Pequena ferida em pe direito ha 5 dias, lenta cicatrizacao, sem sinais flogisticos importantes.",
        "Ressecamento e prurido em pernas ha 2 semanas, suspeita de eczema atopico."],
     "historico":["Sem comorbidades.","Diabetico tipo 2 com cuidado moderado dos pes.","Sem comorbidades.","Atopia conhecida."]},
    # Agudizacao cronica leve
    {"vit": (35,85, 110,145, 72,98, 96,99, 36.4,37.2), "sex":"X",
     "sintomas":[
        "Piora discreta da dor cronica em joelho direito ha 3 dias, EVA 4/10. Osteoartrose conhecida.",
        "Cefaleia tensional habitual mais frequente nos ultimos dias, intensidade leve, EVA 3/10.",
        "Constipacao intestinal habitual com piora ha 1 semana, sem sangramento, sem dor importante.",
        "Tontura postural leve em hipertensa que ajustou medicacao recentemente."],
     "historico":["Osteoartrose poliarticular.","Cefaleia tensional cronica.","Constipacao cronica.","Hipertensa em ajuste de medicacao."]},
    # Otalgia leve
    {"vit": (2,70, 108,138, 72,95, 97,99, 36.4,37.6), "sex":"X",
     "sintomas":[
        "Otalgia leve a direita ha 2 dias, sem febre alta, sem otorreia. Historico recente de IVAS.",
        "Sensacao de ouvido tampado ha 3 dias apos resfriado, hipoacusia leve, sem dor importante.",
        "Otite externa por contato com agua de piscina ha 4 dias, prurido e dor leve.",
        "Tampao de cera com sensacao de plenitude e leve hipoacusia ha 1 semana."],
     "historico":["Otites recorrentes na infancia.","Sem comorbidades.","Frequentador de piscina.","Sem comorbidades."]},
    # Picada de inseto (NOVO)
    {"vit": (5,70, 110,138, 75,95, 97,99, 36.3,37.2), "sex":"X",
     "sintomas":[
        "Picada de mosquito em MMII com area eritematosa de 3cm, prurido intenso, ha 1 dia. Sem febre, sem dispneia.",
        "Picada de abelha em antebraco ha 2 horas, edema local, dor leve, sem reacao sistemica. Ferrao retirado.",
        "Multiplas picadas de pernilongo em pernas ha 3 dias, prurido intenso, lesoes papulares disseminadas."],
     "historico":["Sem comorbidades.","Atopia conhecida.","Picadas previas sem reacoes severas."]},
    # Queimadura solar (NOVO)
    {"vit": (5,70, 110,138, 72,95, 97,99, 36.4,37.3), "sex":"X",
     "sintomas":[
        "Queimadura solar em ombros e dorso ha 1 dia apos exposicao prolongada. Eritema, dor leve EVA 3/10, sem bolhas significativas.",
        "Queimadura de 1o grau em face e MMSS apos sol intenso, ardor, sem comprometimento sistemico.",
        "Eritema solar generalizado em tronco, prurido leve, sem flictenas importantes."],
     "historico":["Sem comorbidades. Pele clara fototipo II.","Sem comorbidades.","Em uso de isotretinoina (acne)."]},
    # Tosse alergica (NOVO)
    {"vit": (5,75, 108,135, 72,95, 96,99, 36.3,37.2), "sex":"X",
     "sintomas":[
        "Tosse seca persistente ha 1 semana, pior a noite, sem febre, sem expectoracao. Coriza clara associada. Suspeita de rinite alergica.",
        "Espirros em salva, prurido nasal, congestao ha 5 dias. Tosse seca esporadica.",
        "Pigarro e tosse seca matinal ha 2 semanas em alergico conhecido. Sem dispneia, sem febre."],
     "historico":["Rinite alergica em uso eventual de loratadina.","Atopia desde a infancia.","Asma alergica em controle."]},
    # Ferida pequena (NOVO)
    {"vit": (15,70, 110,140, 72,92, 97,99, 36.3,37.0), "sex":"X",
     "sintomas":[
        "Pequena escoriacao em joelho direito apos queda na rua, ferida superficial de 2cm sem sangramento ativo.",
        "Cortes superficiais em mao por acidente domestico, sangramento minimo, ferida limpa, dor leve.",
        "Abrasao em cotovelo apos quedinha em bicicleta, sem profundidade, sem comprometimento de estruturas."],
     "historico":["Sem comorbidades. Vacina antitetanica em dia.","Diabetico bem controlado.","Sem comorbidades."]},
]

AZUL = [
    # Renovacao receita
    {"vit": (40,85, 115,140, 68,88, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece para renovacao de receita de medicacao continua de uso cronico. Assintomatico.",
        "Pedido de receita para anti-hipertensivo de uso habitual. Refere medicacao acabou ontem. Sem sintomas.",
        "Solicita renovacao de receita controlada de antidepressivo de uso ha 3 anos, em acompanhamento estavel.",
        "Comparece para receita de medicacao para diabetes, em uso regular, assintomatico.",
        "Veio buscar receita de hormonio de reposicao tireoidiana, em uso ha 10 anos.",
        "Renovacao de receita de medicacao para hipotireoidismo, sem queixas atuais."],
     "historico":["HAS em uso de losartana.","DM2 em uso de metformina.","Depressao em uso de sertralina.","Hipotireoidismo em uso de levotiroxina."]},
    # Resultado de exame
    {"vit": (20,80, 115,140, 68,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece para mostrar resultado de exames laboratoriais solicitados na ultima consulta. Assintomatico.",
        "Trouxe resultado de ultrassom para avaliacao. Sem queixas atuais.",
        "Solicita interpretacao de exame de imagem realizado ha 2 dias. Sem sintomas agudos.",
        "Veio buscar resultado de biopsia. Sem queixas no momento.",
        "Comparece com resultado de hemograma de rotina, totalmente assintomatico."],
     "historico":["Acompanhamento para investigacao de massa abdominal.","Cardiopata em acompanhamento ambulatorial.","Diabetico em acompanhamento.","Sem comorbidades."]},
    # Atestado
    {"vit": (18,65, 115,138, 68,88, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece solicitando atestado medico para justificar falta no trabalho de ontem. Assintomatico.",
        "Pede declaracao de comparecimento. Sem queixas clinicas.",
        "Solicita atestado para academia/atividade fisica. Sem queixas.",
        "Veio buscar atestado de saude para escola. Sem sintomas.",
        "Solicita declaracao de aptidao fisica para concurso publico. Assintomatico."],
     "historico":["Sem comorbidades.","Hipertensa controlada.","Sem comorbidades.","Diabetico compensado."]},
    # Curativo eletivo
    {"vit": (20,85, 115,138, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece para troca de curativo programado de ferida operatoria de cirurgia ha 7 dias. Sem sinais flogisticos.",
        "Curativo de ulcera venosa em MIE, sem piora, sem secrecao purulenta. Programado.",
        "Retirada de pontos de sutura realizada ha 10 dias. Cicatrizacao adequada.",
        "Curativo eletivo de cateter venoso central, sem sinais de infeccao."],
     "historico":["Pos-operatorio de colecistectomia.","Insuficiencia venosa cronica.","Pos-operatorio de pequena cirurgia ambulatorial.","Em quimioterapia, com cateter de longa permanencia."]},
    # Vacinacao orientacao
    {"vit": (2,80, 115,138, 70,92, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Veio para vacinacao de rotina conforme calendario. Assintomatica.",
        "Solicita orientacao sobre prevencao de doencas. Sem queixas atuais.",
        "Pede orientacao sobre uso de medicacao prescrita em outro servico. Sem sintomas.",
        "Comparece para imunizacao contra influenza, conforme campanha. Saudavel."],
     "historico":["Sem comorbidades. Vacinacao em dia.","Idoso, dentro de grupo prioritario para vacinacao.","Profissional da saude.","Gestante de 24 semanas, vacinacao programada."]},
    # Aferir PA rotina
    {"vit": (40,80, 115,145, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece apenas para aferir pressao arterial, conforme orientacao medica anterior. Sem sintomas.",
        "Quer verificar PA porque esta sem aparelho em casa. Assintomatico.",
        "Afericao de PA de rotina mensal, conforme acompanhamento. Sem queixas.",
        "Veio aferir glicemia capilar de rotina. Sem sintomas."],
     "historico":["HAS em uso regular de medicacao.","DM2 em controle.","HAS leve em acompanhamento.","Pre-diabetico em monitoramento."]},
    # Orientacao medicamento
    {"vit": (20,80, 115,138, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Veio pedir orientacao sobre como tomar medicacao prescrita em consulta anterior. Sem queixas.",
        "Tem duvida sobre dose de antibiotico recem-prescrito. Assintomatico.",
        "Solicita esclarecimento sobre interacao entre medicacoes de uso continuo.",
        "Quer entender como aplicar insulina recem-prescrita. Sem sintomas atuais."],
     "historico":["DM2 recem-diagnosticado, iniciando insulina.","Varios medicamentos em uso por multiplas comorbidades.","Sem comorbidades.","Hipertenso em ajuste de medicacao."]},
    # Visita familiar
    {"vit": (20,80, 115,140, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Pede declaracao de acompanhante para idoso internado. Assintomatico.",
        "Comparece a pedido administrativo para preenchimento de formulario. Sem queixas.",
        "Quer informacoes sobre internacao de familiar. Sem sintomas.",
        "Solicita declaracao de saude para fins burocraticos. Sem queixas atuais."],
     "historico":["Sem comorbidades.","Hipertensa controlada.","Sem comorbidades.","Diabetico compensado."]},
    # Encaminhamento (NOVO)
    {"vit": (25,80, 115,140, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece para solicitar encaminhamento para cardiologista, conforme orientacao medica anterior. Sem queixas no momento.",
        "Solicita encaminhamento ambulatorial para endocrinologia. Assintomatico.",
        "Pede encaminhamento para dermatologista para avaliacao de lesao de pele cronica e estavel.",
        "Veio buscar encaminhamento para fisioterapia apos alta hospitalar."],
     "historico":["Hipertenso em acompanhamento estavel.","Diabetico tipo 2 controlado.","Sem comorbidades. Lesao dermatologica cronica em investigacao."]},
    # Atestado periodico (NOVO)
    {"vit": (20,65, 115,140, 70,90, 97,99, 36.3,36.9), "sex":"X",
     "sintomas":[
        "Comparece para renovacao de atestado de saude ocupacional periodico, exigencia da empresa. Assintomatico.",
        "Solicita atestado periodico para CNH (renovacao da carteira de motorista). Sem queixas atuais.",
        "Veio para checkup pre-operatorio de cirurgia eletiva agendada para o proximo mes. Saudavel.",
        "Solicita atestado de saude para viagem internacional. Assintomatico."],
     "historico":["Sem comorbidades. Hemograma recente normal.","Hipertenso bem controlado.","DM2 compensado, ultimo HbA1c 6.8."]},
]


CENARIOS_POR_COR = {
    "VERMELHO": VERMELHO,
    "LARANJA":  LARANJA,
    "AMARELO":  AMARELO,
    "VERDE":    VERDE,
    "AZUL":     AZUL,
}

DISTRIBUICAO = {
    "VERMELHO": 500,
    "LARANJA":  750,
    "AMARELO":  1250,
    "VERDE":    1750,
    "AZUL":     750,
}


# ---------------------------------------------------------------------------
# Geracao
# ---------------------------------------------------------------------------
def gerar_caso(template, cor):
    idade_min, idade_max, sis_min, sis_max, fc_min, fc_max, spo2_min, spo2_max, t_min, t_max = template["vit"]
    idade = random.randint(idade_min, idade_max)
    sex_choice = template["sex"]
    if sex_choice == "M":
        sexo = "Masculino"
    elif sex_choice == "F":
        sexo = "Feminino"
    else:
        sexo = random.choice(["Masculino", "Feminino"])
    sis = random.randint(sis_min, sis_max)
    dia = max(40, min(120, sis + random.randint(-50, -30)))
    return {
        "idade": idade,
        "sexo": sexo,
        "pressao": f"{sis}/{dia}",
        "frequencia_cardiaca": random.randint(fc_min, fc_max),
        "spo2": random.randint(spo2_min, spo2_max),
        "temperatura": round(random.uniform(t_min, t_max), 1),
        "sintomas": random.choice(template["sintomas"]),
        "historico": random.choice(template["historico"]),
        "classificacao": cor,
    }


def gerar_dataset():
    rows = []
    for cor, n_alvo in DISTRIBUICAO.items():
        cenarios = CENARIOS_POR_COR[cor]
        por_cenario = n_alvo // len(cenarios)
        sobra = n_alvo - por_cenario * len(cenarios)
        for i, template in enumerate(cenarios):
            n = por_cenario + (1 if i < sobra else 0)
            for _ in range(n):
                row = gerar_caso(template, cor)
                pa_sis, pa_dia = parse_pa(row["pressao"])
                row["pa_sistolica"] = pa_sis
                row["pa_diastolica"] = pa_dia
                row["pulse_pressure"] = max(0, pa_sis - pa_dia)
                row["sexo_M"] = int(row["sexo"] == "Masculino")
                row["sexo_F"] = int(row["sexo"] == "Feminino")
                fx = faixa_etaria(row["idade"])
                row["idade_faixa"] = fx
                row["flag_pediatrico"] = int(fx == 0)
                row["flag_idoso"] = int(fx == 2)
                row.update(extrair_flags(row["sintomas"], row["historico"]))
                rows.append(row)
    random.shuffle(rows)
    return rows


def salvar_csv(rows, path):
    campos_originais = ["idade", "sexo", "pressao", "frequencia_cardiaca", "spo2",
                        "temperatura", "sintomas", "historico"]
    features = ["pa_sistolica", "pa_diastolica", "pulse_pressure",
                "sexo_M", "sexo_F",
                "idade_faixa", "flag_pediatrico", "flag_idoso",
                *sorted(TERMOS.keys())]
    colunas = campos_originais + features + ["classificacao"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "triagem_dataset.csv"
    rows = gerar_dataset()
    salvar_csv(rows, out)
    counts = Counter(r["classificacao"] for r in rows)
    print(f"[OK] Dataset: {out}")
    print(f"[OK] Total: {len(rows)} linhas, {len(rows[0]) if rows else 0} colunas por linha")
    for cor in ["VERMELHO", "LARANJA", "AMARELO", "VERDE", "AZUL"]:
        n_cen = len(CENARIOS_POR_COR[cor])
        n = counts.get(cor, 0)
        print(f"    {cor:10s}: {n_cen:2d} cenarios -> {n:4d} linhas ({100*n/len(rows):.1f}%)")
