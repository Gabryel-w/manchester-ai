# Avaliação do classificador (golden set)

Conjunto fixo de 30 vinhetas clínicas (6 por cor Manchester) e um runner que
mede a qualidade do classificador em todos os modelos do catálogo.

## Estrutura

| Arquivo | Conteúdo |
|---------|----------|
| `golden_set.json` | Casos canônicos com cor esperada, dados do paciente e flag `trava_esperada` indicando se as travas determinísticas devem disparar |
| `runner.py` | Script standalone que roda o golden set contra os modelos Groq e gera um relatório |
| `results_YYYYMMDD_HHMMSS.json` | Relatórios versionados de cada execução (ignorados pelo git) |

## Como rodar

Pré-requisitos: venv ativo, `GROQ_API_KEY` no `.env` na raiz do projeto.

```bash
# Roda os 3 modelos Groq (default)
python eval/runner.py

# Roda apenas um modelo específico
python eval/runner.py --modelo llama-3.3-70b-versatile

# Inclui os modelos Ollama (precisa do serviço local rodando)
python eval/runner.py --com-ollama
```

## O que o runner mede

- **Acurácia exata**: percentual de cores classificadas exatamente como esperado.
- **Acurácia ± 1 cor**: aceita cores adjacentes (ex.: VERMELHO classificado como
  LARANJA conta como acerto). Métrica útil porque o erro clínico de uma cor
  adjacente é menos grave que de duas cores.
- **Erros graves**: distância ≥ 2 (ex.: VERDE quando o esperado era VERMELHO).
- **Latência média / máxima**: por modelo.
- **Matriz de confusão 5×5**: cor esperada vs. cor obtida.

## Interpretando o relatório

O runner imprime no terminal e salva um JSON em `results_<timestamp>.json`. Cada
caso testado registra:
- `inconsistencia`: se as travas determinísticas precisaram corrigir o LLM.
- `cor_regra`: a cor que as travas forçaram, quando aplicável.

Casos com `trava_esperada: true` no golden set são onde os sinais vitais sozinhos
já forçam a cor — usar esses casos para validar que a integração das travas no
agente está funcionando independente da capacidade do LLM.

## Aviso ético

As vinhetas são sintéticas, escritas para cobrir cenários típicos de cada cor
Manchester. Não representam pacientes reais. A acurácia neste conjunto é
indicativa, não validação clínica formal.
