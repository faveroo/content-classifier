# SpamShield multilíngue v1 — 2026-11-05

## Dados e método

Fonte: [M-Arjun/SpamShield-Datasets](https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets),
revisão `ccf3cdd831235a47361eca4995b82ef221ff3406`, arquivo `combined.parquet`.
SHA-256: `8d24c2fa969f3e3a46f45135325bf4ccf29ce759c01bf9be6c6241a601ba2b48`.
Atribuição: M-Arjun, SpamShield-Datasets, [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Transformações: deduplicação por casefold/espaços e criação de splits.

O arquivo contém 149.359 registros em 23 idiomas. A deduplicação remove 837 registros,
restando 148.522 textos únicos; não foram encontrados rótulos conflitantes para textos
normalizados idênticos. O corpus inclui dados sintéticos segundo o dataset card.

O pipeline existente é mantido: TF-IDF de caracteres 2–5 (até 500 mil features,
`min_df=2`, `sublinear_tf=True`) e regressão logística balanceada, solver liblinear,
`max_iter=1000`. Treino/validação/teste em aproximadamente 70/15/15, estratificados
por idioma × label, seed 42. Somente treino participa do fit do TF-IDF e classificador.
Classificação binária no limiar 0,5; nenhum ajuste de parâmetros com base no teste.

| Split | Mensagens | Mensagens em português |
|---|---:|---:|
| Treino | 103.965 | 882 |
| Validação | 22.278 | 189 |
| Teste | 22.279 | 189 |

Ambiente: Python 3.13.14, scikit-learn 1.9.1, pandas 3.1.0, pyarrow 26.0.0,
joblib 1.6.0 e huggingface_hub 1.33.0.

## Reprodução local

Na raiz do projeto, com o extra `multilingual` instalado e acesso HF autorizado:

```powershell
.\.venv\Scripts\python -m gmail_classifier.cli download-spamshield --cache-dir data/hf-cache
.\.venv\Scripts\python -m gmail_classifier.cli train-spamshield --dataset data/hf-cache/datasets--M-Arjun--SpamShield-Datasets/snapshots/ccf3cdd831235a47361eca4995b82ef221ff3406/combined.parquet
```

Dados em `data/hf-cache`, modelo em `models/spamshield-multilingual.joblib` e relatório
em `reports/spamshield-multilingual.json` são ignorados pelo Git. O cache local foi
selecionado explicitamente nesta execução; o padrão da CLI continua sendo o cache HF.

## Limitações

O corpus é de SMS e não mede desempenho no Gmail nem treina categorias temáticas de
e-mail. Dutch e Italian contêm apenas spam. Deduplicação exata não elimina paráfrases,
traduções ou templates semelhantes entre splits, portanto as métricas podem ser
otimistas. Predominância de inglês também limita a interpretação de médias globais.
As probabilidades não foram calibradas. O baseline de e-mail e sua política permanecem
preservados; nenhuma alteração no Gmail faz parte deste experimento.

## Resultados

Treinamento e avaliação concluídos em 2026-11-05. Relatório agregado completo em
[spamshield-v1.json](spamshield-v1.json), sem mensagens ou credenciais. A execução
terminou sem avisos de convergência. O modelo salvo foi recarregado e a inferência
foi verificada com uma mensagem inventada; isso verifica funcionamento, não acurácia.
Os 71 testes automatizados existentes passaram (`python -m pytest` no ambiente virtual).
Nenhuma mudança no código ou nos hiperparâmetros foi necessária.

| Métrica | Treino | Validação | Teste |
|---|---:|---:|---:|
| Acurácia | 98,22% | 97,37% | 97,36% |
| Precision de spam | 98,89% | 97,97% | 98,32% |
| Recall de spam | 97,65% | 96,91% | 96,52% |
| F1 spam | 98,26% | 97,44% | 97,41% |
| F1 macro | 98,22% | 97,37% | 97,36% |

No teste global: 10.611 legítimos corretos, 189 falsos positivos, 400 spams não
detectados e 11.079 spams corretos. A taxa de falsos positivos sobre legítimos é
1,75% (189/10.800), no limiar binário 0,5. Isso não valida automação de spam no Gmail.

Em português: teste com 189 mensagens (95 ham, 94 spam), acurácia 98,41% e F1 spam
98,40%. Foram 94 legítimos corretos, 1 falso positivo, 2 spams não detectados e
92 spams corretos. A amostra pequena limita a precisão dessa estimativa. O resultado
agregado de validação em português coincide com o teste, mas são splits distintos.
Nenhum parâmetro foi ajustado após observar esses resultados.
