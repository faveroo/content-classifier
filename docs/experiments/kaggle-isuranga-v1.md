# Kaggle isuranga v1 — 2026-11-05

Treino/teste adicional solicitado pelo usuário concluído com
[SPAM + Genuine, Mail contents dataset](https://www.kaggle.com/datasets/isuranga/spam-genuine-mail-contents-dataset),
autor `isuranga`, versão 1, atualização de 2025-09-27. A API pública declara Apache 2.0.
O autor informa que todos os 100.000 registros são **sintéticos**, equilibrados em
50.000 legítimos/50.000 spam. Não são e-mails reais, apesar do nome do dataset.

Arquivo: `email_dataset_100k.csv`, 63.717.161 bytes descomprimidos. ZIP SHA-256:
`122c43aa8d482f9a7ca358af2ee06683a5098672af981de58448197825abd38b`.
Download público sem credenciais; dados, metadados e manifesto permanecem ignorados
pelo Git em `data/kaggle-isuranga`. Somente agregados são publicados.

## Preparação

Features: apenas `subject` + `body_plain`, com HTML como fallback. `raw_text`,
endereços, domínios, SPF/DKIM/DMARC, nomes de anexos, `x_spam_score` e demais metadados
não entram no modelo. IDs e referências são hashes temporários usados somente para
agrupar conversas. Idioma serve apenas à avaliação.

Removidas 4.891 duplicatas, restando 95.109 textos e 6.301 grupos. Componentes conectam
texto exato, templates normalizados, corpos-template longos, assuntos canônicos e
referências de conversa antes do split. Os dois maiores grupos têm 43.432 e 43.338
registros e ficaram em treino. Não os dividimos para forçar proporções de mensagens.

Seed 42, aproximadamente 70/15/15 **dos grupos**, por assinatura de classes:

| Split | Grupos | Mensagens | Legítimos | Spam |
|---|---:|---:|---:|---:|
| Treino | 4.411 | 92.474 | 44.590 | 47.884 |
| Validação | 945 | 1.339 | 275 | 1.064 |
| Teste | 945 | 1.296 | 244 | 1.052 |

Idiomas declarados na origem: inglês 60.032, alemão 9.934, italiano 10.012, espanhol
9.981 e francês 10.041. Sem português. As tags são declarações do gerador.
Não foram detectados grupos sobrepostos ao treino Enron anterior; a auditoria é
heurística e não prova independência de todos os padrões de geração.

## Treino e avaliação

Pipeline clássico existente: TF-IDF de palavras 1–2, normalização de acentos,
regressão logística balanceada, `max_iter=1000`, seed 42. Fit somente em treino.
Nenhum parâmetro/threshold foi alterado após observar testes. Python 3.13.14,
scikit-learn 1.9.1. [Relatório completo](kaggle-isuranga-v1.json).

| Resultado em 0,5 | Kaggle | Anterior Enron |
|---|---:|---:|
| Acurácia de validação, 1.339 mensagens | 100% | 90,14% |
| Acurácia de teste, 1.296 mensagens | 100% | 92,75% |
| Precision / recall / F1 spam no teste | 100% / 100% / 100% | Ver JSON |

Zero falsos positivos/negativos no teste sintético. Acerto perfeito em dados gerados
**não comprova acurácia real**. Regras e vocabulário do gerador podem persistir entre
grupos; o desbalanceamento dos conjuntos reservados também limita a interpretação.
Métricas por idioma e comparação no subconjunto sem sobreposição detectada estão no JSON.

## Diagnóstico externo pós-fit

Modelo congelado avaliado nos 5.915 e-mails históricos externos SpamAssassin do
experimento anterior; nenhuma sobreposição de grupos com o treino Kaggle foi detectada.
É um benchmark reutilizado, já observado. Sem fit, calibração ou seleção de parâmetros.
[Relatório externo](kaggle-isuranga-v1-external.json).

| Métrica externa em 0,5 | Kaggle | Anterior Enron |
|---|---:|---:|
| Acurácia | 68,44% | 44,67% |
| Precision spam | 48,79% | 35,46% |
| Recall spam | 67,18% | 98,95% |
| F1 spam | 56,53% | 52,21% |
| Falsos positivos | 1.274 | 3.254 |
| Falsos negativos | 593 | 19 |

Prever tudo como legítimo daria 69,45% de acurácia, com recall de spam zero. O ganho
sobre o modelo anterior não torna a nova acurácia global satisfatória. Em 0,90,
houve 167 falsos positivos entre 4.108 legítimos (4,07%), precision 70,13% e recall
21,69%. O anterior tinha 121 falsos positivos e precision 90,68% nesse limiar.
Não promover automaticamente pela acurácia em 0,5. Esses números não validam
automação, português atual ou e-mails de múltiplos provedores em produção.

## Categorias e uso

Este dataset só fornece rótulos binários. Não treina Financeiro, RH, Suporte etc.
O estágio temático continua com os 36 legítimos sintéticos anteriores (18 de validação,
18 de teste), com procedência separada. Rotulagem temática real permanece pendente.
Novo modelo: `models/messages-kaggle-v1.joblib`. Relatórios locais:
`reports/messages-kaggle-v1.json` e `reports/messages-kaggle-v1-external.json`.
Todos ignorados pelo Git; modelos anteriores preservados. Selecione o novo caminho
no painel, que mostra aviso específico sobre o corpus sintético.

```powershell
.\.venv\Scripts\python -m gmail_classifier.cli download-kaggle-mail
.\.venv\Scripts\python -m gmail_classifier.cli prepare-kaggle-mail --reference-dataset data/email/public-v1.json
.\.venv\Scripts\python -m gmail_classifier.cli train-messages --dataset data/kaggle-isuranga/prepared-v1.json --category-dataset data/examples/synthetic-v1.json --email-reference models/messages-v1.joblib --model models/messages-kaggle-v1.joblib --report reports/messages-kaggle-v1.json
.\.venv\Scripts\python -m gmail_classifier.cli evaluate-email-external --model models/messages-kaggle-v1.joblib --dataset data/email/public-v1.json --training-dataset data/kaggle-isuranga/prepared-v1.json --reference-model models/messages-v1.joblib --report reports/messages-kaggle-v1-external.json
.\.venv\Scripts\python -m gmail_classifier.cli classify --model models/messages-kaggle-v1.joblib --input caminho/mensagem.eml
```

Referências exigem o experimento anterior localmente. Para só Kaggle, omita
`--reference-dataset`, `--email-reference` e a avaliação externa. Preparação exige
caminho novo se o JSON já existir. 89 testes passaram, incluindo allowlist, labels
estritos, isolamento, comparação e diagnóstico sem fit. Painel verificado com AppTest.
