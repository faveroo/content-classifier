# Baseline sintético v1 — 2026-10-26

Comando: `email-classifier train --dataset data/examples/synthetic-v1.json`.
Resultado completo em `baseline-v1.json`, com hash do dataset, ambiente, política e seed.
Somente métricas agregadas de dados inventados. Modelos e relatórios locais ficam ignorados.

| Medida | Validação (20) | Teste (20) |
|---|---:|---:|
| F1 macro de categorias, sobre legítimos verdadeiros | 0.8333 | 0.9407 |
| Taxa de revisão do fluxo completo | 100% | 100% |
| Sugestões automáticas de spam | 0 | 0 |
| Acerto de label no fluxo completo (abstenção não conta) | 0% | 0% |

Sem sugestões de spam, precision operacional é indefinida (`null`); zero falsos positivos
nessa situação não demonstra segurança de automação. O segundo modelo isolado pode
separar exemplos simples e ainda assim o fluxo completo se abster de todas as decisões.
Os quatro exemplos de treino por classe são insuficientes para confiança operacional.
Não ajustamos thresholds após observar o teste. O próximo passo é um plano separado
para dados representativos e autorizados, com rotulagem e validação de thresholds.

Métricas podem variar entre versões das bibliotecas; este registro preserva o ambiente
exato usado. Para comparar mudanças, manter dataset/splits e registrar novo relatório.

## SpamShield multilíngue v1 — 2026-11-05

Treino/teste no corpus público real concluído. Método, reprodução e resultados em
[spamshield-v1.md](spamshield-v1.md); métricas agregadas por split/idioma em
[spamshield-v1.json](spamshield-v1.json). Os dados e o modelo permanecem ignorados
pelo Git. O experimento binário de SMS é separado do baseline de e-mail acima.

## Mensagens/e-mails v1 — 2026-11-05

Treino público Enron-Spam e teste externo SpamAssassin concluídos. Acurácia interna
98,12% e externa 44,67%: generalização insuficiente. Categorizador temático inicial
continua sintético. [Método e limites](messages-v1.md), [agregados](messages-v1.json).

## Kaggle isuranga v1 — 2026-11-05

Treino/teste adicional em 100 mil e-mails sintéticos: teste gerado 100%, diagnóstico
externo 68,44%, ainda insuficiente para automação. [Método e limites](kaggle-isuranga-v1.md).

## Real v2 — 2026-11-05

Comparação por validação em fontes reais observadas; modelo congelado e simulação
recente privada. Spam diagnóstico: acurácia 98,23%, F1 98,14%. Temas: concordância
com IA de 33,33%, sem aceitação automática. Referência humana independente pendente.
[Guia e limites](../14-real-development.md),
[agregados públicos de spam](real-v2-spam-development.json).
Os resultados privados completos e as anotações não são versionados.
