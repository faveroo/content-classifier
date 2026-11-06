# Avaliação

## Spam

Métricas prioritárias:

- Precision;
- Recall;
- F1-score;
- matriz de confusão.

## Principal risco

Falso positivo:

```text
email legítimo -> classificado como spam
```

Por isso, a decisão automática deverá usar threshold conservador.

## Categorias

Métricas prioritárias:

- F1 macro;
- F1 por categoria;
- matriz de confusão;
- distribuição de confiança.

## Baseline

Toda alteração relevante no modelo deverá ser comparada ao baseline anterior no mesmo conjunto de teste.

## Critério de adoção

Um modelo mais complexo só deverá substituir o atual se demonstrar ganho mensurável e relevante, sem aumentar desnecessariamente custo, latência ou dificuldade operacional.

## Relatório local implementado

O JSON em reports/baseline.json distingue os três splits; apresenta métricas de spam
em 0.5 e na política operacional, falsos positivos, precision de spam (null quando
nenhuma sugestão), categorias avaliadas sobre legítimos verdadeiros, distribuição
de confiança, taxa de revisão e acerto da label no fluxo completo. A matriz de confusão
usa linhas verdadeiras e colunas previstas na ordem labels. Revisão conta como abstenção,
não como classificação correta. O diagnóstico de categorias isolado não mede erros
propagados do primeiro estágio. Não há seleção de parâmetros usando o teste.
80 exemplos sintéticos verificam execução e isolamento, não generalização real.

Relatórios do fluxo privado identificam `source: gmail_reviewed`; relatórios sintéticos
continuam identificados como `synthetic`. Não houve avaliação com mensagens reais
nesta entrega. A coleta recente limitada não é uma amostra aleatória: planejar cobertura
de classes, períodos e templates antes de interpretar métricas. O comando de treino
gera métricas de todos os splits, mas decisões devem usar validação; teste é reservado
à avaliação final. Nenhuma seleção automática de parâmetros/calibração foi adicionada.

## Teste exploratório com rótulos de IA

Em 2026-10-27 foi executado um experimento autorizado com `gmail_ai_labeled`.
Suas métricas medem concordância com rótulos do ChatGPT, não acurácia com referência
humana independente. Relatar separadamente o categorizador isolado e o fluxo com
abstenção. Comparar com o baseline anterior usando as mesmas mensagens e splits.

O relatório identifica procedência por estágio, quantidade de grupos, quantidade de
positivos de spam e se seu recall é avaliável. Classes ausentes não foram aprendidas
ou avaliadas por presunção. Métricas detalhadas, manifesto e comparação desse experimento
permanecem privados. Nenhum ajuste de parâmetros/thresholds foi feito usando seu teste.

## SpamShield multilíngue

O experimento tem split novo, determinístico e estratificado por idioma × label, com
deduplicação textual exata anterior à divisão. Relatórios mostram F1 spam, F1 macro,
acurácia, matriz de confusão e métricas por idioma nos três conjuntos. Use validação
para decisões futuras e mantenha teste como medida final reservada. As métricas incluem
SMS públicos e não são validação externa para e-mails; Dutch e Italian não permitem
estimar separação ham/spam por ausência de ham. Deduplicação exata não elimina vazamento
por paráfrases ou templates semelhantes.

## Teste externo de e-mail — 2026-11-05

Enron-Spam foi dividido por grupos e SpamAssassin reservado como origem externa,
com sobreposição heurística removida antes do treino. Acurácia interna 98,12% versus
externa 44,67% no limiar 0,5: evidência de generalização insuficiente. No limiar
provisório 0,9 houve 121 falsos positivos entre 4.108 legítimos externos (2,95%).
Sem ajustar thresholds ou promover o modelo para automação após esses resultados.

SpamShield foi comparado nos mesmos textos, mas sua composição não garante ausência
de contaminação prévia por esses corpora. Temas continuam sendo avaliados em exemplos
sintéticos, separados de spam e sem alegação de acurácia real. Artefatos agregados e
limites: [messages-v1](experiments/messages-v1.md). A política distingue candidato
temático de label aceita; abstenção não conta como acerto do fluxo.

## Kaggle isuranga — 2026-11-05

Teste sintético agrupado: 1.296 registros, acurácia/F1 spam 100%; Enron anterior
92,75% na mesma amostra. SpamAssassin reutilizado, sem fit/ajuste: acurácia 68,44%,
F1 spam 56,53%. Em 0,9: 167 falsos positivos (4,07%), precision 70,13%, recall 21,69%.
O acerto sintético perfeito não representa qualidade real e não justifica promoção.
Grupos gigantes desbalanceiam os splits; regras do gerador podem persistir entre grupos.
Sem português na versão baixada. [Comparação](experiments/kaggle-isuranga-v1.md).

## Real v2 — seleção por validação e teste temático congelado

Fontes antigas foram reclassificadas como desenvolvimento/diagnóstico, não teste
externo novo. No diagnóstico multiorigem (5.187), acurácia 98,23% e F1 spam 98,14%;
29 falsos positivos no threshold selecionado 0,70. Esses resultados não provam
desempenho em mensagens recentes. Agregados públicos em
[real-v2-spam-development.json](experiments/real-v2-spam-development.json).

Temas: concordância com IA de 6/18 (33,33%), três ambíguos excluídos, sem avaliar
Financeiro/Pessoal e sem referência humana. Nenhuma mudança de modelo após esse teste.
100 mensagens recentes receberam apenas simulação; acurácia/precision permanecem
indefinidas. O novo avaliador humano exige revisão/privacidade e atribuição, aceita
classes ausentes e separa sobreposição ao treino. [Limites](14-real-development.md).
