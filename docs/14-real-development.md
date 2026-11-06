# Desenvolvimento com fontes reais e revisão humana

Em 2026-11-05, o usuário autorizou seguir o roteiro com autonomia. Foram concluídos:
rotulagem experimental temática, comparação clássica com fontes reais, seleção por
validação, congelamento, teste temático reservado e simulação em mensagens novas.
**Validação humana independente continua pendente.** A IA não pode produzir essa
referência e chamá-la de revisão humana.

## Resultado atual

- Spam: detector de caracteres selecionado pela melhor consistência de F1 macro
  entre fontes na validação. Diagnóstico histórico de 5.187 e-mails: acurácia 98,23%,
  F1 spam 98,14%. Não é uma avaliação nova e independente de provedores atuais.
- Temas: 194 e-mails reais com rótulos de IA em desenvolvimento (149 treino/45
  validação), com nove categorias. Teste público reservado: 6/18 de concordância
  (33,33%); três exemplos ambíguos excluídos. Financeiro e Pessoal não são avaliados
  nesse teste. As categorias não atingiram o critério de aceitação automática.
- Coleta recente: 100 mensagens após renovar consentimento somente leitura; cinco
  sobreposições de treino detectadas. Sem rótulos humanos, acurácia e precision são
  indefinidas. Nenhuma mensagem foi alterada.

O **modo ativo é apenas de revisão**. Probabilidades e categorias candidatas são
hipóteses. Para uma mensagem que o detector considera spam, o painel pode mostrar
uma hipótese temática condicionada a ela ser legítima. Isso não aceita uma categoria
nem altera o resultado científico do modelo congelado.

## Usar agora

```powershell
.\.venv\Scripts\python -m gmail_classifier.cli classify --active --input caminho/mensagem.eml
.\.venv\Scripts\python -m gmail_classifier.cli classify --active --channel message --input caminho/mensagem.txt
.\.venv\Scripts\python -m streamlit run src/gmail_classifier/dashboard.py
.\.venv\Scripts\python -m gmail_classifier.experiment_v2 status
```

No modo SMS, mantêm-se os thresholds provisórios SMS 0,90/0,20, sem reaproveitar a
política ajustada para e-mail. O painel e `--active` verificam o hash do artefato e
mantêm revisão obrigatória. Modelos antigos permanecem disponíveis por caminho explícito.

O diretório privado é retornado por `status`; Python da Microsoft Store pode redirecionar
`LOCALAPPDATA`. Não deduza o caminho apenas pela variável do PowerShell. Principais arquivos:

| Arquivo privado | Finalidade |
|---|---|
| `protocol.json` | Critérios e reserva definidos antes da seleção |
| `annotations.json` | Rótulos de IA e motivos; sem confirmações humanas |
| `category-development.json` | Desenvolvimento temático privado |
| `spam-selection.json`, `category-selection.json` | Comparações por validação |
| `messages-real-v2.joblib`, `freeze.json` | Modelo congelado e hashes de entradas |
| `final-evaluation.json` | Resultados finais privados e limites |
| `recent-reserved.json` | Amostra recente sem rótulos humanos |
| `recent-predictions.json` | IDs e hipóteses, sem copiar assunto/corpo |
| `human-review-queue.json` | 100 itens com revisão exigida |
| `human-validation-plan.json`, `RESULTADO.md` | Orientações e resultado privado |

## Método e limites

O SpamAssassin externo anterior já foi observado; agora participa do desenvolvimento
junto ao Enron. Não continua sendo chamado de teste externo independente. Foram
excluídos 180 grupos/965 mensagens relacionados à amostra temática antes de preparar
spam, para evitar que o primeiro estágio visse o teste temático.

Desenvolvimento de spam: 34.754 e-mails, 24.358 treino, 5.209 validação e 5.187
diagnóstico. Textos/templates/conversas são agrupados antes de dividir, com seed 42
e assinatura de fonte/classe. Heurísticas não eliminam todas as paráfrases.

Foram comparados TF-IDF de palavras 1–2, caracteres `char_wb` 3–5 e combinação.
Cada ramo tem no máximo 80 mil features, `min_df=2`, TF sublinear. Normalização
compartilhada limita texto a 12 mil caracteres, remove marcas HTML quando presentes,
normaliza acentos, URLs, e-mails e números. Só conteúdo entra no fit. Regressão
logística balanceada, C=1, seed 42; spam liblinear, temas lbfgs. Sem Transformers.

Spam é escolhido pelo pior F1 macro entre fontes de validação, desempate pelo F1
global. Isso selecionou caracteres, embora a combinação tivesse melhor média global.
O threshold 0,70 atende os critérios empíricos de validação (precision >=98%, FPR
<=1% por fonte, pelo menos dez sugestões); legítimos até 0,30. Não são garantias
de segurança, probabilidades calibradas ou validação em português atual.

Temas usam os 180 exemplos públicos e o lote privado anterior com labels de IA,
com exclusões de ambiguidade e sem misturar rótulos inventados para completar classes.
Os 21 exemplos reservados só foram consultados para anotação depois de congelar os
modelos. Três foram excluídos por ambiguidade, antes de ver suas previsões. Confiança
de anotação é julgamento subjetivo, não probabilidade validada; o lote legado foi
incluído pela autorização anterior, sem nova medição de confiança.

Palavras/caracteres/combinação foram comparados somente em treino/validação. Nenhum
threshold temático atingiu concordância >=90% com pelo menos dez aceites na validação;
por isso, a aceitação foi desabilitada. O teste final ruim não foi usado para mudar
parâmetros ou rótulos. Novas iterações precisam de novos testes; este já foi observado.

O conjunto recente foi coletado após congelamento e só recebeu inferência. Sua
referência humana está em branco. A coleta é limitada, inclui Spam e exclui Lixeira;
não é amostragem aleatória de provedores ou usuários. As cinco sobreposições detectadas
não são contadas como garantia de novidade; o avaliador humano também separa grupos
com sobreposição ao treino.

[Métricas públicas de desenvolvimento de spam](experiments/real-v2-spam-development.json).
Modelos, rótulos, mensagens e resultados privados completos ficam fora do Git.

## Revisão e avaliação humana

O plano privado aponta para as duas coleções originais: a amostra pública temática
e `recent-reserved.json`. O formulário não recebe rótulos da IA como confirmação
humana. Execute `review --input CAMINHO` no arquivo indicado pelo plano e classifique
com base no conteúdo, revisando privacidade e excluindo casos ambíguos.

Depois de confirmar manualmente todos os registros incluídos:

```powershell
.\.venv\Scripts\python -m gmail_classifier.experiment_v2 evaluate-human --collection CAMINHO/recent-reserved.json --report CAMINHO/referencia-humana-v1.json --reviewer "Nome do revisor"
```

Esse avaliador não treina nem altera thresholds. Aceita cobertura parcial de classes,
registra classes ausentes/recall não avaliável, desconta duplicatas exatas e separa
o subconjunto sem sobreposição detectada. Recusa flags pendentes ou procedência de IA.
Flags e atribuição são declarações; não provam identidade independente ou representatividade.
Revisão completa não promove automaticamente o modelo para escrita no Gmail.

## Reproduzir em outro experimento privado

`python -m gmail_classifier.experiment_v2 --directory NOVO_CAMINHO prepare`
prepara outra execução a partir das amostras locais. `view`/`annotate` registram rótulos
de IA; `fit-spam` compara detectores; `fit-topics --allow-private --allow-ai-labels`
seleciona temas e congela; `evaluate` só permite uma avaliação final. `simulate-recent`
e `finalize` preparam hipóteses e revisão. Não reutilize o teste observado como se
fosse novo. Referência humana independente, cobertura por tema e avaliação em mensagens
recentes de outros provedores são as pendências científicas atuais.
