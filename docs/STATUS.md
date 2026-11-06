# Status do Projeto

## Fase atual

**Experimento real v2 congelado, com seleção por validação, temas rotulados por IA e simulação recente concluídos em 2026-11-05. Modo ativo exige revisão; validação humana independente pendente.**
O usuário escolheu OAuth somente leitura para coleta de amostras reais privadas.
Adaptador e fluxo local implementados; configuração Cloud, consentimento e coleta inicial concluídos.

## Escopo desta entrega

Taxonomia v1, schema validado, estrutura Python, baseline TF-IDF + Logistic Regression
em dois estágios, política de revisão, CLI local, testes e instruções de execução.
Agora também: OAuth somente leitura opcional, coleta limitada com minimização de dados,
rotulagem/revisão manual, preparação de dataset privado e simulação offline em lote.
Guia: `docs/12-gmail-readonly-setup.md`.
O comando `review` oferece formulário de rotulagem no navegador local, com backup
e salvamento por mensagem; nenhum rótulo ou confirmação humana é gerado automaticamente.
Por solicitação explícita posterior, o ChatGPT classificou a amostra em um fluxo
experimental separado (`gmail_ai_labeled`), sem marcar revisão humana.

## Validação concluída

- 41 testes automatizados de schema, isolamento, política, persistência e CLI.
- Treinamento e previsão executados em Python 3.13.14.
- Métricas e ambiente registrados em `docs/experiments/baseline-v1.json`.
- Política padrão encaminhou 100% dos exemplos de validação e teste para revisão.
- CI adicionada para Python 3.11 e 3.13; resultado remoto deve ser consultado no GitHub.
- Em 2026-10-27: 59 testes passaram com Python 3.13.14, incluindo OAuth e API simulados,
  extração MIME, paginação, minimização, revisão, caminhos privados, treino e simulação.
- Após adicionar o formulário local: 65 testes passaram, incluindo escape de HTML,
  proteção de origem/CSRF, backup, rejeição de edição desatualizada e preservação de grupos.
- Dependências opcionais Gmail instaladas no ambiente local.
- Autenticação real concluída com `gmail.readonly` em 2026-10-27.
- Primeira coleta: 50 mensagens dos últimos 30 dias, sem Spam/Lixeira; 50 consultadas,
  50 salvas localmente fora do Git, nenhuma descartada na extração.
- Simulação offline com o baseline sintético: 50 previsões, todas `IA/Revisar`, zero
  alterações na caixa. Sem rótulos humanos, não há métricas de acurácia nesta amostra.
- Experimento autorizado: 50 rótulos atribuídos pelo ChatGPT, 32 mensagens em treino,
  9 em validação e 9 em teste; conversas/templates agrupados antes da separação.
- Novo categorizador treinado exclusivamente no split de treino e comparado ao baseline
  nos mesmos splits. Detector de spam anterior preservado por falta de positivos.
- 71 testes passaram após adicionar opt-in de IA, cobertura parcial explícita,
  isolamento de treino, procedência e preservação do detector de spam.
- Modelo, manifesto de rótulos, relatórios e comparação permanecem em armazenamento
  privado. Nenhum parâmetro/threshold foi alterado usando o teste.

## Limites

As 50 mensagens reais e seus derivados permanecem privados. Revisão humana de conteúdo
e rótulos continua pendente; o experimento usa rótulos de IA conforme autorização explícita.
Filtros e minimização assistida não garantem anonimização. As métricas são concordância
com o ChatGPT, não acurácia validada por pessoas. A amostra tem cobertura parcial e não
inclui exemplos positivos de spam. O fluxo completo permanece em revisão para todas
as mensagens, com thresholds provisórios preservados.
Nenhuma escrita Gmail implementada ou realizada.

## Próxima etapa

Ampliar amostra com grupos independentes, categorias raras e spam confirmado; obter
referência humana independente para medir qualidade. Usar validação para futuros ajustes
e reservar novo teste final. Calibração e escrita no Gmail continuam pendentes de etapas
posteriores. Resultado detalhado: `resultado-ai-v1.md` no armazenamento privado local.

## Experimento SpamShield multilíngue (2026-10-28)

Implementado um pipeline paralelo para `M-Arjun/SpamShield-Datasets`: download gated
da HF, deduplicação de textos idênticos, divisão estratificada por idioma/rótulo,
baseline TF-IDF de caracteres + Logistic Regression, métricas globais e por idioma,
e dashboard Streamlit para métricas e classificação local. O corpus declarado contém
23 idiomas e é SMS binário; não altera o baseline de e-mail nem integra Gmail.

Os arquivos do dataset continuam fora do Git; atribuição CC-BY-4.0 registrada na UI e
documentação. Dutch/Italian não contêm ham. O pipeline inicialmente teve smoke test
com dados sintéticos temporários; treino/teste do corpus real estavam pendentes de acesso HF.

### Treino/teste real concluído em 2026-11-05

Autenticação HF local disponível e download autorizado concluído. Corpus com 149.359
registros, 148.522 após remover duplicatas normalizadas, sem conflitos de rótulos
nesses grupos. Divisão seed 42: 103.965 treino, 22.278 validação e 22.279 teste.
Pipeline existente executado sem mudança de código ou hiperparâmetros.

Teste global: acurácia 97,36%, F1 spam 97,41%, precision spam 98,32% e recall 96,52%;
189 falsos positivos e 400 falsos negativos. Teste em português: 189 mensagens,
acurácia 98,41%, 1 falso positivo e 2 falsos negativos. Métricas de SMS, sem validação
de generalização para Gmail. Deduplicação exata não isola paráfrases/templates.

Modelo salvo e inferência após recarregamento verificados; 71 testes automatizados
passaram. Dados em `data/hf-cache`, modelo em `models/spamshield-multilingual.joblib`
e relatório em `reports/spamshield-multilingual.json`, todos ignorados pelo Git.
Registro reproduzível e métricas agregadas: [experimento SpamShield v1](experiments/spamshield-v1.md).
Baseline de e-mail e política preservados, sem alterações no Gmail.

## Mensagens/e-mails de qualquer provedor — 2026-11-05

Importação de texto/EML/MBOX, revisão humana genérica, treino independente de spam e
temas, CLI `classify` e painel disponíveis. Canal de mensagem seleciona SpamShield;
canal e-mail usa Enron-Spam. Artefato `models/messages-v1.joblib`, relatório em
`reports/messages-v1.json`, sem sobrescrever modelos anteriores.

35.719 e-mails após preparação: 20.809 treino, 4.748 validação, 4.247 teste e 5.915
externo SpamAssassin; 3.522 duplicatas exatas e 470 exemplos Enron de grupos sobrepostos
ao externo removidos. Teste interno 98,12% versus externo 44,67% de acurácia em 0,5;
121 falsos positivos externos em 0,9. Detector não validado para automação.

Categorias treinadas com o dataset sintético existente: 36/18/18 legítimos. 180
e-mails reais públicos aguardam revisão temática em armazenamento privado. Nenhum
novo rótulo de IA nem confirmação humana foi fabricado. Cobertura e qualidade reais
de temas, português atual e múltiplos provedores continuam pendentes.

83 testes passaram. Consulte [guia](13-messages-email.md) e
[experimento](experiments/messages-v1.md). Gmail continua somente leitura.

## Dataset Kaggle isuranga — 2026-11-05

Treino/teste solicitado concluído: origem declara 100 mil e-mails sintéticos, cinco
idiomas (sem português), licença Apache 2.0. Após deduplicação, 95.109 textos:
92.474 treino, 1.339 validação e 1.296 teste, com grupos preservados. 89 testes passaram.
Novo modelo `models/messages-kaggle-v1.joblib`, selecionável na CLI/painel, sem
sobrescrever anteriores. Teste sintético 100%; diagnóstico externo SpamAssassin
68,44%, F1 spam 56,53%, com 167 falsos positivos em 0,9. Não promovido automaticamente.
Temas reais continuam pendentes. [Registro completo](experiments/kaggle-isuranga-v1.md).

## Desenvolvimento real v2 — 2026-11-05

Roteiro solicitado com autonomia executado. Comparados palavras, caracteres e união;
seleção pelo conjunto de validação. Spam: fontes reais Enron e SpamAssassin em
desenvolvimento, com quarentena da amostra temática. Selecionado caracteres; diagnóstico
histórico de 5.187 registros teve acurácia 98,23%, F1 spam 98,14% e 29 falsos positivos
na política 0,70. Não é desempenho recente validado.

Anotados 180 e-mails públicos pela IA, com exclusões por ambiguidade e reserva de
teste antes da seleção. Com o lote privado AI anterior: 194 em desenvolvimento,
149 treino/45 validação, nove categorias. Modelo congelado antes de abrir o teste
temático: concordância com IA 6/18 (33,33%), três excluídos. Não é referência humana;
aceitação automática temática desabilitada.

OAuth somente leitura renovado pelo usuário; 100 mensagens recentes coletadas após
congelamento e simuladas offline, cinco sobreposições de treino detectadas. Sem labels
humanos, acurácia permanece indefinida. Fila operacional exige 100 revisões, zero
alterações Gmail. Novo modelo privado ativo no painel e `classify --active`; anteriores
preservados. 105 testes passaram e painel verificado com AppTest.

Pendências científicas: revisão humana independente, dados temáticos mais amplos e
referência recente em português/múltiplos provedores. Fluxo pronto para avaliar essa
revisão, sem preencher confirmações por IA. [Guia](14-real-development.md).
