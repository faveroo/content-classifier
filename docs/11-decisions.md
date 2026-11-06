# Registro de Decisões

Este arquivo resume decisões arquiteturais e de produto relevantes.

## D001 — Repositório público sem código na fase inicial

**Status:** aceito.

O projeto começa com documentação e estrutura de pastas. Código Python será adicionado somente após a etapa de planejamento.

## D002 — Dois classificadores conceituais

**Status:** provisoriamente aceito.

Separar spam e categorização temática permite thresholds, datasets e métricas independentes.

## D003 — Baseline com TF-IDF + Logistic Regression

**Status:** aceito para o primeiro experimento.

Modelos mais complexos deverão ser comparados objetivamente a esse baseline.

## D004 — Labels reversíveis no Gmail

**Status:** aceito.

Durante testes, usar labels `IA/*` em vez de ações destrutivas.

## D005 — Human-in-the-loop

**Status:** planejado.

Casos de baixa confiança deverão ser encaminhados para revisão humana e poderão alimentar ciclos futuros de treinamento.

## D006 — Implementação local autorizada (2026-10-26)

**Status:** aceito. A restrição inicial de D001 foi cumprida e a implementação Python
está autorizada pelo usuário. Gmail e dados privados ficam para etapa explícita posterior.

## D007 — Taxonomia v1 e schema sintético

**Status:** aceito para o baseline local. Preservar as nove categorias documentadas,
com regras de finalidade/ambiguidade em 05. JSON estrito conforme 04, com grupo e split
explícitos. Nenhuma mensagem real ou metadado pessoal nos exemplos públicos.

## D008 — Política conservadora provisória

**Status:** aceito para demonstração. Spam com probabilidade >= 0.90 sugere IA/SPAM;
probabilidade <= 0.20 permite categorização. Intervalo intermediário exige IA/Revisar.
Categoria exige confiança >= 0.60; vocabulário totalmente desconhecido exige revisão.
Probabilidades não são calibradas e thresholds precisam de validação representativa.
Somente sugestões locais, sem ações externas.

## D009 — Gmail OAuth somente leitura e preparação privada (2026-10-27)

**Status:** implementação autorizada pelo usuário, que escolheu OAuth somente leitura.
Configuração Google Cloud e consentimento real concluídos em 2026-10-27, após o
usuário salvar o cliente e concluir o login. Primeira coleta limitada e simulação
concluídas: 50 mensagens, todas encaminhadas para revisão, nenhuma alteração na caixa.
Revisão humana, treino com dados reais e avaliação representativa continuam pendentes.

Usar cliente Desktop dedicado, escopo exclusivo `gmail.readonly` e callback loopback
com PKCE. Adaptador de coleta separado do ML, consulta explícita e limite de 1–500.
Não implementar escrita, criação de labels ou ações sobre mensagens nesta etapa.

Coletar somente assunto/corpo necessários, ignorar anexos e usar IDs aleatórios locais.
Filtros automáticos são minimização, não garantia de anonimização. Exigir revisão
humana de privacidade e rótulos antes de preparar dataset, com isolamento por grupo.
Threads entre coletas e templates exigem agrupamento manual. Campos têm limite de texto.

Aceitar `gmail_reviewed` somente com opt-in para treino/avaliação, sem misturar dados
sintéticos e reais; relatórios indicam a origem correta. Credenciais, tokens, coletas,
datasets, modelos e relatórios privados devem ficar fora do checkout e de árvores Git.
Preservar o baseline e thresholds até medir resultados representativos. Não afirmar
coleta, calibração ou qualidade real com base nos testes simulados.

## D010 — Rotulagem humana antes do treino real (2026-10-27)

**Status:** implementado para viabilizar o treinamento solicitado. A inspeção inicial
da coleta encontrou 50 mensagens sem rótulos, revisão de privacidade ou splits.
O treinamento supervisionado aguarda esses dados; não usar previsões do baseline
como referência humana nem marcar revisões automaticamente.

Disponibilizar o comando `review`, com formulário em loopback e salvamento local,
para revisar texto, categoria, privacidade e grupos/splits. Preservar backup, rejeitar
edições desatualizadas e impedir vazamento do conteúdo via HTML ou logs. Treinar
somente depois de validar revisão, integridade e cobertura das classes. A revisão
inicial pode indicar necessidade de mais coleta para cobrir classes ausentes.

## D011 — Experimento com rótulos do ChatGPT por solicitação explícita (2026-10-27)

**Status:** autorizado pelo usuário e executado. O usuário pediu que o ChatGPT
classificasse as mensagens pendentes e então treinasse/testasse, substituindo a
exigência de rotulagem humana para este experimento. D010 continua válida para o
fluxo de referência humana; esta exceção não permite fingir revisão por uma pessoa.

Usar source `gmail_ai_labeled` e opt-in separado, manter anotações e artefatos privados,
relatar concordância com os rótulos da IA e preservar um teste isolado por grupos.
Permitir cobertura parcial no experimento, explicitando classes não avaliadas. Não
classificar promoções como spam apenas por serem comerciais. Sem ambas as classes de
spam, preservar estágio de referência explicitamente selecionado e não afirmar novo fit.

O categorizador da amostra foi treinado/testado sem alterar algoritmo ou thresholds,
com comparação ao baseline original. O fluxo Gmail continua somente leitura; nenhuma
label foi aplicada. Generalização e automação confiável exigem amostra maior e revisão
independente. Resultados detalhados ficam no relatório privado do experimento.

## D012 — Experimento binário com SpamShield multilíngue (2026-10-28)

**Status:** implementação autorizada pelo pedido do usuário. Usar o corpus gated
`M-Arjun/SpamShield-Datasets` em fluxo Python isolado, mantendo os dados no cache
local e citando a licença CC-BY-4.0. A aquisição exige que o usuário aceite os termos
do dataset no Hugging Face e autentique localmente.

Treinar TF-IDF de caracteres + Logistic Regression para spam/ham, com deduplicação
textual exata e split 70/15/15 estratificado por idioma e label. Criar dashboard local
para métricas por idioma e inferência de texto. O corpus é de SMS e não substitui a
avaliação de e-mail, não treina as categorias da taxonomia existente e não autoriza
alterar Gmail. Dutch/Italian são spam-only. Atribuir a fonte e registrar transformação.

**Execução em 2026-11-05:** acesso HF local disponível, corpus baixado e pipeline
existente treinado/testado com seed 42, sem alterar hiperparâmetros. Nesta execução,
cache explicitamente em `data/hf-cache` (ignorado pelo Git). Publicar somente métricas
agregadas e identificação da revisão/hash em [SpamShield v1](experiments/spamshield-v1.md);
dados e modelo permanecem ignorados. Os resultados não promovem o detector para Gmail
nem substituem a avaliação de e-mail. Nenhum ajuste foi feito usando o teste.

## D013 — Mensagens e e-mails independentes do provedor (2026-11-05)

**Status:** solicitado e implementado. Entrada local por texto/EML/MBOX, sem novas
integrações ou escrita em caixas. Treinar spam e temas com datasets separados,
preservando baseline clássico e política conservadora. Canal explícito escolhe
detector de e-mail ou SMS. Categoria candidata não é label aceita.

Enron-Spam para fit/validação/teste interno por grupos e SpamAssassin para externo;
remoção de duplicatas/templates/threads sobrepostos antes do split. Falha da origem
Enron levou ao espelho CSV atribuído a MWiechmann, sem atribuir licença uniforme aos
textos originais. Corpus/modelo continuam ignorados, relatório agregado versionável.

Treino real de spam executado; teste externo de 44,67% impede alegação de generalização
confiável. Não ajustar parâmetros usando esse teste. Temas iniciais continuam sintéticos;
180 e-mails públicos aguardam revisão temática privada. O novo source `email_reviewed`
permite cobertura parcial explícita no categorizador independente, sem afrouxar o
contrato humano Gmail anterior. Rótulos de IA exigem opt-in e procedência separados.
Nova rotulagem de IA não foi presumida. [Guia](13-messages-email.md).

## D014 — Kaggle isuranga como experimento sintético adicional (2026-11-05)

**Status:** solicitado, implementado e treinado/testado. Download público com hash
e versão; descrição informa 100 mil e-mails sintéticos, licença declarada Apache 2.0.
Usar apenas assunto/corpo, agrupando antes do split e preservando grupos gigantes.
Auditar sobreposição ao treino anterior. Mesmo baseline e thresholds; novo artefato,
comparação nos mesmos textos e diagnóstico externo sem fit. Acerto 100% no sintético
não comprova qualidade real; externo 68,44% ainda insuficiente e política de falsos
positivos pior que a anterior. Não promover automaticamente. Labels binários não
treinam temas. [Registro](experiments/kaggle-isuranga-v1.md).

## D015 — Autonomia para desenvolvimento real, IA e revisão reservada (2026-11-05)

**Status:** o usuário solicitou seguir o roteiro com autonomia. A autorização inclui
rotulagem experimental de IA, comparação clássica, ajustes por validação e simulação;
não é confirmação humana. O Gmail continua somente leitura. Consentimento renovado
pelo usuário após `invalid_grant`; coleta recente de 100 mensagens reservada após
congelar o modelo, sem usá-las no fit ou no ajuste dos thresholds.

Enron/SpamAssassin observados viram desenvolvimento/diagnóstico. Quarentena de 180
grupos/965 registros da amostra temática evita vazamento entre estágios. Comparar
palavras, caracteres e união, selecionar por validação e registrar hashes antes de
abrir a anotação do teste temático. Avaliação final única; testes consultados passam
a ser diagnósticos em novas iterações.

Temas reais têm `email_ai_labeled`, sem revisão humana: 149 treino, 45 validação e
18 de teste avaliáveis (três ambíguos excluídos). A concordância de 33,33% e a falta
de threshold viável impedem aceitação automática. O registro ativo exige revisão;
hipóteses condicionais não alteram o modelo congelado. Correções humanas têm fluxo
independente, atribuição e métricas separadas, sem treino implícito.
[Guia e limites](14-real-development.md). Artefatos reais completos permanecem privados.
