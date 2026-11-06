# Mensagens e e-mails de qualquer provedor

O núcleo classifica assunto e corpo sem depender de Gmail, Outlook ou outro provedor.
Há dois estágios independentes: spam e categoria temática. Mensagens de texto podem
usar o detector SMS SpamShield; e-mails usam o novo detector Enron-Spam. Os dois
compartilham a taxonomia v1 e a política conservadora de sugestões/revisão.

## Estado e limites

Em 2026-11-05, o fluxo de e-mail foi treinado e avaliado com corpora públicos reais.
O teste interno teve acurácia de 98,12%, mas o teste externo teve somente 44,67% no
limiar 0,5. Isso revela generalização insuficiente; não é um detector validado para
uso automático. Nenhum parâmetro foi ajustado usando o teste externo. As categorias
iniciais foram treinadas com 36 exemplos inventados, com 18 de validação e 18 de
teste, e continuam demonstrativas. Não há validação temática em e-mails reais.

Uma amostra de 180 e-mails legítimos públicos foi preparada para revisão em
`%LOCALAPPDATA%\GmailEmailClassifier\email-category-review.json`. Categorias começam
nulas e as flags de revisão começam falsas. Não foi atribuída revisão humana nem
autorizada implicitamente a criação de novos rótulos de IA.

Resultados: [experimento messages-v1](experiments/messages-v1.md).

## Reproduzir o treino público

```powershell
.\.venv\Scripts\python -m gmail_classifier.cli download-email-data
.\.venv\Scripts\python -m gmail_classifier.cli prepare-email-data
.\.venv\Scripts\python -m gmail_classifier.cli train-messages --category-dataset data/examples/synthetic-v1.json --sms-reference models/spamshield-multilingual.joblib
```

O último argumento é opcional e exige um SpamShield treinado localmente. A comparação
usa os mesmos textos de validação, teste e externo. A origem dos exemplos no corpus
SpamShield não permite garantir que sejam independentes do seu treinamento.

Saídas: `models/messages-v1.joblib`, `reports/messages-v1.json`. Corpora, datasets e
modelos continuam ignorados pelo Git; os arquivos contêm mensagens e podem memorizar
dados. O baseline anterior não é sobrescrito. Prepare exige novo caminho de saída
se o dataset já existir, para preservar a divisão anterior.

## Classificar localmente

```powershell
# E-mail de qualquer provedor, sem conectar à conta
.\.venv\Scripts\python -m gmail_classifier.cli classify --model models/messages-v1.joblib --input caminho/mensagem.eml

# Uma caixa exportada ou uma pasta de .eml
.\.venv\Scripts\python -m gmail_classifier.cli classify --model models/messages-v1.joblib --input caminho/caixa.mbox

# Mensagem de texto: detector SMS + categorizador temático
.\.venv\Scripts\python -m gmail_classifier.cli classify --model models/messages-v1.joblib --channel message --input caminho/mensagem.txt

# Painel: selecione Mensagens e e-mails ou SpamShield SMS
.\.venv\Scripts\python -m streamlit run src/gmail_classifier/dashboard.py
```

Também há `--stdin` e `--text` com `--subject` opcional. Use arquivos ou stdin para
conteúdo privado: argumentos podem aparecer no histórico. As saídas da CLI contêm
previsões e procedência, sem reproduzir conteúdo ou nomes dos arquivos. Modelos
joblib devem ser próprios e confiáveis.

`category_candidate` é uma hipótese para revisão, acompanhada de confiança não
calibrada. Não equivale a `category` aceita pela política. Spam provável não recebe
categoria candidata. `IA/Revisar` significa abstenção, não classificação correta.
Os thresholds permanecem 0,90 para spam, 0,20 para permitir categoria e 0,60 para
aceitar categoria. O filtro de vocabulário desconhecido não detecta toda mudança de domínio.

## Treinar categorias com dados reais

```powershell
$privateDir = Join-Path $env:LOCALAPPDATA "GmailEmailClassifier"

# Amostra pública já preparada; ou importe mensagens de qualquer provedor:
.\.venv\Scripts\python -m gmail_classifier.cli import-messages --input caminho/caixa.mbox --output "$privateDir\outra-revisao.json"

# Revisar a amostra pública existente no formulário local
.\.venv\Scripts\python -m gmail_classifier.cli review --input "$privateDir\email-category-review.json"

# Depois de revisar conteúdo, rótulos e splits de todos os registros incluídos:
.\.venv\Scripts\python -m gmail_classifier.cli prepare-dataset --input "$privateDir\email-category-review.json" --output "$privateDir\email-categories-v1.json"
.\.venv\Scripts\python -m gmail_classifier.cli train-messages --category-dataset "$privateDir\email-categories-v1.json" --allow-private --model "$privateDir\messages-reviewed-v1.joblib" --report "$privateDir\messages-reviewed-v1.json"
```

O schema de categorias usa `id`, `group_id`, `subject`, `body`, `is_spam`, `category`,
`source` e `split`. O source genérico humano é `email_reviewed`. O treino temático
aceita cobertura parcial explícita, no mínimo duas categorias em treino, e exige
mensagens de validação/teste cujas classes existam em treino. Não inventa rótulos
para preencher classes ausentes. Spam não participa do fit do categorizador.

Conversa/template permanece no mesmo split. Duplicatas, grupos e templates textuais
normalizados que atravessem splits são rejeitados. A preparação de revisão continua
exigindo confirmação humana e de privacidade; minimização automática não anonimiza
tudo. O modo `email_ai_labeled` é experimental separado e requer autorização e
`--allow-private --allow-ai-labels`; não é gerado automaticamente pelo formulário.
As métricas desse modo representam concordância com IA, sem referência humana.

O comando `sample-email-categories` permite preparar outra amostra com `--output`
novo. Ele seleciona no máximo uma mensagem por grupo do Enron legítimo, preservando
os splits do experimento. O SpamAssassin externo não é usado para treinar categorias.

## Dados públicos e avaliação

- Enron-Spam: Metsis, Androutsopoulos e Paliouras, [fonte original](https://www2.aueb.gr/users/ion/data/enron-spam/).
  Com o servidor original indisponível nesta execução, foi usado o
  [espelho CSV de MWiechmann](https://github.com/MWiechmann/enron_spam_data).
  O repositório do espelho indica GPL-3.0; isso não é declaração de que os direitos
  dos textos originais foram transferidos ou de uma licença uniforme do corpus.
- [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html):
  o README mantém o copyright das mensagens com seus remetentes. Os arquivos servem
  a testes offline; não são enviados a sistemas de e-mail ativos.

Enron-Spam disponibiliza texto pré-processado; SpamAssassin é MIME bruto. Há diferenças
residuais de domínio e preparação. O parser local extrai apenas assunto e corpo,
prefere texto simples em multipart/alternative, elimina anexos e scripts/estilos em
HTML, sem carregar recursos externos. Cabeçalhos de verdict antispam não são features.
Limites: 2 MB por mensagem e 20 mil caracteres por campo; mbox é processado por mensagem.

Os grupos unem textos exatos, templates normalizados (URLs, e-mails, números e
pontuação), corpos-template longos e assuntos de conversas com pelo menos três
palavras e 12 caracteres. São heurísticas conservadoras, não detector completo de
paráfrases. Grupos ambíguos por rótulos exatos conflitantes são excluídos. Grupos Enron
sobrepostos ao externo são removidos antes do split. Os demais grupos são divididos
aproximadamente 70/15/15, seed 42, estratificados por assinatura de rótulos do grupo;
as proporções de mensagens variam com o tamanho dos grupos. O corpus SpamAssassin é
reservado integralmente. Arquivos são lidos dos arquivos comprimidos sem extrair paths.

Os relatórios separam teste interno, externo e qualidade temática. `at_policy` mede
sugestão de spam versus ausência de sugestão de spam; a ausência inclui revisão, não
apenas legítimos aceitos. Não interpretar sua acurácia binária como acerto do fluxo
completo. A avaliação temática apresenta também revisão e acerto final de labels.

## Experimento Kaggle adicional

O dataset isuranga foi treinado/testado separadamente, com dados sintéticos, apenas
assunto/corpo e modelo próprio `models/messages-kaggle-v1.joblib`. Selecione esse
caminho no campo do painel ou em `classify --model`. Os modelos anteriores foram
preservados. [Comandos, métricas e limitações](experiments/kaggle-isuranga-v1.md).
