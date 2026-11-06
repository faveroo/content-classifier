# Estratégia de Dados

## Objetivo

Criar um dataset supervisionado confiável sem expor desnecessariamente informações privadas.

## Unidade de classificação

Cada registro deverá representar uma mensagem de e-mail.

Campos candidatos:

- identificador interno anonimizado;
- remetente ou domínio, quando apropriado;
- assunto;
- corpo textual normalizado;
- label de spam;
- categoria;
- data de coleta opcional;
- origem da rotulagem;
- indicador de revisão humana.

## Dados que não devem ser versionados

- tokens OAuth;
- senhas;
- cookies;
- mensagens reais contendo informações sensíveis;
- anexos privados;
- endereços pessoais desnecessários;
- datasets brutos exportados de caixas reais.

## Estratégias de obtenção

1. dataset público de spam para baseline;
2. mensagens sintéticas para testes de pipeline;
3. amostras reais rotuladas localmente e mantidas fora do Git;
4. aproveitamento de labels existentes do Gmail, após definição de critérios.

## Split

O dataset deverá ser separado em treino, validação e teste, evitando vazamento de exemplos muito semelhantes entre os conjuntos.

## Risco de vazamento

Threads, mensagens duplicadas e templates podem produzir métricas artificialmente altas se versões quase idênticas aparecerem simultaneamente em treino e teste.

## Schema v1 implementado

Arquivo JSON UTF-8: lista não vazia de objetos, sem campos extras:

| Campo | Tipo e regra |
|---|---|
| id | string não vazia e única; identificador sintético |
| group_id | string não vazia; thread/template e variações pertencem ao mesmo grupo |
| subject | string; assunto ou corpo deve conter texto |
| body | string; texto simples, sem anexos |
| is_spam | booleano JSON estrito |
| category | null para spam, categoria v1 para legítimos |
| source | `synthetic` por padrão; `gmail_reviewed` no fluxo privado com opt-in |
| split | `train`, `validation` ou `test` |

O loader rejeita IDs duplicados, conteúdo normalizado idêntico e grupos em splits
diferentes. A detecção automática não identifica todas as paráfrases: agrupar templates
é responsabilidade de quem prepara o dataset. Cada split deve conter spam e todas
as categorias legítimas. Features: somente assunto + corpo. IDs, grupos e labels não
entram no modelo. Sem remetentes, domínios ou outros metadados privados nesta fase.
Validação serve para diagnósticos; teste fica reservado ao relatório final. Thresholds
iniciais são provisórios, não otimizados no teste. Exemplos em `data/examples/` são
inteiramente inventados e não medem generalização real.

## Coleta privada autorizada em 2026-10-27

O [fluxo Gmail somente leitura](12-gmail-readonly-setup.md) cria um arquivo de revisão
fora do Git. Cada registro recebe IDs aleatórios locais, assunto/corpo com minimização
automática e os campos extras `human_reviewed`, `privacy_reviewed` e `exclude`.
Labels e splits começam nulos. O filtro não garante anonimização: revisão manual é obrigatória.
Threads ficam agrupadas dentro de cada coleta; templates e relações entre coletas
precisam de revisão humana. O texto é limitado a 20 mil caracteres por campo.

`prepare-dataset` exige revisão de conteúdo/privacidade e rotulagem de cada registro
incluído. O contrato final mantém os campos v1, com `source: gmail_reviewed`.
IDs, grupos, duplicatas e cobertura por split seguem as mesmas validações do baseline.
O treino privado exige `--allow-private` e caminhos externos ao repositório para
dataset/modelo/relatório; não aceita misturar source sintético e real. Flags humanos
e source são declarações de procedência, não prova de anonimização ou qualidade.

## Revisão assistida por formulário local

O comando `review` permite rotular e minimizar texto no navegador local. Nenhum rótulo
é inferido nem confirmado automaticamente. Flags de revisão só mudam ao salvar escolhas
explícitas. Split e fusão de grupos preservam a conversa original inteira. Há backup
anterior à primeira edição, salvamento por substituição de arquivo e rejeição de
formulários desatualizados. O painel de cobertura não dispensa a validação final do
dataset; nenhuma amostra sem rótulos/revisões é usada para treinar.

## Exceção experimental autorizada: rótulos do ChatGPT

Em 2026-10-27, o usuário solicitou que o próprio ChatGPT classificasse as mensagens
pendentes, treinasse e testasse. O source `gmail_ai_labeled` identifica esses rótulos
provisórios e requer `--allow-private --allow-ai-labels`. Não marcar flags de revisão
humana nem passar esses dados pelo fluxo `gmail_reviewed` como se tivessem validação humana.
Anotações, justificativas, autoria, minimização e divisões ficam em manifesto privado.

O modo experimental permite cobertura parcial explicitamente relatada, preservando
schema, categorias válidas, IDs/textos únicos, isolamento de grupos, três splits não
vazios e pelo menos duas categorias legítimas em treino. Todas as categorias legítimas
presentes nos conjuntos reservados devem existir em treino. O modo humano/sintético
mantém suas exigências de cobertura integral; não misturar sources num dataset.

O splitter experimental agrupa antes de separar e usa seed 42 por assinatura de classes
dos grupos, com aproximadamente 60/20/20 quando há grupos suficientes. Assinaturas
raras ficam em treino (um grupo) ou treino/teste (dois grupos). Cobertura faltante
deve ser declarada, sem fabricar exemplos ou dividir templates para completar classes.

## Corpus externo multilíngue SpamShield (2026-10-28)

O experimento separado usa `M-Arjun/SpamShield-Datasets`, corpus de SMS com campos
`text`, `label` (0 ham, 1 spam), `category` e `language`. A página declara 23 idiomas,
149.359 mensagens e licença CC-BY-4.0, com acesso aos arquivos condicionado à aceitação
no Hugging Face. Dados não são versionados: o parquet permanece no cache local da HF.
É outro contrato de dados, não é convertido ao schema de e-mail nem misturado com os
datasets sintético/privado do baseline.

Duplicatas exatas após casefold e normalização de espaços são colapsadas antes do split;
se textos idênticos tiverem labels conflitantes, o pipeline falha. O split é 70/15/15,
seed 42, estratificado pela combinação idioma × label. Esse agrupamento só cobre
duplicatas textuais exatas, não paráfrases ou famílias de templates.

O dataset é CC-BY-4.0: dar crédito a M-Arjun, linkar a licença e indicar transformação
(deduplicação e criação dos splits). Dutch e Italian contêm apenas spam segundo o card.
O modelo é binário; categorias finas servem apenas para mostrar composição, não como
targets de inferência. Como os registros são SMS, resultados não estimam desempenho em
e-mails ou na caixa Gmail.

## Corpora de e-mail e categorias genéricas (2026-11-05)

Contrato público separado `public_email`, preparado de Enron-Spam (espelho CSV
documentado de MWiechmann quando a origem está indisponível) e SpamAssassin bruto.
Somente métricas agregadas podem ser publicadas; dados e modelos ficam ignorados.
Hashes, origens e limites de direitos estão em [13](13-messages-email.md).
SpamAssassin é externo reservado. Grupos exatos/template/conversas são formados
antes da divisão; sobreposição Enron/externo é removida do Enron. O restante tem
split aproximado 70/15/15 de grupos por assinatura de classe, seed 42. Heurísticas
não garantem isolamento de todas as paráfrases.

O contrato temático independe do corpus de spam. `email_reviewed` identifica revisão
humana de mensagens de qualquer provedor; preparação exige as mesmas confirmações
humanas/privacidade. Cobertura temática parcial é permitida explicitamente nesse novo
contrato, com classes treinadas e ausentes relatadas; o fluxo Gmail humano anterior
mantém cobertura integral. `email_ai_labeled` exige opt-in separado e não é revisão humana.
Nova rotulagem de IA não foi realizada. A amostra de 180 legítimos tem temas pendentes.

## Kaggle isuranga (2026-11-05)

`isuranga/spam-genuine-mail-contents-dataset`: 100 mil e-mails sintéticos segundo o
autor, versão 1, licença declarada Apache 2.0. Features limitadas a assunto/corpo;
excluir score antispam e metadados correlatos. IDs/referências só para agrupamento,
idioma só para métricas. Após deduplicação: 95.109 textos, 6.301 componentes.
70/15/15 de grupos, com concentração de mensagens no treino por dois grupos gigantes.
O contrato declara `training_source: kaggle_isuranga`, sem split externo interno;
diagnóstico externo separado. Dados/modelos ignorados. [Registro](experiments/kaggle-isuranga-v1.md).

## Desenvolvimento real v2 e autorização ampla (2026-11-05)

O usuário autorizou seguir o roteiro com autonomia. Os rótulos temáticos de IA são
`email_ai_labeled`, com motivos e exclusões privados, sem flags humanas. A amostra
teve 159 exemplos de desenvolvimento anotados antes do congelamento e 21 reservados
anotados depois (18 avaliáveis). O lote Gmail AI anterior mantém sua procedência.
Desenvolvimento temático: 194 registros, nove categorias; não é referência humana.

Enron e SpamAssassin anteriores agora são desenvolvimento já observado: 34.754
mensagens após quarentena de 180 grupos/965 exemplos da amostra temática. Splits por
grupo/fonte/classe. Coleta nova de 100 mensagens após congelamento, cinco sobreposições
detectadas, sem rótulos humanos ou fit. Artefatos e coletas permanecem privados.
[Contrato, hashes e revisão](14-real-development.md).
