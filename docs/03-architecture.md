# Arquitetura Conceitual

## Fluxo previsto

```text
Gmail
  |
  v
Leitura e normalização
  |
  v
Pré-processamento
  |
  +--------------------+
  |                    |
  v                    v
Modelo de Spam     Modelo de Categoria
  |                    |
  +----------+---------+
             |
             v
       Política de decisão
             |
      +------+------+
      |             |
      v             v
Aplicar label    IA/Revisar
```

## Componentes futuros

### Gmail Adapter

Responsável exclusivamente pela comunicação com Gmail.

### Preprocessor

Responsável por transformar uma mensagem em uma representação textual consistente.

### Spam Classifier

Modelo binário independente.

### Category Classifier

Modelo multiclasse aplicado a mensagens consideradas legítimas.

### Decision Policy

Converte previsões e níveis de confiança em ações permitidas.

### Training Pipeline

Responsável pelo treinamento, avaliação e persistência dos modelos.

## Separação importante

O código de integração com Gmail não deverá conhecer detalhes internos do algoritmo de treinamento. Da mesma forma, o modelo não deverá depender diretamente da Gmail API.

## Implementação local

Na inferência, spam é avaliado primeiro; apenas probabilidade <= 0.20 habilita o
modelo de categoria. O intervalo incerto vai diretamente para revisão, e >= 0.90
sugere IA/SPAM. O vocabulário totalmente desconhecido também exige revisão.
A saída é sugestão local; o adaptador Gmail/OAuth opcional permite exclusivamente leitura.
Treinamento e avaliação ficam separados da política e do loader de dados.

`gmail.py` cuida de OAuth, paginação e extração de texto, sem importar ML.
`review.py` valida revisão humana e prepara arquivos; a simulação offline carrega
um modelo já treinado, sem conectar ao Gmail. `private_storage.py` bloqueia caminhos
privados dentro do checkout e de árvores Git. Coleta, revisão, treino e inferência
são comandos separados; o dataset nunca depende da caixa de produção durante o fit.

## Núcleo independente de provedor (2026-11-05)

`mail_input.py` extrai assunto/corpo de EML/MBOX e texto; `email_corpora.py` adquire
corpora públicos e agrupa exemplos antes dos splits. `message_model.py` treina spam
e categorias a partir de datasets separados, mantendo a política e artefato Baseline.
`text_privacy.py` compartilha minimização com o adaptador Gmail, sem dependência da API.
CLI `classify` e painel selecionam detector de e-mail ou SMS explicitamente. Uma
categoria candidata é diagnóstico; só a política aceita labels. Guia: [13](13-messages-email.md).

`experiment_v2.py` separa preparação, anotação, seleção por validação, congelamento,
avaliação única e simulação. `text_features.py` é a normalização compartilhada e
serializável. `active_model.py` verifica o artefato privado e exige revisão na camada
operacional, preservando as saídas científicas congeladas. [Fluxo real](14-real-development.md).
