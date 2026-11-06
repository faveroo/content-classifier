# Plano de Machine Learning

## Tipo de problema

Classificação supervisionada de texto.

## Baseline planejado

### Representação

TF-IDF.

### Classificador

Logistic Regression.

### Motivos

- implementação simples;
- treinamento rápido;
- boa referência para classificação textual;
- suporte a probabilidades;
- facilidade de interpretação e depuração;
- funciona bem com matrizes esparsas.

## Experimentos posteriores

Somente após o baseline:

- Linear SVM;
- Naive Bayes;
- combinações de word n-grams e character n-grams;
- embeddings;
- Sentence Transformers;
- modelos Transformer em português/multilíngues.

## Features candidatas

Conteúdo textual:

- assunto;
- corpo;
- remetente/domínio;
- combinação ponderada dos campos.

Metadados poderão ser avaliados separadamente.

## Hierarquia prevista

1. classificador de spam;
2. caso não seja spam, classificador de categoria.

Isso permite métricas, thresholds e ciclos de treinamento independentes.

## Baseline local v0.1 implementado

Dois pipelines independentes: TF-IDF de palavras (unigramas e bigramas,
acentos normalizados), LogisticRegression com class_weight=balanced,
random_state=42 e max_iter=1000. Assunto + corpo; sem features de metadados.
Fit exclusivamente no split train; categorizador treinado somente com legítimos.
Thresholds provisórios conforme D008; não ajustados pelo conjunto de teste.
Artefato joblib contém versão de formato, pacote, sklearn e política. CLI e setup no README.

Desde 2026-10-27, o mesmo baseline aceita datasets privados revisados com opt-in.
A preparação de dados é separada do treinamento e não há mudança no algoritmo ou
nos thresholds. Métricas com referência humana independente ainda estão pendentes.

## Experimento com supervisão do ChatGPT

Autorizado e executado em 2026-10-27, mantendo TF-IDF + Logistic Regression e thresholds.
O novo categorizador usa exclusivamente o split de treino da amostra rotulada pela IA.
Quando não há duas classes de spam no treino, o modo experimental exige explicitamente
um `--spam-reference` confiável, copia esse estágio sem fit e registra sua procedência.
Não se pode afirmar que spam foi retreinado nessa situação. O artefato persistido guarda
fontes dos estágios e indica ausência de referência humana. O modelo original é preservado.

## Experimento SpamShield de SMS multilíngue

Separado dos dois estágios e da taxonomia de e-mail, o detector binário usa TF-IDF de
caracteres 2–5 com limite de 500 mil features e Logistic Regression balanceada.
N-gramas de caracteres acomodam escrita e scripts variados com o baseline clássico.
O split estratifica idioma × label; reporta métricas gerais e por idioma em treino,
validação e teste. Parâmetros não são escolhidos com base no teste. Nenhum threshold de
ação Gmail foi alterado. A inferência expõe probabilidades binárias, sem alegação de
calibração; o modelo não classifica as categorias finas do corpus.

## Classificação independente de mensagens/e-mails

`train-messages` treina TF-IDF de palavras 1–2 + Logistic Regression balanceada
separadamente: spam no Enron-Spam, temas somente nos legítimos do dataset temático.
O pipeline clássico existente é preservado. Há opt-in para dados temáticos reais/IA,
procedência por estágio e avaliação temática isolada e com política. Inicialmente o
categorizador usa exemplos sintéticos, sem alegação de qualidade real. A seleção de
canal permite inferência com o detector SMS já treinado, sem misturar corpora ou
probabilidades. O teste externo ruim impede promoção a automação; nenhum hiperparâmetro
foi escolhido a partir dele. [Método e operação](13-messages-email.md).

Kaggle isuranga usa o mesmo pipeline em dataset sintético separado, sem ajustar
thresholds pelo teste. Novo artefato, comparação com Enron nos mesmos textos e
diagnóstico externo pós-fit separado. Labels binários não ensinam categorias temáticas.

## Desenvolvimento real v2

Comparação clássica de palavras, caracteres e união, com normalização uniforme e
limites de features. Seleção de spam pelo pior F1 macro de fonte em validação;
temas pelo F1 macro de validação. Ambos selecionaram caracteres. Policy escolhida
apenas por validação: spam 0,70, legítimos até 0,30; temas sem threshold viável,
aceitação desabilitada. Modelos congelados antes da anotação do teste temático.
Saídas ativas exigem revisão humana independente; SMS preserva política provisória
própria. Não reutilizar teste observado para novas escolhas. [Método](14-real-development.md).
