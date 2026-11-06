# Mensagens e e-mails v1 — 2026-11-05

O detector de e-mail foi treinado e testado, mas **não demonstrou generalização
confiável**: 98,12% de acurácia no teste interno e 44,67% no corpus externo.
O categorizador temático inicial é demonstrativo, treinado com dados sintéticos.
Não há validação temática em mensagens reais nem alteração em caixas de e-mail.

## Método

Enron-Spam como origem de treino/validação/teste por grupos; SpamAssassin reservado
integralmente como externo. O servidor original Enron estava indisponível; o download
usou o [espelho CSV de MWiechmann](https://github.com/MWiechmann/enron_spam_data), que
descreve sua transformação do corpus de Metsis, Androutsopoulos e Paliouras.
[Fonte original](https://www2.aueb.gr/users/ion/data/enron-spam/) e
[SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/readme.html).
O mirror indica GPL-3.0 para seu repositório; direitos dos textos originais não são
presumidos uniformes. SpamAssassin mantém copyright dos remetentes. Os dados não
são redistribuídos neste repositório.

Agrupamento antes dos splits por texto exato, templates normalizados, corpos-template
longos e assunto canônico de conversa. Isso não elimina todas as paráfrases ou
traduções. 3.522 duplicatas textuais e 470 exemplos Enron pertencentes a grupos
sobrepostos ao externo foram removidos. O agrupamento por assinatura de classes
usa seed 42 e aproximadamente 70/15/15 dos grupos; tamanhos de mensagens variam.

Modelo: pipeline clássico existente, TF-IDF de palavras 1–2 com normalização de
acentos, regressão logística balanceada, `max_iter=1000`, seed 42. Fit somente em
treino. Não houve mudança de parâmetros após observar o teste, nem calibração.
Python 3.13.14 e scikit-learn 1.9.1.

```powershell
.\.venv\Scripts\python -m gmail_classifier.cli download-email-data
.\.venv\Scripts\python -m gmail_classifier.cli prepare-email-data
.\.venv\Scripts\python -m gmail_classifier.cli train-messages --category-dataset data/examples/synthetic-v1.json --sms-reference models/spamshield-multilingual.joblib
```

Hashes dos arquivos comprimidos e do dataset preparado, métricas detalhadas e
procedência em [messages-v1.json](messages-v1.json), somente agregados sem mensagens.
O modelo local é `models/messages-v1.joblib`; o relatório é
`reports/messages-v1.json`. Dados/modelo/relatório local são ignorados pelo Git.
O baseline e o detector SpamShield anteriores foram preservados.

## Spam — limiar 0,5

| Conjunto | Mensagens | Acurácia | Precision spam | Recall spam | Falsos positivos | Falsos negativos |
|---|---:|---:|---:|---:|---:|---:|
| Treino Enron | 20.809 | 99,15% | 98,36% | 99,88% | 165 | 12 |
| Validação Enron | 4.748 | 98,55% | 97,30% | 99,64% | 61 | 8 |
| Teste Enron | 4.247 | 98,12% | 96,92% | 99,33% | 66 | 14 |
| Externo SpamAssassin | 5.915 | 44,67% | 35,46% | 98,95% | 3.254 | 19 |

Diferenças de assunto, distribuição e preparação dos corpora permanecem. Enron-Spam
é pré-processado, enquanto SpamAssassin é MIME bruto. O resultado externo impede
interpretar o teste interno como qualidade para e-mails de origens variadas.

Com o limiar conservador de spam 0,90, o externo teve 121 legítimos sugeridos como
spam em 4.108 legítimos (2,95%) e recall de sugestões de spam de 65,14%. Isso ainda
não comprova segurança de automação. Ausência de sugestão de spam inclui revisão;
a acurácia binária dessa política não equivale a acerto de labels no fluxo completo.

O SpamShield existente foi comparado nos mesmos textos: acurácia 98,89% no teste
Enron e 44,62% no externo. Sua composição de dados de origem é parcialmente opaca;
não se pode garantir que o benchmark seja independente do treino desse modelo.
Nenhum detector é promovido automaticamente com base nessa comparação.

## Categorias temáticas

Dataset sintético v1 existente: 36 legítimos em treino, 18 em validação e 18 em teste,
com as nove categorias da taxonomia. Spam não participa do fit temático. Acurácia
isolada: 83,33% em validação e 94,44% em teste, apenas exemplos inventados.
No fluxo com o novo detector de e-mail, a taxa de revisão temática no teste é
88,89%. Não apresentar acurácia isolada como qualidade ponta a ponta ou qualidade real.

O fluxo genérico permite treinar temas com e-mails reais revisados e cobertura
parcial explicitamente relatada. Foram preparados 180 e-mails públicos legítimos,
no máximo um por grupo, em armazenamento privado para revisão. Rótulos temáticos
e confirmações humanas não foram gerados. O conjunto externo SpamAssassin não foi
usado para fit temático. [Guia de revisão e operação](../13-messages-email.md).

## Verificação

83 testes automatizados passaram: parsing MIME/HTML/anexos e formatos, isolamento de
vocabulário/coeficientes em ambos os estágios, separação por origem, sobreposição de
templates e conversas, opt-ins, persistência, importação privada e política de revisão.
Inferência foi executada nos canais e-mail e mensagem com textos inventados. Painel
Streamlit verificado com AppTest: classificação no modo e-mail e abertura do modo SMS,
sem exceções da aplicação. São verificações funcionais, não validação adicional de ML.
