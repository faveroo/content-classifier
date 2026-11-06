# Modelos locais

`email-classifier train` grava dois pipelines e sua política em `baseline.joblib`.
Artefatos são ignorados pelo Git. Carregar joblib pode executar código: use somente
arquivos de origem confiável. A versão do pacote e do scikit-learn deve coincidir;
caso contrário retreine. Não publicar modelos treinados com dados privados.

`train-messages` cria `messages-v1.joblib`, separando as origens de spam e de categorias.
O modelo público inicial usa spam Enron-Spam e temas sintéticos; generalização externa
de spam é insuficiente e a qualidade temática real está pendente. Modelos com categorias
reais devem ser salvos fora de árvores Git. `classify --channel message` reutiliza o
SpamShield local para spam e mantém o categorizador temático do modelo escolhido.
