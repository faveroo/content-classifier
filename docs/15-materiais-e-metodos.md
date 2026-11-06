# Materiais e métodos para classificação de mensagens e emails

## 1 Delineamento do projeto

Foi desenvolvido um sistema experimental de processamento de linguagem natural para identificar spam e atribuir categorias temáticas a mensagens e e-mails. A implementação separa duas tarefas de aprendizagem supervisionada: classificação binária entre spam e mensagem legítima, denominada ham, e classificação multiclasse com uma única categoria temática por mensagem legítima. As nove categorias são Financeiro, RH, Suporte, Sistemas, Comercial, Administrativo, Notificações, Pessoal e Outros.

O processamento do conteúdo é independente do provedor. O sistema recebe texto, arquivos EML e caixas MBOX, além de oferecer um adaptador opcional para coleta no Gmail com OAuth somente leitura. Treinamento, inferência e integração com a caixa postal são componentes separados. A descrição considera os experimentos registrados até 5 de novembro de 2026, com destaque para o experimento real-v2, atualmente congelado e operado com revisão obrigatória.

## 2 Materiais e fontes de dados

### 2 1 Corpora públicos de emails reais

O Enron-Spam foi utilizado como fonte histórica de mensagens legítimas e spam. O corpus é atribuído a V. Metsis, I. Androutsopoulos e G. Paliouras. Como o servidor original estava indisponível durante a obtenção dos dados, foi empregado o espelho CSV mantido por MWiechmann, com registro dessa procedência. O repositório do espelho descreve 33.716 e-mails; esse número caracteriza a distribuição de origem e não corresponde automaticamente ao conjunto final após processamento [1, 2].

Também foi utilizado o SpamAssassin Public Corpus, composto por mensagens legítimas e spam em formato bruto. Foram obtidos os arquivos 20030228_easy_ham.tar.bz2, 20030228_easy_ham_2.tar.bz2, 20030228_hard_ham.tar.bz2, 20030228_spam.tar.bz2 e 20050311_spam_2.tar.bz2. A documentação pública informa aproximadamente 6.047 mensagens para a composição descrita, com diferenças entre versões e duplicatas a considerar. Os direitos sobre as mensagens permanecem com seus remetentes [3].

Na primeira experiência com e-mails, o Enron forneceu treino, validação e teste interno; o SpamAssassin foi reservado como avaliação externa. Depois que seus resultados foram observados, esses conjuntos históricos passaram a ter função de desenvolvimento e diagnóstico no real-v2. Assim, o SpamAssassin deixou de constituir um teste externo independente nessa etapa.

### 2 2 Dataset sintético adicional do Kaggle

Foi conduzido um experimento separado com o dataset SPAM + Genuine Mail contents dataset, publicado por Isuranga Nipun Kumara, usuário isuranga, no Kaggle. A versão 1, atualizada em 27 de setembro de 2025, contém 100.000 e-mails artificiais, sendo 50.000 ham e 50.000 spam. O autor declara que não há e-mails reais nem informações pessoais, e os metadados consultados indicam licença Apache 2.0 [4].

O arquivo utilizado foi email_dataset_100k.csv. Foram selecionados somente assunto e corpo textual, com extração de HTML quando necessário. Campos como pontuação sintética de spam, autenticação SPF/DKIM/DMARC, endereços e demais metadados foram excluídos das variáveis preditoras para reduzir atalhos artificiais de classificação. Esse dataset fornece rótulos binários e não fornece as nove categorias temáticas do projeto.

### 2 3 Dataset de mensagens curtas SpamShield

O SpamShield-Datasets, disponibilizado por M-Arjun no Hugging Face, foi empregado em um experimento auxiliar de detecção de spam em SMS. O arquivo combined.parquet contém 149.359 registros em 23 idiomas. O cartão do dataset informa combinação de fontes públicas com aumento sintético e licença CC BY 4.0. Foi fixada a revisão ccf3cdd831235a47361eca4995b82ef221ff3406 [5].

O detector resultante é acionado explicitamente no canal de mensagens curtas. Seus dados e métricas não são apresentados como evidência de desempenho em e-mails. O SpamShield também não fornece diretamente a taxonomia temática adotada pelo projeto.

### 2 4 Dados locais e amostras privadas

O primeiro baseline utilizou 80 mensagens inteiramente inventadas, armazenadas em data/examples/synthetic-v1.json. Essa base permitiu verificar o funcionamento dos dois estágios sem depender de mensagens pessoais. Seu desempenho tem caráter demonstrativo.

Uma coleta privada inicial reuniu 50 mensagens do Gmail, referentes aos 30 dias anteriores a 27 de outubro de 2026, sem incluir Spam e Lixeira. Essa amostra recebeu rótulos experimentais do ChatGPT com procedência gmail_ai_labeled, sem confirmação de revisão humana. Os 50 exemplos eram legítimos e, portanto, não permitiam treinar um detector de spam com duas classes. Posteriormente, o lote foi reaproveitado no desenvolvimento temático do real-v2.

Para ampliar a categorização temática, foram separados 180 e-mails públicos legítimos do Enron, com no máximo um representante por grupo. A anotação experimental por IA manteve procedência email_ai_labeled e excluiu casos considerados ambíguos. A coleta privada mais recente reuniu outras 100 mensagens após o congelamento do modelo, exclusivamente para inferência e preparação de revisão humana. Essas mensagens não participaram do treinamento.

## 3 Organização de treino validação e teste

Treino designa os exemplos usados para ajustar vocabulário, pesos TF-IDF e coeficientes do classificador. Validação designa os exemplos usados para comparar candidatos e selecionar limiares. Teste reservado designa uma avaliação após a seleção. Um conjunto já observado não recupera independência ao ser reutilizado em experimentos posteriores.

### 3 1 Partições do experimento atual

Antes da preparação de spam do real-v2, foram retirados 180 grupos associados à amostra temática, correspondentes a 965 mensagens. Essa quarentena evita exposição do primeiro estágio ao conjunto temático reservado. O desenvolvimento de spam resultante reuniu 34.754 e-mails históricos do Enron e do SpamAssassin.

| Tarefa no real-v2 | Treino | Validação | Avaliação posterior | Origem e finalidade |
|---|---:|---:|---:|---|
| Detecção de spam | 24.358 | 5.209 | 5.187 | Enron e SpamAssassin; avaliação histórica de diagnóstico |
| Categorização temática | 149 | 45 | 18 avaliados | Enron e lote Gmail anterior no desenvolvimento; teste reservado do Enron |
| Simulação recente | 0 | 0 | 100 sem referência humana | Gmail; somente inferência e fila de revisão |

Na tarefa temática, os 180 exemplos públicos foram inicialmente separados em 135 de treino, 24 de validação e 21 de teste. Os 159 exemplos públicos de desenvolvimento receberam anotação antes do congelamento; 15 foram excluídos por ambiguidade. Após incorporar o lote privado anterior e aplicar os controles de agrupamento e duplicação, o desenvolvimento temático utilizado totalizou 194 exemplos, distribuídos em 149 de treino e 45 de validação.

Os 21 exemplos públicos reservados foram anotados apenas depois do congelamento. Três casos ambíguos foram excluídos antes da consulta às previsões, restando 18 avaliados. O teste temático cobre sete das nove categorias, sem exemplos de Financeiro e Pessoal. A comparação com esses rótulos mede concordância com anotação de IA, sem equivalência a acurácia validada por pessoas.

Na amostra recente de 100 mensagens foram detectadas cinco sobreposições com o treinamento. Por isso, sua coleta posterior ao congelamento não permite tratar todos os registros como independentes. A ausência de referência humana impede calcular acurácia ou precisão válidas para essa amostra.

### 3 2 Partições dos experimentos anteriores e auxiliares

| Experimento | Treino | Validação | Teste | Observação |
|---|---:|---:|---:|---|
| Sintético local | 40 | 20 | 20 | Inclui spam; estágio temático usa 36, 18 e 18 legítimos |
| Gmail inicial | 32 | 9 | 9 | Somente legítimos com temas de IA; lote depois reutilizado |
| SpamShield SMS | 103.965 | 22.278 | 22.279 | 148.522 textos após retirar 837 duplicatas |
| Enron inicial | 20.809 | 4.748 | 4.247 | SpamAssassin separado com 5.915 exemplos externos |
| Kaggle sintético | 92.474 | 1.339 | 1.296 | 95.109 textos após retirar 4.891 duplicatas |

As linhas representam experiências distintas e não devem ser somadas como uma única população. No SpamShield, a divisão aproximada de 70/15/15 foi estratificada por idioma e classe, com semente 42; os subconjuntos em português contêm 882, 189 e 189 mensagens. A deduplicação foi textual exata após normalização de maiúsculas e espaços.

Nos experimentos de e-mail, a divisão ocorreu por grupos de textos, templates e conversas, com semente 42. No Kaggle, os 6.301 grupos foram separados em aproximadamente 70/15/15 dos grupos, resultando em proporções desiguais de mensagens: os dois maiores grupos, com 43.432 e 43.338 registros, permaneceram no treino. O conjunto não contém português entre seus idiomas declarados. O diagnóstico adicional no SpamAssassin reutilizou um benchmark já observado.

## 4 Preparação do conteúdo e prevenção de vazamento

Para mensagens MIME, foi priorizado o corpo em texto simples, com conversão de HTML quando necessário. Conteúdos de script, estilo e cabeçalho HTML foram descartados; anexos e mensagens anexadas não foram utilizados como corpo. O importador limita o tamanho a 2 MB por mensagem e a 20 mil caracteres por campo textual. Assunto e corpo são combinados para formar a entrada do modelo.

No real-v2, uma função compartilhada entre treino e inferência limita essa entrada a 12 mil caracteres, extrai texto de HTML quando reconhecido, aplica casefold e normalização Unicode NFKD com remoção de acentos. URLs, endereços de e-mail e sequências numéricas são substituídos por tokens padronizados; espaços repetidos são reduzidos. Não são consultados os destinos dos links.

A separação em grupos precede a divisão amostral. As heurísticas conectam textos idênticos, templates normalizados, corpos longos semelhantes por template, assuntos canônicos e identificadores de conversa quando disponíveis. A divisão do real-v2 utiliza assinaturas de origem e classe para distribuir os grupos. Nos temas, também são verificadas conexões após a preparação dos textos, excluindo do desenvolvimento grupos relacionados ao teste reservado.

Somente o conteúdo textual participa do ajuste. Origem do corpus, split, identificadores, rótulos e confirmações de revisão não entram nas características preditoras. O vocabulário TF-IDF é aprendido exclusivamente no treino e aplicado aos demais conjuntos sem novo ajuste. Esses controles reduzem vazamento conhecido; não demonstram ausência de todas as paráfrases, traduções ou padrões compartilhados de geração.

## 5 Técnicas de aprendizagem de máquina

### 5 1 Representação textual e classificadores

Foi adotado um baseline clássico de NLP com vetorização TF-IDF e regressão logística. O TF-IDF converte o texto em uma matriz esparsa que pondera a frequência dos termos no documento e sua frequência no corpus de treinamento. Ele constitui a representação dos dados; a regressão logística é o algoritmo supervisionado que aprende a separar as classes [6, 7].

O detector de spam utiliza regressão logística binária e estima um escore de probabilidade para spam. O categorizador utiliza regressão logística multiclasse para estimar escores das nove categorias e selecionar a mais provável. Cada mensagem legítima recebe uma categoria candidata. O projeto não utiliza classificação temática multirrótulo.

No real-v2 foram comparadas três representações, mantendo o classificador e a preparação do texto consistentes:

| Variante | Características TF-IDF | Configuração compartilhada |
|---|---|---|
| Palavras | Unigramas e bigramas de palavras | Até 80 mil características por ramo; min_df igual a 2; frequência sublinear; float32 |
| Caracteres | N-gramas de 3 a 5 caracteres com char_wb | Mesmos limites; construção dentro dos limites das palavras |
| Combinação | União dos ramos de palavras e caracteres | Até 160 mil características no total dos dois ramos |

Os classificadores utilizaram C igual a 1, ponderação balanceada das classes, semente 42 e máximo de 1.500 iterações. O otimizador foi liblinear na tarefa binária e lbfgs na tarefa temática. A seleção por validação favoreceu caracteres em ambos os estágios. Não houve calibração adicional dos escores de probabilidade.

Os baselines anteriores de e-mail usaram palavras de 1 a 2 termos e regressão logística balanceada, com máximo de 1.000 iterações. O detector auxiliar SpamShield usou caracteres de 2 a 5 posições, analisador char, até 500 mil características e regressão logística liblinear balanceada. Esse analisador é diferente do char_wb empregado no real-v2.

### 5 2 Papel da IA na anotação

O ChatGPT foi utilizado para atribuir rótulos temáticos experimentais durante a preparação dos dados autorizados. Esses rótulos são supervisão produzida por IA, com procedência registrada e sem marcação de revisão humana. A confiança declarada na anotação é um julgamento subjetivo e não uma probabilidade validada.

A inferência operacional utiliza os modelos clássicos locais. O projeto não executa um modelo generativo para classificar cada mensagem recebida, nem emprega Transformers no pipeline treinado. Também não há treinamento contínuo automático a partir da caixa postal.

## 6 Seleção dos modelos e dos limiares

Para spam, a comparação priorizou o maior F1 macro da fonte com pior desempenho na validação. O F1 macro global foi utilizado como desempate. Esse critério procura reduzir a dependência de uma média global favorável quando há diferença entre corpora. Para temas, a escolha utilizou o F1 macro na validação entre as classes treinadas.

Os limiares foram avaliados apenas na validação. Para sugerir spam, a grade foi 0,50; 0,60; 0,70; 0,80; 0,85; 0,90; 0,92; 0,94; 0,96; 0,98; 0,99; 0,995; 0,999 e 1,00. Exigiram-se precisão de spam de pelo menos 98%, taxa de falsos positivos de no máximo 1% por fonte e pelo menos dez sugestões. Entre os candidatos admissíveis, foi maximizada a revocação, com preferência pelo limiar mais alto em caso de empate. O valor selecionado foi 0,70.

O limiar de encaminhamento a legítimos foi escolhido na grade 0,01; 0,02; 0,05; 0,10; 0,15; 0,20; 0,25 e 0,30. Foram exigidos pelo menos 20 aceites, no máximo 1% de spam entre os aceites e passagem de no máximo 1% do spam da validação. Foi selecionado 0,30, buscando a maior cobertura admissível.

Para aceitação temática, foram testados 0,40; 0,50; 0,60; 0,70; 0,80; 0,90 e 0,95, exigindo pelo menos dez aceites e concordância mínima de 90% com os rótulos de IA da validação. Nenhum limiar satisfez o critério. A aceitação automática de categorias foi explicitamente desabilitada; o valor de bloqueio 1,00 é acompanhado de uma configuração de desativação.

Os valores são critérios empíricos de validação e não garantias de qualidade em novas caixas postais. No modo ativo, uma camada adicional encaminha todas as mensagens para revisão, enquanto faltar referência humana independente. O canal SMS mantém seus limiares provisórios próprios, de 0,90 para spam e 0,20 para legítimos, sem herdar os limiares ajustados para e-mail.

## 7 Fluxo de classificação

1. Receber texto, arquivo EML ou MBOX, ou mensagem coletada pelo adaptador Gmail somente leitura.
2. Extrair assunto e corpo e aplicar a mesma normalização utilizada no treinamento.
3. Selecionar o canal de e-mail ou, explicitamente, o detector auxiliar de mensagens curtas.
4. Transformar o conteúdo com o TF-IDF já ajustado e estimar o escore de spam. Se não houver características reconhecidas no vocabulário, encaminhar para revisão.
5. No canal de e-mail, produzir hipótese de spam para escore maior ou igual a 0,70; encaminhar escores intermediários para revisão; considerar a etapa temática quando o escore for menor ou igual a 0,30.
6. Estimar a categoria temática candidata e verificar vocabulário, limiar e configuração de aceitação. No real-v2, a aceitação temática permanece desabilitada.
7. Aplicar a política do modo ativo, que conserva as hipóteses e exige revisão humana para todos os casos. Exibir o resultado sem alterar mensagens na caixa postal.

O painel pode apresentar uma categoria candidata condicionada à mensagem ser legítima mesmo quando o detector sugere spam. Essa informação auxilia a revisão e não equivale a aceitar um tema para spam. Nenhuma etapa aplica labels, move ou exclui e-mails no Gmail.

## 8 Avaliação e controle experimental

A avaliação binária registra matriz de confusão, acurácia, precisão, revocação, F1 de spam, F1 macro e taxa de falsos positivos. Precisão corresponde à proporção de spam verdadeiro entre as previsões de spam; revocação corresponde à proporção de spam detectado entre os exemplos de spam da referência. A taxa de falsos positivos considera mensagens legítimas incorretamente previstas como spam.

Para temas, são calculadas métricas por categoria e F1 macro, com indicação das classes sem exemplos no conjunto avaliado. Enquanto os rótulos forem de IA, os resultados devem ser descritos como concordância com essa referência. Na política com abstenção, também são relevantes a cobertura das sugestões e a proporção de mensagens encaminhadas para revisão. A métrica de um estágio isolado não representa o acerto do fluxo completo.

Modelos, limiares e hashes dos arquivos foram congelados antes da anotação do teste temático reservado. Os resultados finais não foram utilizados para modificar esse experimento. Novas iterações exigem novos conjuntos de avaliação. O diagnóstico de spam utiliza corpora históricos já observados e a simulação recente permanece sem referência humana; nenhum deles comprova generalização para português atual ou para todos os provedores.

O procedimento disponível para avaliação humana exige identificação declarada do revisor e confirmação de revisão de conteúdo e privacidade. Ele separa exemplos com sobreposição detectada ao treinamento e não ajusta modelos ou limiares. Essa avaliação está pendente na etapa descrita.

## 9 Ambiente computacional e privacidade

A implementação utiliza Python, scikit-learn e joblib, com interface local em Streamlit. Os registros dos experimentos de 5 de novembro de 2026 indicam Python 3.13.14 e scikit-learn 1.9.1. Os pipelines ajustados são persistidos para reaplicar exatamente a mesma transformação durante a inferência. A semente 42, as versões, a procedência e os hashes dos insumos e artefatos são registrados para apoiar a reprodução.

Credenciais OAuth, tokens, mensagens privadas, anotações e modelos derivados de dados privados ficam fora do versionamento. Apenas dados demonstrativos inventados, código, documentação e relatórios agregados apropriados são mantidos no repositório. A autorização Gmail limita-se à leitura; o sistema não envia mensagens nem modifica a caixa postal.

Os procedimentos detalhados e os agregados podem ser conferidos em docs/04-data-strategy.md, docs/06-ml-plan.md, docs/07-evaluation.md e docs/14-real-development.md. Os relatórios dos experimentos anteriores estão em docs/experiments. A revisão humana independente e a ampliação da avaliação temática e em e-mails recentes constituem etapas futuras.

## Referências

[1] METSIS, V.; ANDROUTSOPOULOS, I.; PALIOURAS, G. Enron-Spam datasets. Fonte original. Disponível em: https://www2.aueb.gr/users/ion/data/enron-spam/. Distribuição utilizada por meio do espelho indicado em [2].

[2] MWIECHMANN. Enron spam data. Espelho CSV e documentação de transformação. Disponível em: https://github.com/MWiechmann/enron_spam_data. Acesso em: 5 nov. 2026.

[3] APACHE SPAMASSASSIN. Public Corpus README. Disponível em: https://spamassassin.apache.org/old/publiccorpus/readme.html. Acesso em: 5 nov. 2026.

[4] KUMARA, Isuranga Nipun. SPAM + Genuine Mail contents dataset. Kaggle, versão 1, 2025. Disponível em: https://www.kaggle.com/datasets/isuranga/spam-genuine-mail-contents-dataset. Metadados públicos arquivados localmente durante a obtenção do dataset. Consulta em: 5 nov. 2026.

[5] M-ARJUN. SpamShield-Datasets. Hugging Face. Disponível em: https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets. Revisão fixada indicada na seção 2 3. Acesso em: 5 nov. 2026.

[6] SCIKIT-LEARN DEVELOPERS. TfidfVectorizer. Documentação da versão 1.9.1. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html. Acesso em: 5 nov. 2026.

[7] SCIKIT-LEARN DEVELOPERS. LogisticRegression. Documentação da versão 1.9.1. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. Acesso em: 5 nov. 2026.
