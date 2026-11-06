# Privacidade e Segurança

## Princípios

- minimização de dados;
- privilégio mínimo;
- credenciais fora do Git;
- ações reversíveis;
- logs sem conteúdo sensível;
- revisão humana em baixa confiança.

## Gmail

Tokens de acesso e refresh tokens são segredos e nunca devem aparecer no repositório.

## Dataset

Se e-mails reais forem utilizados para treinamento, o dataset deverá permanecer em armazenamento local ou privado apropriado, salvo quando estiver devidamente anonimizado e houver autorização para publicação.

## Logs

Evitar registrar corpo integral de mensagens. Preferir:

- ID interno;
- categoria prevista;
- confiança;
- ação tomada;
- versão do modelo;
- timestamp.

## Repositório público

Todo conteúdo versionado deve ser seguro para exposição pública.

## Fluxo privado de leitura (2026-10-27)

Armazenamento padrão: `%LOCALAPPDATA%\GmailEmailClassifier` no Windows ou
`~/.local/share/GmailEmailClassifier` nos demais ambientes. Caminhos privados são
resolvidos e rejeitados se estiverem no checkout ou em uma árvore Git detectável.
No Windows, a proteção depende das permissões herdadas da conta; o código não
implementa criptografia nem garante confidencialidade de pastas compartilhadas.

Filtros substituem e-mails, URLs e alguns números, mas não todos os dados pessoais.
O arquivo de revisão permanece privado. Não são persistidos cabeçalhos de remetente/
destinatário, IDs Gmail ou conteúdo bruto. Pseudônimos não possuem mapa reverso salvo.
Revise nomes, assinaturas, dados financeiros e segredos antes de preparar o dataset.

Saída de coleta/autenticação contém apenas status e contagens. Erros externos são
substituídos por mensagens genéricas, sem imprimir respostas OAuth/API. Não habilitar
logs HTTP de depuração ao usar dados reais. Artefatos privados e relatórios ficam fora
do Git. Apague coletas/datasets/modelos/relatórios quando não forem mais necessários;
revogue o acesso na Conta Google e remova o token ao encerrar a integração. Não há
rotina de retenção automática ou exclusão segura do disco implementada.

O formulário de revisão usa servidor HTTP restrito a `127.0.0.1`, sem assets remotos
ou scripts. Conteúdo das mensagens é escapado como texto. Cabeçalhos impedem cache,
embedding em frames e carregamento de recursos externos. Salvamentos exigem Host e
Origin locais e token CSRF; requisições não entram em logs. A interface é acessível
a processos da mesma máquina enquanto estiver aberta; encerre-a ao terminar a revisão.
O backup `review.before-review.json` permanece privado, com o texto anterior à edição.

A classificação da amostra pelo ChatGPT foi explicitamente solicitada pelo titular
em 2026-10-27. O conteúdo necessário à anotação foi consultado nessa tarefa; o treino
e a inferência do baseline continuam locais, sem chamar um serviço de LLM no código.
Dados derivados, justificativas, scripts específicos da amostra e métricas detalhadas
permanecem fora do Git. Não declarar revisão humana de privacidade onde houve apenas
minimização assistida pela IA, que não garante anonimização completa.

EML/MBOX e mensagens de qualquer provedor seguem a mesma proteção: coleções para
revisão e derivados temáticos reais fora de árvores Git. Importação não confirma
revisão nem faz chamadas de rede. Texto de HTML é extraído sem renderização ou
carregamento de recursos. Corpora públicos brutos, datasets e modelos ficam ignorados;
somente métricas agregadas são publicadas. O formulário e a nova CLI não alteram caixas.

No real-v2, anotações, modelos, coletas e relatórios completos ficam em armazenamento
privado. O registro ativo verifica o hash do modelo e a camada operacional exige
revisão para todas as hipóteses. As coleções humanas permanecem sem labels/flags
preenchidas pela IA. `experiment_v2 view` é uma ferramenta de leitura para anotação
autorizada: sua saída contém texto minimizado privado e não deve ser publicada como log.
O painel fica em loopback; nenhum artefato privado foi enviado a hospedagem externa.
