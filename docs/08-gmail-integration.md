# Integração com Gmail

Em 2026-11-05, o token anterior retornou `invalid_grant`. O comando explícito
`gmail-auth` agora permite novo consentimento quando a renovação falha, mantendo o
token antigo até verificar os novos escopos. A coleta nunca abre o navegador por conta
própria. O usuário concluiu o consentimento `gmail.readonly`, seguido de coleta privada
de 100 mensagens recentes para simulação, sem alterações na caixa.

## Objetivo

Usar a Gmail API oficial para ler mensagens necessárias à classificação e aplicar labels autorizadas.

## Autenticação

OAuth 2.0.

Nenhuma credencial deverá ser versionada no repositório.

## Permissões

A aplicação deverá solicitar somente os escopos estritamente necessários.

## Fluxo futuro

1. autenticar usuário;
2. buscar mensagens candidatas;
3. extrair remetente, assunto e corpo;
4. normalizar conteúdo;
5. executar classificadores;
6. aplicar política de confiança;
7. adicionar label apropriada;
8. registrar resultado sem armazenar conteúdo sensível desnecessariamente.

## Segurança operacional

Na fase inicial, não:

- excluir mensagens;
- marcar diretamente como spam oficial;
- remover labels existentes;
- arquivar automaticamente mensagens importantes.

Preferir labels `IA/*`, que tornam as decisões reversíveis e auditáveis.

## Etapa somente leitura implementada em 2026-10-27

O usuário escolheu OAuth somente leitura para preparar amostras reais. O adaptador
opcional usa exclusivamente `https://www.googleapis.com/auth/gmail.readonly`,
cliente OAuth Desktop dedicado e callback loopback com PKCE. Credenciais e tokens
ficam fora do repositório; tokens com escopos adicionais são rejeitados.

`gmail-auth` abre o consentimento; `gmail-collect` apenas reutiliza/renova um token.
A coleta usa `users.messages.list` e `users.messages.get(format=full)`, com consulta
explícita, paginação e limite de 1–500 mensagens consultadas. Spam/Lixeira ficam fora
por padrão e sua inclusão exige flag. Não há busca separada de anexos nem métodos de escrita.

`predict-batch` faz simulação sobre a coleta local, sem conexão ao Gmail.
Não há criação/aplicação de labels nesta implementação. Configuração Cloud e
consentimento real concluídos em 2026-10-27. Primeira coleta de 50 mensagens recentes
e simulação offline concluídas, com todas encaminhadas para revisão e nenhuma
alteração na caixa. Revisão de privacidade e rótulos ainda pendente.
Instruções e referências oficiais no
[guia de configuração](12-gmail-readonly-setup.md).
