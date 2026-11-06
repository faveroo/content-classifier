# Dados

Somente `examples/synthetic-v1.json` é permitido no Git por padrão. Seus 80 exemplos
foram escritos manualmente e são inteiramente fictícios. Cada mensagem tem objetivo
distinto e grupo próprio; futuras variações do mesmo template/thread devem compartilhar
o group_id e permanecer no mesmo split. Distribuição: 40 treino, 20 validação, 20 teste,
com quatro/dois/dois exemplos por categoria e de spam. Consulte o schema em
`docs/04-data-strategy.md`. Coletas privadas e datasets temáticos reais devem ficar
fora do repositório, conforme `docs/12-gmail-readonly-setup.md`; nunca nesta pasta.

Corpora públicos de experimento podem ficar em `email/` e `hf-cache/`, explicitamente
ignorados pelo Git. Nunca adicionar os arquivos brutos ou os datasets derivados ao
versionamento. Procedência, limites de direitos e comandos em `docs/13-messages-email.md`.
