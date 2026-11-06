# AGENTS.md

Este arquivo contém instruções para qualquer assistente de IA, agente de código ou colaborador que trabalhe neste repositório.

## Regra principal

O projeto tem **baseline local com dados sintéticos** (2026-10-26) e **etapa Gmail OAuth somente leitura, coleta privada e revisão humana** autorizada em 2026-10-27.

Python está autorizado para schema, baseline local, CLI e testes. OAuth Gmail somente leitura e preparação de dados reais privados estão autorizados; conexão depende de configuração no Google Cloud e consentimento no navegador. Escrita no Gmail exige uma etapa explícita posterior. Execute `python -m pytest` antes de concluir mudanças de código.

Em 2026-10-27 o usuário também autorizou explicitamente o ChatGPT a classificar a
amostra pendente, treinar e testar. Rótulos de IA devem ter procedência própria
(`gmail_ai_labeled`), sem fingir revisão humana ou apresentar concordância como
acurácia validada por pessoas. O fluxo experimental exige opt-in separado; o fluxo
de dataset humano continua exigindo revisão. Artefatos reais permanecem privados.

Em 2026-11-05 o usuário autorizou seguir o roteiro com autonomia no projeto: rotulagem
experimental por IA das amostras, desenvolvimento com fontes reais, comparação de
baselines clássicos, escolha por validação e simulação reservada. Usar procedência
`email_ai_labeled`, opt-ins de treino e artefatos privados. Não fabricar revisão humana.
O experimento real-v2 fica congelado; testes já observados são diagnósticos para
trabalhos futuros. O modo ativo exige revisão enquanto faltar referência humana.
Gmail permanece somente leitura; consentimento foi renovado pelo usuário no navegador.

## Antes de realizar mudanças

Leia, nesta ordem:

1. `README.md`
2. `docs/STATUS.md`
3. `docs/01-vision.md`
4. `docs/02-scope.md`
5. o documento específico da área alterada
6. `docs/11-decisions.md`

## Fonte de verdade

- visão e objetivos: `docs/01-vision.md`
- escopo: `docs/02-scope.md`
- arquitetura: `docs/03-architecture.md`
- dados e dataset: `docs/04-data-strategy.md`
- categorias: `docs/05-label-taxonomy.md`
- Machine Learning: `docs/06-ml-plan.md`
- avaliação: `docs/07-evaluation.md`
- Gmail: `docs/08-gmail-integration.md`
- segurança e privacidade: `docs/09-privacy-security.md`
- etapas: `docs/10-roadmap.md`
- decisões consolidadas: `docs/11-decisions.md`
- estado atual: `docs/STATUS.md`

## Regras arquiteturais

- separar integração com Gmail da lógica de Machine Learning;
- separar treinamento de inferência;
- não acoplar o dataset diretamente à caixa de e-mail em produção;
- registrar métricas antes de aumentar complexidade do modelo;
- usar confiança mínima antes de ações automáticas;
- priorizar reversibilidade das ações realizadas no Gmail;
- evitar armazenar dados pessoais desnecessários;
- nunca versionar credenciais OAuth, tokens, mensagens privadas ou datasets reais contendo informações sensíveis.

## Machine Learning

A abordagem inicial deverá ser um baseline de NLP clássico. Não introduzir Transformers ou modelos grandes sem antes comparar objetivamente o resultado com o baseline.

## Atualização da documentação

Sempre que uma decisão importante for tomada:

1. atualizar o documento temático correspondente;
2. registrar a decisão em `docs/11-decisions.md`;
3. atualizar `docs/STATUS.md` quando houver mudança de etapa.
