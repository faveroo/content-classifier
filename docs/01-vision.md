# Visão do Projeto

## Problema

Caixas de e-mail recebem mensagens de naturezas muito diferentes. Filtros tradicionais ajudam, mas não representam necessariamente as categorias e prioridades específicas do usuário.

## Visão

Criar um classificador de mensagens e e-mails que aprenda a partir de exemplos rotulados,
independente do provedor. A organização de uma conta Gmail é uma integração opcional.

O projeto deve servir também como estudo prático de:

- Machine Learning supervisionado;
- NLP;
- preparação e avaliação de datasets;
- integração OAuth 2.0;
- Gmail API;
- operação segura de modelos em dados reais.

## Resultado desejado

Ao receber uma mensagem, o sistema deverá ser capaz de produzir algo conceitualmente semelhante a:

```text
spam_probability: 0.03
category: Financeiro
category_confidence: 0.92
action: apply_label
label: IA/Financeiro
```

O formato técnico definitivo será definido durante a implementação.

## Princípios

1. Segurança antes de automação.
2. Métricas antes de complexidade.
3. Dados de qualidade antes de quantidade.
4. Revisão humana para casos incertos.
5. Privacidade por padrão.
