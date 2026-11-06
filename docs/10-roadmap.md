# Roadmap

## Etapa 0 — Documentação

- [x] definir visão inicial;
- [x] definir arquitetura conceitual;
- [x] registrar estratégia de dados;
- [x] definir baseline de ML;
- [x] documentar segurança;
- [ ] revisar taxonomia com casos reais;
- [ ] definir dataset inicial.

## Etapa 1 — Dataset

- [ ] definir schema;
- [ ] coletar exemplos permitidos;
- [ ] criar processo de anonimização;
- [ ] criar splits de treino/validação/teste;
- [ ] documentar distribuição das classes.

## Etapa 2 — Baseline de spam

- [ ] implementar pré-processamento;
- [ ] implementar TF-IDF;
- [ ] treinar Logistic Regression;
- [ ] medir métricas;
- [ ] registrar baseline.

## Etapa 3 — Categorização

- [ ] treinar classificador multiclasse;
- [ ] avaliar categorias confundidas;
- [ ] ajustar taxonomia;
- [ ] definir threshold para revisão.

## Etapa 4 — Gmail em modo seguro

- [ ] configurar projeto Google Cloud;
- [ ] configurar OAuth;
- [ ] ler mensagens de teste;
- [ ] criar labels `IA/*`;
- [ ] classificar sem alterar mensagens;
- [ ] habilitar aplicação de labels.

## Etapa 5 — Human-in-the-loop

- [ ] identificar classificações incertas;
- [ ] registrar correções humanas;
- [ ] preparar exemplos corrigidos para retreinamento;
- [ ] versionar métricas e modelos.

## Etapa 6 — Experimentos avançados

- [ ] testar alternativas clássicas;
- [ ] avaliar embeddings;
- [ ] comparar Transformers com baseline;
- [ ] avaliar custo/benefício.

## Entrega local de 2026-10-26

- [x] encerrar bloqueio de implementação da Etapa 0 por autorização do usuário;
- [x] taxonomia v1 com critérios e ambiguidade (validação real pendente);
- [x] schema e validação de integridade;
- [x] dataset público sintético e splits explícitos, com distribuição documentada;
- [x] baseline de spam e categorização, treinamento e inferência separados;
- [x] política provisória de revisão humana, métricas e registro do experimento;
- [x] CLI local, testes, CI e documentação de execução;
- [ ] coleta real autorizada, anonimização e revisão da taxonomia com dados representativos;
- [ ] calibração e escolha de thresholds na validação, antes de avaliação final;
- [ ] integração Gmail em etapa futura explícita.

Os itens anteriores do roadmap referentes a dados representativos continuam pendentes.
O baseline sintético cobre a engenharia das Etapas 1–3, sem afirmar conclusão empírica
dessas etapas em produção.

## Etapa de leitura autorizada em 2026-10-27

- [x] implementar OAuth Desktop somente leitura, separado do ML;
- [x] implementar coleta limitada com consulta explícita e IDs locais;
- [x] implementar minimização de texto e exigir revisão humana complementar;
- [x] preparar contrato privado revisado e treino com opt-in fora do Git;
- [x] implementar simulação offline em lote, sem alterações na caixa;
- [x] testar OAuth/API com objetos simulados e documentar configuração;
- [x] configurar projeto/cliente no Google Cloud e autenticar conta real;
- [x] executar primeira coleta privada de 50 mensagens e simulação offline (2026-10-27);
- [ ] coletar e revisar amostras reais representativas;
- [ ] revisar taxonomia, medir resultados e escolher thresholds na validação;
- [ ] avaliar conjunto de teste final sem usá-lo para ajustes;
- [ ] autorizar e implementar aplicação reversível de labels em etapa separada.

A implementação permite iniciar a coleta; os itens de execução real e avaliação
empírica continuam pendentes. A lista da entrega anterior registra seu escopo histórico.

## Experimento com supervisão de IA autorizado em 2026-10-27

- [x] atribuir rótulos pelo ChatGPT a pedido explícito, com procedência separada;
- [x] agrupar conversas/templates e separar treino/validação/teste sem sobreposição;
- [x] treinar categorizador clássico com o treino privado e preservar detector anterior de spam;
- [x] comparar categorizador e fluxo completo ao baseline nos mesmos splits;
- [x] registrar métricas e limitações em artefatos privados, sem alegar referência humana;
- [ ] obter avaliação humana independente e ampliar cobertura de categorias/spam.

Essa entrega experimental não conclui a validação empírica com referência humana nem
autoriza escrita no Gmail. O teste consultado não deve orientar ajustes futuros.

## Roteiro com autonomia — 2026-11-05

- [x] rotular experimentalmente a amostra pública com procedência de IA e exclusões de ambiguidade;
- [x] separar amostra temática dos dados de spam, incluindo quarentena dos grupos relacionados;
- [x] combinar fontes reais históricas como desenvolvimento já observado;
- [x] comparar palavras, caracteres e combinação, selecionando somente por validação;
- [x] escolher política empírica por validação e desabilitar aceitação temática sem suporte suficiente;
- [x] congelar modelos e entradas antes da anotação do teste temático reservado;
- [x] avaliar o teste uma vez, registrar baixa concordância temática sem mudar parâmetros;
- [x] renovar OAuth com consentimento humano e simular 100 mensagens recentes sem fit ou escrita;
- [x] disponibilizar modo ativo de revisão e avaliação humana com classes ausentes;
- [ ] obter referência humana independente e avaliar mensagens recentes/português/outros provedores;
- [ ] ampliar dados temáticos, reservar novo teste e comprovar ganho antes de aceitação automática;
- [ ] calibrar probabilidades com referência apropriada;
- [ ] autorizar implementação de escrita reversível no Gmail em etapa separada.

Concluída a execução experimental automatizável; validação humana e qualidade de
produção não foram presumidas. [Guia](14-real-development.md).
