# Taxonomia de Labels — v1

Uma mensagem tem `is_spam` booleano. Spam tem `category: null`; mensagens legítimas
recebem exatamente uma categoria abaixo. Classificar pela finalidade principal e ação
esperada do destinatário, não por palavras isoladas nem pelo fato de ser automático.
Versão operacional para exemplos sintéticos; validar utilidade com dados autorizados no futuro.

| Categoria | Inclui / exemplo | Não inclui |
|---|---|---|
| Financeiro | boletos, notas fiscais, cobranças, pagamentos; corrigir valor de fatura | negociação de proposta ainda sem obrigação financeira |
| RH | férias, benefícios, recrutamento, admissão, folha e contracheque | pagamento de fornecedor |
| Suporte | chamado, investigação e solução de problema solicitado por usuário | alerta automático sem pedido de atendimento |
| Sistemas | disponibilidade, acesso, segurança, integração e manutenção técnica | chamado humano solicitando solução |
| Comercial | propostas, vendas, negociação e fornecedores na fase de oferta | cobrança de compra já realizada |
| Administrativo | contratos, documentos, instalações e logística administrativa | comunicação genérica sem demanda administrativa |
| Notificações | confirmação de evento ou atualização informativa sem ação específica | avisos de pagamento, RH ou infraestrutura com tema definido |
| Pessoal | conversa privada, família e compromissos pessoais | mensagem profissional enviada por pessoa física |
| Outros | tema legítimo fora das anteriores, como conteúdo cultural informativo | incerteza entre categorias conhecidas |

## Ambiguidade

1. Spam é avaliado primeiro. Publicidade não solicitada, fraude e mensagens indesejadas
   podem ser spam; uma oferta comercial esperada é Comercial.
2. Escolher o objetivo da ação principal. “Corrigir valor do boleto” é Financeiro;
   “portal não abre, investiguem o erro” é Suporte, mesmo citando um boleto.
3. Contracheque e férias são RH; pagamento de fornecedor é Financeiro.
4. Proposta de contrato é Comercial; assinatura e arquivo do contrato aprovado é Administrativo.
5. Alerta automático de servidor é Sistemas; pedido de ajuda para resolver falha é Suporte.
6. Confirmação de pagamento é Financeiro, mesmo automática. Notificações é residual
   para eventos sem tema mais específico; conversa sobre jantar em família é Pessoal.
7. Havendo dois pedidos de igual importância, registrar para revisão humana antes de
   entrar no dataset. Não usar Outros como substituto para incerteza.

## Labels sugeridas

`IA/SPAM`, `IA/Revisar` e `IA/<Categoria>` com a grafia exata da tabela.
A CLI apenas sugere labels. Nenhuma ação é executada no Gmail.

Na experiência real-v2, a taxonomia v1 foi mantida. Exemplos ambíguos são excluídos
da anotação experimental, sem usar Outros como atalho. Nove classes aparecem no
treino, mas Financeiro e Pessoal não são avaliados no pequeno teste reservado.
Baixa concordância impede aceitação automática temática; hipóteses continuam em revisão.
