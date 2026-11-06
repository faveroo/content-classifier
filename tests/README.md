# Testes

Execute `python -m pytest` após instalar `.[dev]`.
Cobertura de schema, duplicatas e grupos entre splits, isolamento do treino,
limites da política, vocabulário desconhecido, persistência, relatórios e CLI.
Não se exige acurácia em frases de treino nem se confunde desempenho sintético
com qualidade em mensagens reais. A CI executa testes e treino em Python 3.11 e 3.13.

O fluxo genérico inclui testes offline de parsing EML/MBOX/MIME/HTML, descarte de anexos,
separação de fontes e templates, isolamento de ambos os fits, opt-in de procedência,
importação para revisão e roteamento entre detectores. Não é necessário baixar corpora
ou instalar extras Gmail/Streamlit/multilingual para executar os testes unitários.

Real-v2 adiciona seleção apenas por validação, isolamento de grupos, restrição do
teste antes de congelamento, hashes/avaliação única, procedência AI, opt-ins, ausência
de acurácia fabricada em mensagens sem rótulo, revisão ativa obrigatória e avaliação
humana com cobertura parcial. OAuth expirado só abre consentimento no comando explícito.
105 testes passaram na entrega real-v2; AppTest verificou a interface privada.
