"""Local Streamlit dashboard for SpamShield metrics and interactive predictions."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gmail_classifier.spamshield import load_spamshield_model, predict


st.set_page_config(page_title="SpamShield · Classificador multilíngue", page_icon="🛡️",
                   layout="wide")
mode = st.sidebar.radio("Experimento", ["Mensagens e e-mails", "SpamShield SMS"])
if mode == "Mensagens e e-mails":
    from gmail_classifier.message_dashboard import render
    render()
    st.stop()
st.title("SpamShield · Classificador multilíngue")
st.caption("Experimento local com corpus de SMS em 23 idiomas")

model_path = Path("models/spamshield-multilingual.joblib")
report_path = Path("reports/spamshield-multilingual.json")
if not model_path.exists() or not report_path.exists():
    st.info("Treine o experimento primeiro. Os arquivos do modelo e do relatório ficam locais e não são versionados.")
    st.code("python -m pip install -e '.[multilingual,dashboard]'\n"
            "hf auth login\n"
            "# Aceite as condições na página do dataset antes de baixar\n"
            "python -m gmail_classifier.cli train-spamshield")
    st.link_button("Abrir dataset no Hugging Face",
                   "https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets")
    st.stop()

try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = load_spamshield_model(model_path)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Não foi possível carregar os artefatos locais: {exc}")
    st.stop()

st.warning(report["dataset_note"])
top = st.columns(4)
top[0].metric("Mensagens", f"{report['total_count']:,}")
top[1].metric("Idiomas", str(len(report["languages"])))
top[2].metric("F1 macro · teste", f"{report['splits']['test']['macro_f1']:.3f}")
top[3].metric("F1 spam · teste", f"{report['splits']['test']['spam_f1']:.3f}")

tab_predict, tab_metrics, tab_data = st.tabs(["Classificar texto", "Avaliação", "Dataset"])
with tab_predict:
    st.subheader("Experimente uma mensagem")
    st.write("Cole texto de exemplo. A inferência acontece localmente neste ambiente.")
    text = st.text_area("Texto da mensagem", height=180, placeholder="Digite ou cole uma mensagem…")
    if st.button("Classificar", type="primary", disabled=not text.strip()):
        result = predict(model, text)
        spam = result["label"] == "spam"
        st.success("Spam provável" if spam else "Ham provável")
        left, right = st.columns(2)
        left.metric("Probabilidade de spam", f"{result['spam_probability']:.1%}")
        right.metric("Probabilidade de ham", f"{result['ham_probability']:.1%}")
        st.progress(result["spam_probability"])
        st.caption("Probabilidades do classificador, sem calibração independente; use como experimento.")

with tab_metrics:
    split_name = st.selectbox("Conjunto", options=["test", "validation", "train"], index=0,
                              format_func=lambda value: {"test": "Teste", "validation": "Validação", "train": "Treino"}[value])
    metrics = report["splits"][split_name]
    cards = st.columns(4)
    cards[0].metric("Exemplos", f"{metrics['count']:,}")
    cards[1].metric("Acurácia", f"{metrics['accuracy']:.1%}")
    cards[2].metric("F1 macro", f"{metrics['macro_f1']:.3f}")
    cards[3].metric("F1 spam", f"{metrics['spam_f1']:.3f}")
    st.subheader("Desempenho por idioma")
    language_rows = [{"Idioma": language, "Amostras": values["count"],
                      "Spam": values["spam_count"], "F1 spam": values["spam_f1"],
                      "F1 macro": values["macro_f1"], "Acurácia": values["accuracy"]}
                     for language, values in metrics["by_language"].items()]
    st.dataframe(language_rows, use_container_width=True, hide_index=True,
                 column_config={
                     "F1 spam": st.column_config.ProgressColumn("F1 spam", min_value=0, max_value=1, format="%.3f"),
                     "F1 macro": st.column_config.ProgressColumn("F1 macro", min_value=0, max_value=1, format="%.3f"),
                     "Acurácia": st.column_config.ProgressColumn("Acurácia", min_value=0, max_value=1, format="%.3f"),
                 })
    st.subheader("Matriz de confusão")
    st.caption("Linhas = classe real; colunas = classe prevista. Ordem: ham, spam.")
    st.dataframe({"Real / Prevista": ["ham", "spam"],
                  "Previsto ham": [metrics["confusion_matrix"][0][0], metrics["confusion_matrix"][1][0]],
                  "Previsto spam": [metrics["confusion_matrix"][0][1], metrics["confusion_matrix"][1][1]]},
                 use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Composição do experimento")
    st.write(f"Fonte: [{report['source']}](https://huggingface.co/datasets/{report['source']})")
    st.write(f"Licença: [{report['license']}](https://creativecommons.org/licenses/by/4.0/)")
    st.write(report["attribution"] + ". Os dados foram deduplicados por texto normalizado e separados em treino/validação/teste por idioma × rótulo.")
    st.write("Idiomas incluídos: " + ", ".join(report["languages"]))
    st.subheader("Distribuição por rótulo detalhado")
    category_rows = [{"Categoria detalhada": key, "Mensagens": value}
                     for key, value in report["category_distribution"].items()]
    st.bar_chart(pd.DataFrame(category_rows), x="Categoria detalhada", y="Mensagens")
    st.caption("O modelo usa apenas o rótulo binário do dataset. As categorias detalhadas aparecem para inspeção da composição, sem serem previstas.")
    with st.expander("Detalhes dos splits"):
        st.json({name: {"count": value["count"], "distribution": value["distribution"],
                        "language_count": value["language_count"]}
                 for name, value in report["splits"].items()})
        st.caption(f"SHA-256 do arquivo: {report['dataset_sha256']}")

st.divider()
st.caption("O corpus contém SMS, não e-mails. A avaliação não comprova desempenho em Gmail. CC-BY-4.0 exige atribuição da fonte.")
