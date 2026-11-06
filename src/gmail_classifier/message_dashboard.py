"""Offline interface for independent spam and thematic category stages."""

from pathlib import Path
import json

import streamlit as st

from .baseline import Baseline, load_model
from .dataset import CATEGORIES
from .mail_input import parse_email
from .message_model import classify_message
from .spamshield import load_spamshield_model
from .active_model import active_configuration, require_review
from .policy import Policy
from .private_storage import private_home


def render():
    st.title("Mensagens e e-mails")
    st.caption("Classificação local, independente do provedor")
    active = None
    if (private_home() / "active-model.json").exists():
        try:
            active = active_configuration()
        except (OSError, ValueError, KeyError):
            st.warning("O experimento privado não pôde ser validado; confira seus arquivos locais.")
    model_path = Path(st.text_input("Modelo treinado neste ambiente", str(active["model"]) if active else "models/messages-v1.joblib"))
    use_review_gate = bool(active and model_path.resolve() == active["model"].resolve())
    st.caption("Carregue apenas modelos próprios e confiáveis: arquivos joblib podem executar código.")
    if not model_path.exists():
        st.info("Treine o modelo de e-mail primeiro; consulte docs/13-messages-email.md.")
        return
    try:
        model = load_model(model_path)
    except (OSError, ValueError):
        st.error("Modelo indisponível ou incompatível; retreine neste ambiente.")
        return
    source = (model.provenance or {}).get("category_source", "unknown")
    if source == "synthetic":
        st.warning("As categorias foram treinadas com exemplos inventados e servem para demonstrar o fluxo. Precisam de e-mails reais rotulados para avaliação de qualidade.")
    elif source.endswith("ai_labeled"):
        st.warning("Categorias experimentais com rótulos de IA; não há acurácia validada por revisão humana independente.")
    spam_source = (model.provenance or {}).get("spam_source", "unknown")
    if use_review_gate:
        st.warning("Experimento com fontes reais históricas e temas rotulados por IA. Validação humana independente pendente: todas as hipóteses passam por revisão.")
    elif spam_source == "isuranga/spam-genuine-mail-contents-dataset":
        st.warning("Este detector foi treinado com e-mails sintéticos do Kaggle. Acerto no corpus gerado não comprova qualidade em mensagens reais. Resultados são sugestões para revisão.")
    else:
        st.warning("O detector de e-mail é experimental. O teste externo mostrou generalização insuficiente. Resultados são sugestões para revisão.")
    predict_tab, metrics_tab = st.tabs(["Classificar", "Avaliação"])
    with predict_tab:
        channel = st.radio("Tipo de entrada", ["E-mail", "Mensagem de texto"], horizontal=True)
        upload = st.file_uploader("Arquivo opcional", type=["eml", "txt"])
        subject = st.text_input("Assunto (opcional)")
        body = st.text_area("Mensagem", height=220)
        if st.button("Classificar localmente", type="primary"):
            try:
                if upload:
                    raw = upload.getvalue()
                    if len(raw) > 2_000_000:
                        raise ValueError("Arquivo excede 2 MB")
                    message = parse_email(raw) if upload.name.lower().endswith(".eml") else {"subject": subject, "body": raw.decode("utf-8")[:20_000]}
                else:
                    message = {"subject": subject, "body": body}
                if channel == "Mensagem de texto":
                    active_predictor = Baseline(load_spamshield_model("models/spamshield-multilingual.joblib"), model.category,
                        Policy(category_threshold=model.policy.category_threshold), model.provenance)
                else:
                    active_predictor = model
                result = classify_message(active_predictor, **message, conditional_topic=use_review_gate)
                if use_review_gate:
                    result = require_review(result, active)
                st.metric("Probabilidade de spam", f"{result['spam_probability']:.1%}")
                if result["category"]:
                    st.write(f"Categoria: **{result['category']}** · confiança {result['category_confidence']:.1%}")
                elif result["category_candidate"]:
                    title = "Hipótese de categoria se for legítima" if result.get("category_candidate_condition") else "Categoria candidata para revisão"
                    st.write(f"{title}: **{result['category_candidate']}** · confiança {result['category_candidate_confidence']:.1%}")
                st.write("Sugestão: **" + result["label"] + "**")
                if result["action"] == "review":
                    st.info("Revisão necessária: confiança insuficiente ou conteúdo fora do vocabulário aprendido.")
                st.caption("Probabilidades sem calibração independente. Nenhuma mensagem é enviada ou alterada.")
            except (OSError, ValueError, UnicodeError):
                st.error("Confira o texto, o arquivo e a disponibilidade do modelo para este tipo de entrada.")
        st.caption("Categorias disponíveis: " + ", ".join(CATEGORIES))
    with metrics_tab:
        default_report = active["report"] if use_review_gate else Path("reports") / (model_path.stem + ".json")
        report_path = Path(st.text_input("Relatório deste treinamento", str(default_report)))
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                st.write(report["warning"])
                rows = []
                for name, values in report.get("spam", report.get("spam_diagnostic", {})).items():
                    value = values["at_0_5"]
                    rows.append({"Conjunto": name, "Mensagens": value["count"],
                        "Acurácia": value["accuracy"], "Precision spam": value["spam_precision"],
                        "Recall spam": value["spam_recall"], "Falsos positivos": value["false_positives"]})
                st.dataframe(rows, hide_index=True)
                st.caption("Métricas binárias no limiar 0,5. O fluxo de sugestões usa limiares conservadores e pode pedir revisão.")
                st.write(report.get("category_warning", report.get("warning", "")))
                st.json(report.get("category", report.get("topic_test", {})))
            except (OSError, ValueError, KeyError):
                st.error("Relatório inválido ou incompatível com este painel.")
