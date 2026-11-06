"""Loopback-only human review. Email text never enters logs or remote services."""

import hashlib
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .dataset import CATEGORIES, FIELDS, SPLITS, validate_records
from .private_storage import private_path, write_private_json


REVIEW_FIELDS = FIELDS | {"human_reviewed", "privacy_reviewed", "exclude"}


def load_collection(path):
    raw = private_path(path).read_bytes()
    collection = json.loads(raw)
    if (not isinstance(collection, dict) or collection.get("format_version") != 1
            or collection.get("source") not in {"gmail_reviewed", "email_reviewed"}
            or not isinstance(collection.get("records"), list) or not collection["records"]):
        raise ValueError("Invalid review collection")
    for row in collection["records"]:
        if not isinstance(row, dict) or set(row) != REVIEW_FIELDS:
            raise ValueError("Unexpected review fields")
        if row["source"] != collection["source"]:
            raise ValueError("Mixed review sources")
        if any(not isinstance(row[key], str) for key in ("id", "group_id", "subject", "body")):
            raise ValueError("Invalid text or identifier")
        if any(type(row[key]) is not bool for key in ("exclude", "human_reviewed", "privacy_reviewed")):
            raise ValueError("Invalid review flags")
    return collection, hashlib.sha256(raw).hexdigest()


def progress(collection):
    rows = collection["records"]
    included = [r for r in rows if not r["exclude"]]
    reviewed = sum(r["human_reviewed"] and r["privacy_reviewed"] for r in included)
    coverage = {split: sorted({"SPAM", *CATEGORIES} - {
        "SPAM" if r["is_spam"] is True else r["category"]
        for r in included if r["split"] == split and type(r["is_spam"]) is bool
    }) for split in SPLITS}
    ready = False
    if reviewed == len(included) and included:
        try:
            validate_records([{key: r[key] for key in FIELDS} for r in included], allow_private=True)
            ready = True
        except (ValueError, TypeError):
            pass
    return {"total": len(rows), "excluded": len(rows) - len(included),
            "reviewed": reviewed, "pending": len(included) - reviewed,
            "missing_classes": coverage, "ready": ready}


def save_review(path, revision, index, form):
    """Validate first, preserve a backup, and reject stale forms before replacing a file."""
    path = private_path(path)
    collection, actual_revision = load_collection(path)
    if revision != actual_revision:
        raise ValueError("O arquivo mudou em outra janela. Recarregue antes de salvar.")
    rows = collection["records"]
    if not 0 <= index < len(rows):
        raise ValueError("Mensagem inexistente.")
    label = form.get("label", "")
    split = form.get("split", "")
    group = form.get("group_id", "").strip()
    subject, body = form.get("subject", ""), form.get("body", "")
    excluded = form.get("exclude") == "on"
    human = form.get("human_reviewed") == "on"
    privacy = form.get("privacy_reviewed") == "on"
    if label not in {"", "SPAM", *CATEGORIES} or split not in {"", *SPLITS}:
        raise ValueError("Categoria ou divisão inválida.")
    if not group or len(group) > 200 or max(len(subject), len(body)) > 20000:
        raise ValueError("Confira grupo e limites de texto (20 mil caracteres por campo).")
    if not excluded and human and (not label or not (subject + body).strip()):
        raise ValueError("Escolha a classificação e mantenha algum texto antes de confirmar.")
    old_group = rows[index]["group_id"]
    # Moving a thread into a template group must not break up the original thread.
    for row in rows:
        if row["group_id"] == old_group:
            row["group_id"] = group
        if row["group_id"] == group:
            row["split"] = split or None
    rows[index].update(subject=subject, body=body, is_spam=(label == "SPAM") if label else None,
                       category=label if label in CATEGORIES else None, exclude=excluded,
                       human_reviewed=human, privacy_reviewed=privacy)
    backup = path.with_name(path.stem + ".before-review.json")
    if not backup.exists():
        original, _ = load_collection(path)
        write_private_json(backup, original)
    descriptor, temporary = tempfile.mkstemp(prefix=".review-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(collection, stream, ensure_ascii=False, indent=2)
        # Do not clobber an external edit made during validation/backup creation.
        if hashlib.sha256(path.read_bytes()).hexdigest() != revision:
            raise ValueError("O arquivo mudou. Recarregue antes de salvar.")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return collection


def page(collection, revision, index, csrf):
    row = collection["records"][index]
    status = progress(collection)
    label = "SPAM" if row["is_spam"] is True else row["category"] or ""

    def options(values, selected):
        return "".join(f'<option value="{escape(value, quote=True)}"'
                       f'{" selected" if value == selected else ""}>{escape(title)}</option>'
                       for value, title in values)

    labels = options([("", "Escolha a classificação"), ("SPAM", "Spam"),
                      *((category, category) for category in CATEGORIES)], label)
    splits = options([("", "Definir depois"), ("train", "Treino"),
                      ("validation", "Validação"), ("test", "Teste reservado")], row["split"] or "")
    coverage = "".join(f'<li>{split}: {escape(", ".join(missing)) or "cobertura completa"}</li>'
                       for split, missing in status["missing_classes"].items())
    checks = "".join(f'<label class="check"><input type="checkbox" name="{key}"'
                     f'{" checked" if row[key] else ""}> {text}</label>' for key, text in [
                         ("human_reviewed", "Confirmo a classificação desta mensagem"),
                         ("privacy_reviewed", "Revisei o texto e removi dados pessoais e segredos desnecessários"),
                         ("exclude", "Excluir esta mensagem da amostra de treinamento")])
    return f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Revisão de e-mails</title>
<style>body{{font:16px system-ui;margin:0;background:#f3f5f8;color:#172238}}main{{max-width:1000px;margin:auto;padding:28px}}
header,form,aside{{background:white;padding:24px;border-radius:14px;margin-bottom:18px}}h1{{margin:0 0 10px}}p{{line-height:1.5}}
label{{display:block;margin:16px 0 7px;font-weight:600}}input,textarea,select,button{{font:inherit;box-sizing:border-box}}
input[type=text],textarea,select{{width:100%;padding:10px;border:1px solid #99a9bf;border-radius:6px}}textarea{{min-height:240px}}
.check{{font-weight:400}}button{{padding:12px 20px;background:#174ac0;color:white;border:0;border-radius:7px;cursor:pointer}}
nav{{display:flex;gap:22px;margin:18px 0}}a{{color:#174ac0}}summary{{cursor:pointer}}small{{color:#41516b}}</style>
<main><header><h1>Prepare o treinamento</h1><p>Mensagem {index + 1} de {status["total"]} ·
{status["reviewed"]} revisadas · {status["pending"]} pendentes · {status["excluded"]} excluídas</p>
<small>Revisão local. Nenhuma alteração no Gmail. Classifique pela finalidade principal da mensagem.</small></header>
<nav><a href="/?i={max(0, index - 1)}">← Anterior</a><a href="/?i={min(len(collection["records"]) - 1, index + 1)}">Próxima →</a></nav>
<form method="post" action="/save"><input type="hidden" name="csrf" value="{csrf}">
<input type="hidden" name="revision" value="{revision}"><input type="hidden" name="index" value="{index}">
<label for="subject">Assunto — edite para remover dados privados</label>
<input id="subject" name="subject" type="text" maxlength="20000" value="{escape(row["subject"], quote=True)}">
<label for="body">Texto da mensagem — revise nomes, assinaturas e detalhes confidenciais</label>
<textarea id="body" name="body" maxlength="20000">{escape(row["body"])}</textarea>
<label for="label">Classificação correta</label><select id="label" name="label">{labels}</select>
<details><summary>Guia das categorias</summary><p>Financeiro: cobranças e pagamentos; RH: férias, benefícios e recrutamento;
Suporte: pedido de solução de problema; Sistemas: alertas e manutenção técnica; Comercial: ofertas e negociação;
Administrativo: contratos e logística; Notificações: avisos sem tema específico; Pessoal: conversas privadas;
Outros: tema legítimo fora das anteriores. Se estiver em dúvida, deixe pendente ou exclua a mensagem.</p></details>
{checks}<details><summary>Separação de dados e agrupamento</summary>
<p>Distribua grupos aproximadamente em 60% treino, 20% validação e 20% teste. Mantenha conversas e templates semelhantes no mesmo grupo.
A divisão escolhida será aplicada a todo o grupo; ao unir grupos, as mensagens da conversa original serão mantidas juntas.</p>
<label for="group">Grupo de conversa ou template</label><input id="group" name="group_id" type="text" maxlength="200" required value="{escape(row["group_id"], quote=True)}">
<label for="split">Divisão do grupo</label><select id="split" name="split">{splits}</select></details>
<p><button type="submit">Salvar e próxima</button></p><small>Salvamento por mensagem. A primeira edição cria uma cópia de segurança local.</small></form>
<aside><h2>{"Pronto para preparar o dataset" if status["ready"] else "Ainda faltam etapas antes do treinamento"}</h2>
<p>Revisão, rótulos, divisões e cobertura por classe precisam estar completos.
Esta amostra inicial pode exigir mais coletas. Nenhum modelo foi treinado com ela ainda.</p>
<details><summary>Classes que faltam em cada divisão</summary><ul>{coverage}</ul></details>
<p>Quando terminar, avise no chat para validar a amostra e continuar o treinamento.</p></aside></main></html>'''


def make_server(path, port=0):
    path = private_path(path)
    load_collection(path)
    csrf = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No request paths, payloads or email content in terminal logs.

        def respond(self, code, body, content_type="text/html; charset=utf-8"):
            encoded = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(encoded)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, "Acesso recusado.")
            location = urlsplit(self.path)
            try:
                collection, revision = load_collection(path)
                if location.path == "/status":
                    return self.respond(200, json.dumps(progress(collection)), "application/json")
                if location.path != "/":
                    return self.respond(404, "Página inexistente.")
                index = int(parse_qs(location.query).get("i", ["0"])[0])
                if not 0 <= index < len(collection["records"]):
                    raise ValueError("Invalid index")
                self.respond(200, page(collection, revision, index, csrf))
            except (ValueError, OSError, TypeError, KeyError):
                self.respond(400, "Não foi possível abrir a amostra. Confira o arquivo local.")

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if not self.valid_host() or self.headers.get("Origin") != origin:
                return self.respond(403, "Acesso recusado.")
            if self.path != "/save":
                return self.respond(404, "Página inexistente.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 512000:
                    raise ValueError("Invalid size")
                data = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                if any(len(values) != 1 for values in data.values()):
                    raise ValueError("Duplicate form fields")
                form = {key: values[0] for key, values in data.items()}
                if not secrets.compare_digest(form.get("csrf", ""), csrf):
                    return self.respond(403, "Acesso recusado.")
                index = int(form["index"])
                collection = save_review(path, form["revision"], index, form)
                self.send_response(303)
                self.send_header("Location", f"/?i={min(index + 1, len(collection['records']) - 1)}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            except (ValueError, OSError, TypeError, KeyError):
                self.respond(400, "Não foi possível salvar. Confira a classificação, o grupo e os limites de texto; se outra janela editou o arquivo, recarregue a página.")

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    return server


def serve(path, *, port=0, open_browser=True):
    with make_server(path, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Revisão local: {url}", flush=True)
        print("Progresso salvo por mensagem. Ctrl+C encerra o servidor.", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
