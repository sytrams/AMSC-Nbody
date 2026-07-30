#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from subprocess import PIPE, Popen, run
from typing import Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pyotp
import requests

import config


URL_RE = re.compile(r"https?://[^\s\"'>]+")
CALLBACK_RE = re.compile(r"globalprotectcallback:[^\s\"'<>]+")
CODE_LABEL_RE = re.compile(r"(?:code|codice)[^A-Za-z0-9]{0,20}([A-Z0-9-]{4,})", re.IGNORECASE)
CODE_TOKEN_RE = re.compile(r"\b[A-Z0-9]{4,}(?:-[A-Z0-9]{2,})*\b")
OTP_HINT_RE = re.compile(
    r"\b(?:otp|totp|mfa|2fa|two[\s-]?factor|one[\s-]?time|verification code|passcode|token|codice)\b",
    re.IGNORECASE,
)
OTP_FIELD_TOKENS = ("otp", "totp", "code", "token", "passcode", "pin", "verify", "verification")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass
class HtmlForm:
    action: str
    method: str = "post"
    attrs: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)


class FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[HtmlForm] = []
        self._current_form: HtmlForm | None = None
        self.text_parts: list[str] = []
        self.code_like_values: list[str] = []
        self.meta_refresh_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}

        if tag == "form":
            self._current_form = HtmlForm(
                action=attrs_dict.get("action", ""),
                method=attrs_dict.get("method", "post").lower(),
                attrs=attrs_dict,
            )
            self.forms.append(self._current_form)
            return

        if tag == "input" and self._current_form is not None:
            name = attrs_dict.get("name")
            if name:
                self._current_form.inputs[name] = attrs_dict.get("value", "")

        if tag == "meta":
            http_equiv = attrs_dict.get("http-equiv", "").lower()
            content = attrs_dict.get("content", "")
            if http_equiv == "refresh":
                match = re.search(r"url=(.+)$", content, re.IGNORECASE)
                if match:
                    self.meta_refresh_url = match.group(1).strip(" '\"")

        if tag in {"code", "pre", "strong", "b", "span", "div"}:
            value = attrs_dict.get("value")
            if value:
                self.code_like_values.append(value.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._current_form = None

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.text_parts.append(text)
            if len(text) <= 128:
                self.code_like_values.append(text)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


DEBUG_DIR = Path(os.environ.get("VPN_DEBUG_DIR", "/tmp/vpn-debug"))
DEBUG_ENABLED = os.environ.get("VPN_DEBUG", "1") != "0"
DEBUG_COUNTER = 0


def debug_path(name: str) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_DIR / name


def extract_title(html: str) -> str | None:
    match = TITLE_RE.search(html)
    if not match:
        return None
    return " ".join(match.group(1).split())


def summarize_forms(forms: Iterable[HtmlForm]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for index, form in enumerate(forms, start=1):
        summary.append(
            {
                "index": index,
                "id": form.attrs.get("id", ""),
                "class": form.attrs.get("class", ""),
                "method": form.method,
                "action": form.action,
                "inputs": sorted(form.inputs.keys()),
            }
        )
    return summary


def debug_dump_response(label: str, response: requests.Response) -> None:
    global DEBUG_COUNTER
    parser = parse_forms(response.text)
    title = extract_title(response.text)
    merged_text = " ".join(parser.text_parts)
    text_snippet = merged_text[:400]
    forms_summary = summarize_forms(parser.forms)

    log(
        f"[debug] {label}: status={response.status_code} url={response.url} "
        f"title={title!r} forms={len(forms_summary)} "
        f"login_form={'yes' if pick_login_form(parser.forms) else 'no'} "
        f"otp_form={'yes' if pick_otp_form(parser.forms) else 'no'} "
        f"auto_form={'yes' if pick_auto_submit_form(parser.forms) else 'no'} "
        f"meta_refresh={parser.meta_refresh_url or '-'}"
    )
    if text_snippet:
        log(f"[debug] {label}: text-snippet={text_snippet!r}")
    for form_info in forms_summary:
        log(
            "[debug] "
            f"{label}: form#{form_info['index']} id={form_info['id']!r} "
            f"class={form_info['class']!r} method={form_info['method']!r} "
            f"action={form_info['action']!r} inputs={form_info['inputs']!r}"
        )

    if not DEBUG_ENABLED:
        return

    DEBUG_COUNTER += 1
    stem = f"{DEBUG_COUNTER:02d}_{label}"
    html_path = debug_path(f"{stem}.html")
    json_path = debug_path(f"{stem}.json")
    html_path.write_text(response.text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "label": label,
                "status_code": response.status_code,
                "url": response.url,
                "title": title,
                "meta_refresh_url": parser.meta_refresh_url,
                "forms": forms_summary,
                "text_snippet": text_snippet,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"[debug] {label}: wrote {html_path} and {json_path}")


def parse_forms(html: str) -> FormParser:
    parser = FormParser()
    parser.feed(html)
    return parser


def _field_names(inputs: dict[str, str]) -> set[str]:
    return {name.lower() for name in inputs}


def pick_login_form(forms: Iterable[HtmlForm]) -> HtmlForm | None:
    ranked: list[tuple[int, HtmlForm]] = []
    for form in forms:
        names = _field_names(form.inputs)
        score = 0
        if any("pass" in name for name in names):
            score += 5
        if any("user" in name or "email" in name for name in names):
            score += 5
        if form.attrs.get("id") == "kc-form-login":
            score += 3
        action = form.action.lower()
        if "login" in action or "authenticate" in action:
            score += 2
        if score:
            ranked.append((score, form))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def pick_otp_form(forms: Iterable[HtmlForm]) -> HtmlForm | None:
    ranked: list[tuple[int, HtmlForm]] = []
    for form in forms:
        names = _field_names(form.inputs)
        score = 0
        form_id = form.attrs.get("id", "").lower()
        form_class = form.attrs.get("class", "").lower()
        form_action = form.action.lower()

        if form_id == "form_scelta2fa":
            score += 12
        if "identificazionedfa.do" in form_action:
            score += 8
        if any(any(token in name for token in OTP_FIELD_TOKENS) for name in names):
            score += 6
        if any(token in form_id for token in ("otp", "mfa", "token", "verify", "twofactor")):
            score += 4
        if any(token in form_class for token in ("otp", "mfa", "token", "verify", "twofactor")):
            score += 3
        if any(token in form_action for token in ("otp", "mfa", "token", "verify", "twofactor")):
            score += 2
        if "evn_continua" in names:
            score += 2
        if len(names) == 1 and not any("pass" in name or "user" in name for name in names):
            score += 1
        if score:
            ranked.append((score, form))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def pick_auto_submit_form(forms: Iterable[HtmlForm]) -> HtmlForm | None:
    ranked: list[tuple[int, HtmlForm]] = []
    for form in forms:
        names = _field_names(form.inputs)
        if any(token in name for name in names for token in ("pass", "otp", "totp")):
            continue

        score = 0
        if form.attrs.get("id") == "myform":
            score += 10
        if form.attrs.get("name") == "hiddenform":
            score += 5
        if names and all(
            token not in name
            for name in names
            for token in ("user", "email", "login", "pass", "otp", "totp")
        ):
            score += 2
        if len(names) >= 1:
            score += 1

        action = form.action.lower()
        if any(token in action for token in ("saml", "auth", "login", "resume")):
            score += 2

        if score:
            ranked.append((score, form))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def fill_login_payload(form: HtmlForm, username: str, password: str) -> dict[str, str]:
    payload = dict(form.inputs)
    username_field = next(
        (name for name in payload if any(token in name.lower() for token in ("user", "email", "login"))),
        None,
    )
    password_field = next((name for name in payload if "pass" in name.lower()), None)

    if username_field is None or password_field is None:
        raise RuntimeError("Could not identify username/password fields in the login form.")

    payload[username_field] = username
    payload[password_field] = password

    if "credentialId" in payload:
        payload["credentialId"] = ""
    if "rememberMe" in payload and not payload["rememberMe"]:
        payload["rememberMe"] = "on"

    return payload


def fill_otp_payload(form: HtmlForm, otp_code: str) -> dict[str, str]:
    payload = dict(form.inputs)
    otp_field = next(
        (name for name in payload if any(token in name.lower() for token in OTP_FIELD_TOKENS)),
        None,
    )
    if otp_field is None:
        raise RuntimeError("Could not identify OTP field in the MFA form.")

    payload[otp_field] = otp_code
    if "evn_continua" in payload and not payload["evn_continua"]:
        payload["evn_continua"] = "evento"
    if "login" in payload and not payload["login"]:
        payload["login"] = "Log In"
    return payload


def extract_auth_data(response: requests.Response) -> str:
    direct_match = CALLBACK_RE.search(response.url)
    if direct_match:
        return direct_match.group(0)

    body_match = CALLBACK_RE.search(response.text)
    if body_match:
        return body_match.group(0)

    decoded_url = unquote(response.url)
    decoded_url_match = CALLBACK_RE.search(decoded_url)
    if decoded_url_match:
        return decoded_url_match.group(0)

    decoded_body = unquote(response.text)
    decoded_body_match = CALLBACK_RE.search(decoded_body)
    if decoded_body_match:
        return decoded_body_match.group(0)

    parsed_url = urlparse(response.url)
    query_code = parse_qs(parsed_url.query).get("code")
    if query_code:
        return query_code[0]

    parser = parse_forms(response.text)

    for form in parser.forms:
        for name, value in form.inputs.items():
            if "code" in name.lower() and value:
                return value.strip()

    for candidate in parser.code_like_values:
        text = candidate.strip()
        labelled = CODE_LABEL_RE.search(text)
        if labelled:
            return labelled.group(1)

    merged_text = "\n".join(parser.text_parts)
    labelled = CODE_LABEL_RE.search(merged_text)
    if labelled:
        return labelled.group(1)

    for candidate in CODE_TOKEN_RE.findall(merged_text):
        if any(ch.isdigit() for ch in candidate) or "-" in candidate:
            return candidate

    raise RuntimeError("Could not extract the final authentication payload from the response page.")


def submit_form(session: requests.Session, base_url: str, form: HtmlForm, payload: dict[str, str]) -> requests.Response:
    action = urljoin(base_url, form.action or "")
    redacted_payload = {
        key: ("<redacted>" if any(token in key.lower() for token in ("pass", "otp", "token")) else value)
        for key, value in payload.items()
    }
    log(
        f"[debug] submitting form method={form.method} action={action} "
        f"id={form.attrs.get('id', '')!r} payload_keys={sorted(payload.keys())!r} "
        f"payload_preview={redacted_payload!r}"
    )
    if form.method == "get":
        response = session.get(action, params=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
    else:
        response = session.post(action, data=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def describe_response(response: requests.Response) -> str:
    parser = parse_forms(response.text)
    return (
        f"url={response.url} forms={len(parser.forms)} "
        f"login_form={'yes' if pick_login_form(parser.forms) else 'no'} "
        f"otp_form={'yes' if pick_otp_form(parser.forms) else 'no'} "
        f"auto_form={'yes' if pick_auto_submit_form(parser.forms) else 'no'} "
        f"meta_refresh={'yes' if parser.meta_refresh_url else 'no'} "
        f"otp_hint={'yes' if OTP_HINT_RE.search(' '.join(parser.text_parts)) else 'no'}"
    )


def advance_to_interactive_page(session: requests.Session, response: requests.Response) -> requests.Response:
    for step in range(1, 8):
        debug_dump_response(f"interactive_step_{step}", response)
        parser = parse_forms(response.text)

        if pick_login_form(parser.forms) or pick_otp_form(parser.forms):
            if step > 1:
                log(f"Reached interactive page after {step - 1} redirect step(s): {response.url}")
            return response

        if parser.meta_refresh_url:
            next_url = urljoin(response.url, parser.meta_refresh_url)
            log(f"Following meta refresh to {next_url}")
            response = session.get(next_url, timeout=config.REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            continue

        auto_form = pick_auto_submit_form(parser.forms)
        if auto_form is not None:
            log(f"Submitting intermediate redirect form at {response.url}")
            response = submit_form(session, response.url, auto_form, dict(auto_form.inputs))
            continue

        return response

    raise RuntimeError("Too many intermediate redirect steps while waiting for the PoliMi login page.")


def page_has_callback(response: requests.Response) -> bool:
    return CALLBACK_RE.search(response.url) is not None or CALLBACK_RE.search(response.text) is not None


def page_looks_like_otp(response: requests.Response) -> bool:
    parser = parse_forms(response.text)
    if pick_otp_form(parser.forms):
        return True
    merged_text = "\n".join(parser.text_parts)
    return OTP_HINT_RE.search(merged_text) is not None


def authenticate_link(link: str) -> str:
    username = config.require(config.POLIMI_USERNAME, "POLIMI_USERNAME", "CINECA_USERNAME")
    password = config.require(config.POLIMI_PASSWORD, "POLIMI_PASSWORD", "CINECA_PASSWORD")
    otp_secret = config.require(config.POLIMI_OTP_SECRET, "POLIMI_OTP_SECRET", "CINECA_OTP_SECRET")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    log(f"Loading VPN login page from {link}")
    response = session.get(link, timeout=config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    debug_dump_response("initial_fetch", response)
    response = advance_to_interactive_page(session, response)

    parser = parse_forms(response.text)
    login_form = pick_login_form(parser.forms)
    if login_form is None:
        raise RuntimeError(
            "Could not find the PoliMi credential form in the remote browser page. "
            f"Page summary: {describe_response(response)}"
        )

    log("Submitting PoliMi credentials")
    login_payload = fill_login_payload(login_form, username, password)
    response = submit_form(session, response.url, login_form, login_payload)
    debug_dump_response("after_login_submit_raw", response)
    response = advance_to_interactive_page(session, response)
    log(f"Post-login page summary: {describe_response(response)}")

    post_login_parser = parse_forms(response.text)
    if pick_login_form(post_login_parser.forms) and not pick_otp_form(post_login_parser.forms) and not page_has_callback(response):
        raise RuntimeError(
            "The credential submission returned the Polimi login page again instead of the 2FA page. "
            "This usually means the login identifier is wrong for the web form. "
            "Polimi web login expects the person code (for example 10774182), while cluster SSH usually uses "
            "u<person_code> (for example u10774182). "
            f"Page summary: {describe_response(response)}"
        )

    for attempt in range(3):
        if page_has_callback(response):
            break

        otp_parser = parse_forms(response.text)
        otp_form = pick_otp_form(otp_parser.forms)
        if otp_form is None:
            if page_looks_like_otp(response):
                raise RuntimeError(
                    "The page appears to require OTP/MFA, but the script could not identify the OTP form. "
                    f"Page summary: {describe_response(response)}"
                )
            break
        otp_code = pyotp.TOTP(otp_secret).now()
        log(f"Submitting OTP challenge #{attempt + 1}")
        otp_payload = fill_otp_payload(otp_form, otp_code)
        response = submit_form(session, response.url, otp_form, otp_payload)
        debug_dump_response(f"after_otp_submit_raw_{attempt + 1}", response)
        response = advance_to_interactive_page(session, response)
        log(f"Post-OTP page summary #{attempt + 1}: {describe_response(response)}")

    auth_data = extract_auth_data(response)
    if auth_data.startswith("globalprotectcallback:"):
        log("Extracted GlobalProtect callback payload from VPN page")
    else:
        log("Extracted fallback approval code from VPN page")
    return auth_data


def build_gpauth_command(portal: str, fix_openssl: bool, gpauth_bin: str) -> list[str]:
    command = [gpauth_bin]
    if fix_openssl:
        command.append("--fix-openssl")
    command.extend([portal, "--browser", "remote"])
    return command


def get_vpn_cookie(portal: str, fix_openssl: bool, gpauth_bin: str) -> str:
    command = build_gpauth_command(portal, fix_openssl, gpauth_bin)
    log(f"Launching {' '.join(command)}")
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    link: str | None = None
    stderr_lines: list[str] = []
    while True:
        line = process.stderr.readline()
        if not line:
            break
        stderr_lines.append(line)
        sys.stderr.write(line)
        sys.stderr.flush()
        if link is None:
            match = URL_RE.search(line)
            if match:
                link = match.group(0)
                break

    if link is None:
        process.wait(timeout=5)
        raise RuntimeError("gpauth did not print a remote browser URL.")

    auth_data = authenticate_link(link)
    if not auth_data.startswith("globalprotectcallback:"):
        raise RuntimeError(
            "Authentication reached the final page, but the response did not contain a "
            f"globalprotectcallback payload. Got: {auth_data[:80]}"
        )
    process.stdin.write(auth_data + "\n")
    process.stdin.flush()
    process.stdin.close()

    cookie = process.stdout.read().strip()
    remaining_stderr = process.stderr.read()
    if remaining_stderr:
        sys.stderr.write(remaining_stderr)
        sys.stderr.flush()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"gpauth exited with status {return_code}.")
    if not cookie:
        raise RuntimeError("gpauth did not emit an auth cookie on stdout.")
    return cookie


def connect_with_cookie(portal: str, cookie: str, use_sudo: bool, fix_openssl: bool, gpclient_bin: str) -> int:
    command: list[str] = []
    if use_sudo:
        command.extend(["sudo", "-E"])
    command.append(gpclient_bin)
    if fix_openssl:
        command.append("--fix-openssl")
    command.extend(["connect", portal, "--cookie-on-stdin"])
    log(f"Launching {' '.join(command)}")
    completed = run(command, input=cookie + "\n", text=True, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate PoliMi VPN login for GlobalProtect remote-browser authentication."
    )
    parser.add_argument("--portal", default=config.PORTAL, help="VPN portal hostname.")
    parser.add_argument("--link", default=config.LINK, help="Remote browser URL emitted by gpauth or gpclient.")
    parser.add_argument("--gpauth-bin", default=config.GPAUTH_BIN, help="Path to the gpauth binary.")
    parser.add_argument("--gpclient-bin", default=config.GPCLIENT_BIN, help="Path to the gpclient binary.")
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Run gpclient without sudo even if GPCLIENT_USE_SUDO is enabled.",
    )
    parser.add_argument(
        "--fix-openssl",
        action="store_true",
        default=(config.GPAUTH_FIX_OPENSSL or config.GPCLIENT_FIX_OPENSSL),
        help="Pass --fix-openssl to gpauth and gpclient.",
    )
    parser.add_argument(
        "--print-cookie-only",
        action="store_true",
        help="Authenticate with gpauth and print only the resulting VPN cookie to stdout.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.link:
        print(authenticate_link(args.link))
        return 0

    if not args.portal:
        raise RuntimeError("Set PORTAL or pass --portal when no LINK is provided.")

    cookie = get_vpn_cookie(args.portal, args.fix_openssl, args.gpauth_bin)
    if args.print_cookie_only:
        print(cookie)
        return 0

    return connect_with_cookie(
        portal=args.portal,
        cookie=cookie,
        use_sudo=(config.GPCLIENT_USE_SUDO and not args.no_sudo),
        fix_openssl=args.fix_openssl,
        gpclient_bin=args.gpclient_bin,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1)
