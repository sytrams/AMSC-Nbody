#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from subprocess import PIPE, STDOUT, Popen
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import pyotp
import requests

import config


URL_RE = re.compile(r"https?://[^\s\"'>]+")
CODE_LABEL_RE = re.compile(r"(?:code|codice)[^A-Za-z0-9]{0,20}([A-Z0-9-]{4,})", re.IGNORECASE)
CODE_TOKEN_RE = re.compile(r"\b[A-Z0-9]{4,}(?:-[A-Z0-9]{2,})*\b")


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
        if any("user" in name for name in names):
            score += 5
        if "kc-form-login" == form.attrs.get("id"):
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
        if any("otp" in name or "totp" in name for name in names):
            score += 6
        if any("code" in name for name in names):
            score += 3
        if "otp" in form.attrs.get("id", "").lower():
            score += 4
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
        (name for name in payload if any(token in name.lower() for token in ("otp", "totp", "code"))),
        None,
    )
    if otp_field is None:
        raise RuntimeError("Could not identify OTP field in the MFA form.")

    payload[otp_field] = otp_code
    if "login" in payload and not payload["login"]:
        payload["login"] = "Log In"
    return payload


def extract_code(response: requests.Response) -> str:
    parsed_url = urlparse(response.url)
    query_code = parse_qs(parsed_url.query).get("code")
    if query_code:
        return query_code[0]

    parser = parse_forms(response.text)

    for form in parser.forms:
        for name, value in form.inputs.items():
            lowered = name.lower()
            if "code" in lowered and value:
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

    raise RuntimeError("Could not extract the final approval code from the response page.")


def submit_form(session: requests.Session, base_url: str, form: HtmlForm, payload: dict[str, str]) -> requests.Response:
    action = urljoin(base_url, form.action or "")
    method = form.method or "post"
    if method == "get":
        response = session.get(action, params=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
    else:
        response = session.post(action, data=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def authenticate_link(link: str) -> str:
    username = config.require(config.POLIMI_USERNAME, "POLIMI_USERNAME", "CINECA_USERNAME")
    password = config.require(config.POLIMI_PASSWORD, "POLIMI_PASSWORD", "CINECA_PASSWORD")
    otp_secret = config.require(config.POLIMI_OTP_SECRET, "POLIMI_OTP_SECRET", "CINECA_OTP_SECRET")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    log(f"Loading VPN login page from {link}")
    response = session.get(link, timeout=config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    parser = parse_forms(response.text)
    login_form = pick_login_form(parser.forms)
    if login_form is None:
        raise RuntimeError("Could not find the PoliMi credential form in the remote browser page.")

    log("Submitting PoliMi credentials")
    login_payload = fill_login_payload(login_form, username, password)
    response = submit_form(session, response.url, login_form, login_payload)

    otp_parser = parse_forms(response.text)
    otp_form = pick_otp_form(otp_parser.forms)
    if otp_form is not None:
        otp_code = pyotp.TOTP(otp_secret).now()
        log("Submitting OTP challenge")
        otp_payload = fill_otp_payload(otp_form, otp_code)
        response = submit_form(session, response.url, otp_form, otp_payload)

    code = extract_code(response)
    log("Extracted approval code from VPN page")
    return code


def stream_gpclient(portal: str, use_sudo: bool, fix_openssl: bool, gpclient_bin: str) -> int:
    command: list[str] = []
    if use_sudo:
        command.extend(["sudo", "-E"])
    command.append(gpclient_bin)
    if fix_openssl:
        command.append("--fix-openssl")
    command.extend(["connect", "--browser", "remote", portal])

    log(f"Launching {' '.join(command)}")
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    assert process.stdin is not None

    link: str | None = None
    for line in process.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()
        if link is None:
            match = URL_RE.search(line)
            if match:
                link = match.group(0)
                break

    if link is None:
        process.wait(timeout=5)
        raise RuntimeError("gpclient did not print a remote browser URL.")

    code = authenticate_link(link)
    process.stdin.write(code + "\n")
    process.stdin.flush()
    process.stdin.close()

    for line in process.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()

    return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate PoliMi VPN login for gpclient remote-browser authentication."
    )
    parser.add_argument("--portal", default=config.PORTAL, help="VPN portal hostname.")
    parser.add_argument("--link", default=config.LINK, help="Remote browser URL emitted by gpclient.")
    parser.add_argument("--gpclient-bin", default=config.GPCLIENT_BIN, help="Path to the gpclient binary.")
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Run gpclient without sudo even if GPCLIENT_USE_SUDO is enabled.",
    )
    parser.add_argument(
        "--fix-openssl",
        action="store_true",
        default=config.GPCLIENT_FIX_OPENSSL,
        help="Pass --fix-openssl to gpclient.",
    )
    parser.add_argument(
        "--print-code-only",
        action="store_true",
        help="Only print the final approval code to stdout. Requires --link or LINK.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.link:
        code = authenticate_link(args.link)
        print(code)
        return 0

    if not args.portal:
        raise RuntimeError("Set PORTAL or pass --portal when no LINK is provided.")

    return stream_gpclient(
        portal=args.portal,
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
