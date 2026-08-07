from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape, unescape
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import signal
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from typing import TextIO
from urllib.parse import urlsplit

import pyotp
from playwright.sync_api import BrowserContext, Locator, Page, sync_playwright


CALLBACK_PREFIX = "globalprotectcallback:"
CALLBACK_PATTERN = re.compile(r"globalprotectcallback:[^\s\"'<>]+", re.IGNORECASE)
HTTP_URL_PATTERN = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
AUTH_TIMEOUT_SECONDS = 60
AUTH_URL_TIMEOUT_SECONDS = 30
VPN_READY_TIMEOUT_SECONDS = 180
PROCESS_STOP_TIMEOUT_SECONDS = 15
POLL_INTERVAL_MS = 250
PROCESS_STARTED_AT = monotonic()

VPN_READY_PATTERN = re.compile(
    r"Connected to VPN|Wrote PID .*gpclient\.lock|Configured as .*tunnel",
    re.IGNORECASE,
)

LOGIN_BUTTON_PATTERN = re.compile(r"log in|sign in|Accedi", re.IGNORECASE)
OTP_BUTTON_PATTERN = re.compile(
    r"evn_continua|continua|continue|verify|submit",
    re.IGNORECASE,
)
CONFIRM_BUTTON_PATTERN = re.compile(
    r"continue|continua|approve|authorize|allow|consenti",
    re.IGNORECASE,
)
TERMINAL_ERROR_PATTERN = re.compile(
    r"invalid|incorrect|failed|expired|already\s+(?:been\s+)?used|"
    r"non\s+valid|errat|scadut|errore|riprova|try\s+again",
    re.IGNORECASE,
)

USERNAME_SELECTOR = (
    'input[autocomplete="username"], '
    'input[name*="login" i], '
    'input[type="email"]'
)
PASSWORD_SELECTOR = 'input[type="password"]'
OTP_SELECTOR = (
    'input[name*="otp" i], '
    'input[name*="totp" i], '
    'input[autocomplete="one-time-code"]'
)
ERROR_SELECTOR = (
    '[role="alert"], '
    '[aria-live="assertive"], '
    '.alert-danger, '
    '.error, '
    '[class*="error" i], '
    '[id*="error" i]'
)


class ShutdownRequested(Exception):
    """Raised when the workflow asks the background VPN process to stop."""


@dataclass
class VpnDiagnostics:
    directory: Path
    stage: str = "startup"
    result: str = "starting"
    page_title: str | None = None
    credential_form_visible: bool = False
    otp_form_visible: bool = False
    callback_found: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0

    def payload(self) -> dict[str, str | bool | float | None]:
        return {
            "stage": self.stage,
            "result": self.result,
            "page_title": self.page_title,
            "credential_form_visible": self.credential_form_visible,
            "otp_form_visible": self.otp_form_visible,
            "callback_found": self.callback_found,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


def log_event(stage: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    elapsed = monotonic() - PROCESS_STARTED_AT
    print(
        f"[vpn-auth {timestamp} +{elapsed:07.2f}s] {stage}: {message}",
        file=sys.stderr,
        flush=True,
    )


def required_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required secret {name} is missing")
    return value


def required_status_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required status-file environment variable {name} is missing"
        )
    path = Path(value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def required_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required directory environment variable {name} is missing"
        )
    path = Path(value).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_secrets() -> tuple[str, ...]:
    return tuple(
        os.environ.get(name, "")
        for name in (
            "POLIMI_USERNAME",
            "POLIMI_PASSWORD",
            "POLIMI_TOTP",
            "POLIMI_VPN",
        )
    )


def remove_status_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def write_status_file(path: Path, message: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(message.rstrip() + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def install_shutdown_handlers(shutdown_event: Event) -> None:
    def request_shutdown(signum: int, _frame) -> None:
        signal_name = signal.Signals(signum).name
        log_event("lifecycle", f"received {signal_name}; shutdown requested")
        shutdown_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def normalize_callback(value: str | None) -> str | None:
    if not value:
        return None

    decoded = unescape(value.strip())
    start = decoded.lower().find(CALLBACK_PREFIX)
    if start == -1:
        return None

    callback = decoded[start:].rstrip(".,);]")
    if not callback.lower().startswith(CALLBACK_PREFIX):
        return None
    return callback


def active_pages(context: BrowserContext) -> list[Page]:
    return [page for page in reversed(context.pages) if not page.is_closed()]


def get_callback(context: BrowserContext) -> str | None:
    for page in active_pages(context):
        for frame in page.frames:
            callback = normalize_callback(frame.url)
            if callback:
                return callback

            for selector, attribute in (
                ('a[href^="globalprotectcallback:" i]', "href"),
                ('[value^="globalprotectcallback:" i]', "value"),
            ):
                try:
                    elements = frame.locator(selector)
                    for index in range(elements.count()):
                        callback = normalize_callback(
                            elements.nth(index).get_attribute(attribute)
                        )
                        if callback:
                            return callback
                except Exception:
                    # A redirect may detach the frame while it is inspected.
                    continue

            try:
                match = CALLBACK_PATTERN.search(frame.content())
            except Exception:
                # Navigations can make a frame temporarily unavailable.
                continue
            if match:
                return normalize_callback(match.group(0))

    return None


def first_visible(locator: Locator) -> Locator | None:
    try:
        for index in range(locator.count()):
            candidate = locator.nth(index)
            if candidate.is_visible():
                return candidate
    except Exception:
        pass
    return None


def find_login_controls(
    context: BrowserContext,
) -> tuple[Page, Locator, Locator] | None:
    for page in active_pages(context):
        username = first_visible(page.locator(USERNAME_SELECTOR))
        password = first_visible(page.locator(PASSWORD_SELECTOR))
        if username is not None and password is not None:
            return page, username, password
    return None


def find_otp_control(context: BrowserContext) -> tuple[Page, Locator] | None:
    for page in active_pages(context):
        otp = first_visible(page.locator(OTP_SELECTOR))
        if otp is not None:
            return page, otp
    return None


def find_confirmation_control(context: BrowserContext) -> tuple[Page, Locator] | None:
    for page in active_pages(context):
        button = first_visible(
            page.get_by_role("button", name=CONFIRM_BUTTON_PATTERN)
        )
        if button is not None:
            return page, button
    return None


def insert_credentials(
    page: Page,
    username_field: Locator,
    password_field: Locator,
    username: str,
    password: str,
) -> None:
    username_field.fill(username)
    password_field.fill(password)

    button = first_visible(page.get_by_role("button", name=LOGIN_BUTTON_PATTERN))
    if button is None:
        raise RuntimeError("Could not find the credential submission button.")
    button.click()


def next_unused_otp(
    page: Page,
    previous_otp: str | None,
    deadline: float,
    stage: str,
) -> str:
    totp = pyotp.TOTP(required_secret("POLIMI_TOTP"))
    announced_wait = False

    while True:
        otp_code = totp.now()
        if otp_code != previous_otp:
            return otp_code

        if monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for a fresh OTP value.")

        if not announced_wait:
            log_event(
                stage,
                "another OTP was requested in the same TOTP window; "
                "waiting for a fresh value",
            )
            announced_wait = True
        page.wait_for_timeout(500)


def insert_otp(
    page: Page,
    otp_field: Locator,
    previous_otp: str | None,
    deadline: float,
    stage: str,
) -> str:
    otp_code = next_unused_otp(page, previous_otp, deadline, stage)
    otp_field.fill(otp_code)

    button = first_visible(page.get_by_role("button", name=OTP_BUTTON_PATTERN))
    if button is None:
        raise RuntimeError("Could not find the OTP submission button.")
    button.click()
    return otp_code


def redact_sensitive_values(text: str, secrets: tuple[str, ...]) -> str:
    sanitized = CALLBACK_PATTERN.sub("[callback redacted]", text)
    sanitized = UUID_PATTERN.sub("[identifier redacted]", sanitized)

    def redact_url(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0).rstrip(".,;"))
        hostname = parsed.hostname or "host"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            port = ""
        return f"{parsed.scheme}://{hostname}{port}/[url redacted]"

    sanitized = HTTP_URL_PATTERN.sub(redact_url, sanitized)
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[redacted]")
    return sanitized


def sanitize_text(text: str, secrets: tuple[str, ...]) -> str:
    sanitized = redact_sensitive_values(text, secrets)
    return " ".join(sanitized.split())[:500]


def visible_errors(context: BrowserContext, secrets: tuple[str, ...]) -> list[str]:
    messages: list[str] = []
    for page in active_pages(context):
        try:
            candidates = page.locator(ERROR_SELECTOR)
            for index in range(min(candidates.count(), 20)):
                candidate = candidates.nth(index)
                if not candidate.is_visible():
                    continue
                message = sanitize_text(candidate.inner_text(), secrets)
                if message and message not in messages:
                    messages.append(message)
        except Exception:
            continue
    return messages


def page_diagnostics(context: BrowserContext, secrets: tuple[str, ...]) -> str:
    summaries: list[str] = []
    for page in active_pages(context):
        try:
            parsed = urlsplit(page.url)
            if parsed.scheme.lower() == "globalprotectcallback":
                location = f"{CALLBACK_PREFIX}[redacted]"
            elif parsed.hostname:
                location = f"{parsed.scheme}://{parsed.hostname}/[path redacted]"
            else:
                location = parsed.scheme or "unknown"
            title = sanitize_text(page.title(), secrets)
            summaries.append(f"url={location!r}, title={title!r}")
        except Exception:
            continue

    errors = visible_errors(context, secrets)
    details = "; ".join(summaries) or "no active page"
    details += (
        f"; credential_form={find_login_controls(context) is not None}"
        f"; otp_form={find_otp_control(context) is not None}"
        f"; confirmation_button={find_confirmation_control(context) is not None}"
    )
    if errors:
        details += f"; visible_errors={errors!r}"
    return details


def prepare_diagnostics(diagnostics: VpnDiagnostics) -> None:
    for name in ("summary.md", "failure.json", "failure.png"):
        (diagnostics.directory / name).unlink(missing_ok=True)


def capture_page_state(
    diagnostics: VpnDiagnostics,
    context: BrowserContext,
) -> None:
    secrets = runtime_secrets()
    pages = active_pages(context)
    if pages:
        try:
            diagnostics.page_title = sanitize_text(pages[0].title(), secrets) or None
        except Exception:
            diagnostics.page_title = None

    diagnostics.credential_form_visible = find_login_controls(context) is not None
    diagnostics.otp_form_visible = find_otp_control(context) is not None
    errors = visible_errors(context, secrets)
    if errors and not diagnostics.error:
        diagnostics.error = errors[0]


def write_diagnostic_summary(diagnostics: VpnDiagnostics) -> None:
    lines = [
        f"- VPN result: `{diagnostics.result}`",
        f"- Stage: `{diagnostics.stage}`",
        f"- Elapsed time: `{diagnostics.elapsed_seconds:.2f} seconds`",
    ]
    if diagnostics.page_title:
        lines.append(f"- Page title: {diagnostics.page_title}")
    lines.extend(
        (
            "- Credential form detected: "
            + ("yes" if diagnostics.credential_form_visible else "no"),
            "- OTP form detected: "
            + ("yes" if diagnostics.otp_form_visible else "no"),
            "- Callback detected: "
            + ("yes" if diagnostics.callback_found else "no"),
        )
    )
    if diagnostics.error:
        lines.append(f"- Error: {diagnostics.error}")

    (diagnostics.directory / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_failure_png(
    diagnostics: VpnDiagnostics,
    context: BrowserContext,
) -> None:
    fields = [
        ("Result", diagnostics.result),
        ("Stage", diagnostics.stage),
        ("Page title", diagnostics.page_title or "Not available"),
        (
            "Credential form detected",
            "Yes" if diagnostics.credential_form_visible else "No",
        ),
        ("OTP form detected", "Yes" if diagnostics.otp_form_visible else "No"),
        ("Callback detected", "Yes" if diagnostics.callback_found else "No"),
        ("Error", diagnostics.error or "No browser error was visible"),
        ("Elapsed", f"{diagnostics.elapsed_seconds:.2f} seconds"),
    ]
    rows = "".join(
        "<tr><th>"
        + escape(label)
        + "</th><td>"
        + escape(str(value))
        + "</td></tr>"
        for label, value in fields
    )
    html = f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <style>
              body {{ background: #f6f8fa; color: #1f2328; font: 16px sans-serif; }}
              main {{ background: white; border: 1px solid #d0d7de; border-radius: 12px;
                      margin: 40px auto; max-width: 850px; padding: 32px; }}
              h1 {{ margin-top: 0; }}
              table {{ border-collapse: collapse; width: 100%; }}
              th, td {{ border-top: 1px solid #d8dee4; padding: 12px; text-align: left; }}
              th {{ width: 35%; }}
              .notice {{ color: #57606a; font-size: 14px; }}
            </style>
          </head>
          <body>
            <main>
              <h1>VPN authentication failure</h1>
              <p class="notice">Sanitized diagnostic card; no authentication page,
              callback, cookie, username, password, or OTP is included.</p>
              <table>{rows}</table>
            </main>
          </body>
        </html>
    """

    diagnostic_page = context.new_page()
    try:
        diagnostic_page.set_content(html, wait_until="domcontentloaded")
        diagnostic_page.screenshot(
            path=str(diagnostics.directory / "failure.png"),
            full_page=True,
        )
    finally:
        diagnostic_page.close()


def record_failure(
    diagnostics: VpnDiagnostics,
    exc: Exception,
    context: BrowserContext | None = None,
) -> None:
    diagnostics.result = "failed"
    diagnostics.elapsed_seconds = round(monotonic() - PROCESS_STARTED_AT, 2)
    diagnostics.error = sanitize_text(str(exc), runtime_secrets()) or type(exc).__name__
    if context is not None:
        capture_page_state(diagnostics, context)

    write_diagnostic_summary(diagnostics)
    (diagnostics.directory / "failure.json").write_text(
        json.dumps(diagnostics.payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if context is not None:
        try:
            write_failure_png(diagnostics, context)
        except Exception as screenshot_exc:
            log_event(
                "diagnostics",
                "could not create sanitized failure image: "
                f"{type(screenshot_exc).__name__}",
            )

    log_event(
        "diagnostics",
        f"wrote sanitized failure diagnostics for stage={diagnostics.stage}",
    )


def record_connection_success(diagnostics: VpnDiagnostics) -> None:
    diagnostics.stage = "vpn"
    diagnostics.result = "connected"
    diagnostics.elapsed_seconds = round(monotonic() - PROCESS_STARTED_AT, 2)
    diagnostics.error = None
    write_diagnostic_summary(diagnostics)
    log_event("diagnostics", "wrote sanitized VPN connection summary")


def browser_session(
    context: BrowserContext,
    url: str,
    stage: str,
    previous_otp: str | None,
) -> tuple[str, str | None]:
    username = required_secret("POLIMI_USERNAME")
    password = required_secret("POLIMI_PASSWORD")
    secrets = (username, password, previous_otp or "")
    deadline = monotonic() + AUTH_TIMEOUT_SECONDS
    credentials_submitted = False
    otp_submitted = False
    confirmation_submitted = False
    used_otp = previous_otp
    observed_callbacks: list[str] = []
    preserve_pages_for_diagnostics = False

    def observe_request(request) -> None:
        callback = normalize_callback(request.url)
        if callback:
            observed_callbacks.append(callback)

    context.on("request", observe_request)

    page = context.new_page()
    try:
        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=AUTH_TIMEOUT_SECONDS * 1_000,
            )
        except Exception as exc:
            callback = observed_callbacks[-1] if observed_callbacks else None
            callback = callback or get_callback(context)
            if callback:
                log_event(
                    stage,
                    f"browser authentication completed; callback_length={len(callback)}",
                )
                return callback, used_otp
            raise RuntimeError(
                f"Could not load the {stage} authentication page; "
                f"{page_diagnostics(context, secrets)}"
            ) from exc

        log_event(stage, f"browser page ready; {page_diagnostics(context, secrets)}")

        while monotonic() < deadline:
            callback = observed_callbacks[-1] if observed_callbacks else None
            callback = callback or get_callback(context)
            if callback:
                log_event(
                    stage,
                    f"browser authentication completed; callback_length={len(callback)}",
                )
                return callback, used_otp

            errors = visible_errors(context, secrets)
            terminal_error = next(
                (message for message in errors if TERMINAL_ERROR_PATTERN.search(message)),
                None,
            )
            if terminal_error:
                raise RuntimeError(
                    f"{stage.capitalize()} authentication was rejected: {terminal_error}"
                )

            login_controls = find_login_controls(context)
            if login_controls is not None and not credentials_submitted:
                login_page, username_field, password_field = login_controls
                log_event(stage, "submitting credentials")
                insert_credentials(
                    login_page,
                    username_field,
                    password_field,
                    username,
                    password,
                )
                credentials_submitted = True
                continue

            otp_control = find_otp_control(context)
            if otp_control is not None and not otp_submitted:
                otp_page, otp_field = otp_control
                log_event(stage, "submitting OTP")
                used_otp = insert_otp(
                    otp_page,
                    otp_field,
                    previous_otp,
                    deadline,
                    stage,
                )
                secrets = (username, password, used_otp)
                otp_submitted = True
                continue

            confirmation_control = find_confirmation_control(context)
            if confirmation_control is not None and not confirmation_submitted:
                _, confirmation_button = confirmation_control
                log_event(stage, "confirming browser authentication")
                confirmation_button.click()
                confirmation_submitted = True
                continue

            pages = active_pages(context)
            if not pages:
                raise RuntimeError(
                    f"All browser pages closed before {stage} authentication completed."
                )
            pages[0].wait_for_timeout(POLL_INTERVAL_MS)

        raise RuntimeError(
            f"Timed out after {AUTH_TIMEOUT_SECONDS}s waiting for {stage} "
            f"authentication; {page_diagnostics(context, secrets)}"
        )
    except Exception:
        preserve_pages_for_diagnostics = True
        raise
    finally:
        context.remove_listener("request", observe_request)
        if not preserve_pages_for_diagnostics:
            # Pages are disposable; the shared context retains the IdP cookies.
            for auth_page in active_pages(context):
                try:
                    auth_page.close()
                except Exception:
                    pass


def read_process_output(stream: TextIO, output_queue: Queue[str | None]) -> None:
    secrets = runtime_secrets()
    try:
        for line in stream:
            print(redact_sensitive_values(line, secrets), end="", flush=True)
            output_queue.put(line)
    finally:
        output_queue.put(None)


def recent_output_summary(
    recent_lines: deque[str],
    secrets: tuple[str, ...],
) -> str:
    if not recent_lines:
        return "no gpclient output captured"
    sanitized = [sanitize_text(line, secrets) for line in recent_lines]
    return " | ".join(line for line in sanitized if line)[-2_000:]


def stop_gpclient(
    gpclient: subprocess.Popen[str],
    output_thread: Thread,
) -> None:
    if gpclient.stdin is not None and not gpclient.stdin.closed:
        try:
            gpclient.stdin.close()
        except Exception:
            pass

    if gpclient.poll() is None:
        log_event("cleanup", "requesting GlobalProtect disconnect")
        try:
            result = subprocess.run(
                ["sudo", "-n", "gpclient", "disconnect"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=PROCESS_STOP_TIMEOUT_SECONDS,
                check=False,
            )
            log_event(
                "cleanup",
                f"disconnect command exited with status {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            log_event("cleanup", "disconnect command timed out")
        except Exception as exc:
            log_event(
                "cleanup",
                f"disconnect command failed: {type(exc).__name__}: {exc}",
            )

    try:
        return_code = gpclient.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        log_event("cleanup", f"gpclient exited with status {return_code}")
    except subprocess.TimeoutExpired:
        log_event("cleanup", f"gpclient PID {gpclient.pid} did not exit; sending TERM")
        try:
            subprocess.run(
                ["sudo", "-n", "kill", "-TERM", str(gpclient.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            try:
                gpclient.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_event(
                    "cleanup",
                    f"gpclient PID {gpclient.pid} ignored TERM; sending KILL",
                )
                subprocess.run(
                    ["sudo", "-n", "kill", "-KILL", str(gpclient.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                gpclient.wait(timeout=5)
        except Exception as exc:
            log_event(
                "cleanup",
                f"could not stop gpclient PID {gpclient.pid}: "
                f"{type(exc).__name__}: {exc}",
            )

    output_thread.join(timeout=5)
    if output_thread.is_alive():
        log_event("cleanup", "gpclient output reader is still shutting down")


def wait_for_vpn_ready(
    context: BrowserContext,
    gpclient: subprocess.Popen[str],
    output_queue: Queue[str | None],
    shutdown_event: Event,
    diagnostics: VpnDiagnostics,
) -> None:
    assert gpclient.stdin is not None

    url_pattern = re.compile(r"https?://[^\s)\]]+")
    waiting_for_auth_url = False
    auth_url_deadline: float | None = None
    vpn_deadline = monotonic() + VPN_READY_TIMEOUT_SECONDS
    auth_round = 0
    previous_otp: str | None = None
    recent_lines: deque[str] = deque(maxlen=12)
    secrets = runtime_secrets()
    diagnostics.stage = "vpn"
    diagnostics.result = "connecting"

    log_event(
        "gpclient",
        f"waiting for VPN readiness; timeout={VPN_READY_TIMEOUT_SECONDS}s",
    )

    while monotonic() < vpn_deadline:
        if shutdown_event.is_set():
            raise ShutdownRequested

        now = monotonic()
        if auth_url_deadline is not None and now >= auth_url_deadline:
            raise RuntimeError(
                f"gpclient announced manual authentication but did not provide a URL "
                f"within {AUTH_URL_TIMEOUT_SECONDS}s; "
                f"recent_output={recent_output_summary(recent_lines, secrets)}"
            )

        if gpclient.poll() is not None and output_queue.empty():
            raise RuntimeError(
                f"gpclient exited before VPN readiness with status {gpclient.returncode}; "
                f"recent_output={recent_output_summary(recent_lines, secrets)}"
            )

        try:
            line = output_queue.get(timeout=0.5)
        except Empty:
            continue

        if line is None:
            if gpclient.poll() is None:
                continue
            raise RuntimeError(
                f"gpclient output ended before VPN readiness with status "
                f"{gpclient.returncode}; "
                f"recent_output={recent_output_summary(recent_lines, secrets)}"
            )

        recent_lines.append(line.strip())

        if VPN_READY_PATTERN.search(line):
            log_event("gpclient", "VPN readiness marker detected")
            return

        if "Manual Authentication Required" in line:
            waiting_for_auth_url = True
            auth_url_deadline = monotonic() + AUTH_URL_TIMEOUT_SECONDS
            log_event("gpclient", "manual browser authentication requested")
            continue

        if not waiting_for_auth_url:
            continue

        match = url_pattern.search(line)
        if not match:
            continue

        url = match.group(0).rstrip(".,);]")
        auth_round += 1
        if auth_round == 1:
            stage = "portal"
        elif auth_round == 2:
            stage = "gateway"
        else:
            stage = f"authentication-round-{auth_round}"

        diagnostics.stage = stage
        diagnostics.result = "authenticating"
        diagnostics.page_title = None
        diagnostics.credential_form_visible = False
        diagnostics.otp_form_visible = False
        diagnostics.callback_found = False
        log_event(stage, "received one-use remote-browser URL")
        callback, previous_otp = browser_session(
            context,
            url,
            stage,
            previous_otp,
        )
        if not callback.lower().startswith(CALLBACK_PREFIX):
            raise RuntimeError(
                f"Expected {CALLBACK_PREFIX}, but browser authentication "
                "did not return a valid callback."
            )

        gpclient.stdin.write(callback.rstrip("\r\n") + "\n")
        gpclient.stdin.flush()
        diagnostics.callback_found = True
        log_event(
            stage,
            f"authentication succeeded; callback_detected=yes; "
            f"callback_length={len(callback)}",
        )
        waiting_for_auth_url = False
        auth_url_deadline = None

    raise RuntimeError(
        f"VPN readiness was not detected within {VPN_READY_TIMEOUT_SECONDS}s; "
        f"recent_output={recent_output_summary(recent_lines, secrets)}"
    )


def wait_for_shutdown(
    gpclient: subprocess.Popen[str],
    output_queue: Queue[str | None],
    shutdown_event: Event,
) -> None:
    recent_lines: deque[str] = deque(maxlen=12)
    runtime_secrets = tuple(
        os.environ.get(name, "")
        for name in (
            "POLIMI_USERNAME",
            "POLIMI_PASSWORD",
            "POLIMI_TOTP",
            "POLIMI_VPN",
        )
    )

    log_event("lifecycle", "VPN is ready and will remain connected until cancelled")
    while not shutdown_event.wait(0.5):
        while True:
            try:
                line = output_queue.get_nowait()
            except Empty:
                break
            if line is not None:
                recent_lines.append(line.strip())

        if gpclient.poll() is not None:
            raise RuntimeError(
                f"gpclient exited while the VPN was expected to remain connected "
                f"with status {gpclient.returncode}; "
                f"recent_output={recent_output_summary(recent_lines, runtime_secrets)}"
            )

    log_event("lifecycle", "workflow requested VPN shutdown")


def start_vpn(
    ready_file: Path,
    shutdown_event: Event,
    diagnostics: VpnDiagnostics,
) -> int:
    log_event("startup", "launching shared headless browser context")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        log_event("startup", "launching gpclient with remote-browser authentication")
        gpclient = subprocess.Popen(
            [
                "sudo",
                "gpclient",
                "connect",
                required_secret("POLIMI_VPN"),
                "--browser",
                "remote",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        assert gpclient.stdout is not None
        output_queue: Queue[str | None] = Queue()
        output_thread = Thread(
            target=read_process_output,
            args=(gpclient.stdout, output_queue),
            name="gpclient-output",
            daemon=True,
        )
        output_thread.start()

        try:
            wait_for_vpn_ready(
                context,
                gpclient,
                output_queue,
                shutdown_event,
                diagnostics,
            )
            record_connection_success(diagnostics)
            write_status_file(ready_file, "ready")
            log_event("status", f"published VPN readiness in {ready_file.name}")
            wait_for_shutdown(gpclient, output_queue, shutdown_event)
            return 0
        except ShutdownRequested:
            raise
        except Exception as exc:
            if diagnostics.result != "failed":
                record_failure(diagnostics, exc, context)
            raise
        finally:
            try:
                stop_gpclient(gpclient, output_thread)
            finally:
                context.close()
                browser.close()
                log_event("cleanup", "browser and VPN resources released")


def main() -> int:
    diagnostics = VpnDiagnostics(required_directory("VPN_DIAGNOSTICS_DIR"))
    prepare_diagnostics(diagnostics)
    ready_file = required_status_path("VPN_READY_FILE")
    failed_file = required_status_path("VPN_FAILED_FILE")
    remove_status_file(ready_file)
    remove_status_file(failed_file)

    shutdown_event = Event()
    install_shutdown_handlers(shutdown_event)

    try:
        return start_vpn(ready_file, shutdown_event, diagnostics)
    except ShutdownRequested:
        log_event("lifecycle", "shutdown completed before VPN readiness")
        return 0
    except Exception as exc:
        remove_status_file(ready_file)
        if diagnostics.result != "failed":
            record_failure(diagnostics, exc)
        failure = f"{type(exc).__name__}: {sanitize_text(str(exc), runtime_secrets())}"
        write_status_file(failed_file, failure)
        log_event("status", f"published VPN failure in {failed_file.name}: {failure}")
        raise
    finally:
        remove_status_file(ready_file)


if __name__ == "__main__":
    raise SystemExit(main())
