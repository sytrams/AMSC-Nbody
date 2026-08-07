from collections import deque
from datetime import datetime, timezone
from html import unescape
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
from threading import Thread
from time import monotonic
from typing import TextIO
from urllib.parse import urlsplit

import pyotp
from playwright.sync_api import BrowserContext, Locator, Page, sync_playwright


CALLBACK_PREFIX = "globalprotectcallback:"
CALLBACK_PATTERN = re.compile(r"globalprotectcallback:[^\s\"'<>]+", re.IGNORECASE)
AUTH_TIMEOUT_SECONDS = 60
AUTH_URL_TIMEOUT_SECONDS = 30
VPN_READY_TIMEOUT_SECONDS = 180
SSH_TIMEOUT_SECONDS = 30
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


def sanitize_text(text: str, secrets: tuple[str, ...]) -> str:
    sanitized = CALLBACK_PATTERN.sub("[callback redacted]", text)
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[redacted]")
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
                location = f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
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
    finally:
        context.remove_listener("request", observe_request)
        # Pages are disposable; the shared context retains the IdP cookies.
        for auth_page in active_pages(context):
            try:
                auth_page.close()
            except Exception:
                pass


def read_process_output(stream: TextIO, output_queue: Queue[str | None]) -> None:
    try:
        for line in stream:
            print(line, end="", flush=True)
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


def run_ssh_probe() -> None:
    username = required_secret("SSH_USERNAME")
    host = required_secret("CLUSTER_IP_ADDRESS")
    identity_file = Path(
        os.environ.get("SSH_IDENTITY_FILE", str(Path.home() / ".ssh" / "cineca"))
    ).expanduser()
    if not identity_file.is_file():
        raise RuntimeError(f"SSH identity file does not exist: {identity_file}")

    command = [
        "ssh",
        "-vv",
        "-T",
        "-i",
        str(identity_file),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=1",
        f"{username}@{host}",
        "true",
    ]

    log_event(
        "ssh",
        f"starting noninteractive connectivity probe; timeout={SSH_TIMEOUT_SECONDS}s",
    )
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=SSH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        sanitized = sanitize_text(output, (username, host))
        if sanitized:
            log_event("ssh-output", sanitized)
        raise RuntimeError(
            f"SSH connectivity probe timed out after {SSH_TIMEOUT_SECONDS}s"
        ) from exc

    sanitized_output = sanitize_text(result.stdout or "", (username, host))
    if sanitized_output:
        log_event("ssh-output", sanitized_output)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH connectivity probe exited with status {result.returncode}"
        )
    log_event("ssh", "connectivity probe succeeded")


def wait_for_vpn_ready(
    context: BrowserContext,
    gpclient: subprocess.Popen[str],
    output_queue: Queue[str | None],
) -> None:
    assert gpclient.stdin is not None

    url_pattern = re.compile(r"https?://[^\s)\]]+")
    waiting_for_auth_url = False
    auth_url_deadline: float | None = None
    vpn_deadline = monotonic() + VPN_READY_TIMEOUT_SECONDS
    auth_round = 0
    previous_otp: str | None = None
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

    log_event(
        "gpclient",
        f"waiting for VPN readiness; timeout={VPN_READY_TIMEOUT_SECONDS}s",
    )

    while monotonic() < vpn_deadline:
        now = monotonic()
        if auth_url_deadline is not None and now >= auth_url_deadline:
            raise RuntimeError(
                f"gpclient announced manual authentication but did not provide a URL "
                f"within {AUTH_URL_TIMEOUT_SECONDS}s; "
                f"recent_output={recent_output_summary(recent_lines, runtime_secrets)}"
            )

        if gpclient.poll() is not None and output_queue.empty():
            raise RuntimeError(
                f"gpclient exited before VPN readiness with status {gpclient.returncode}; "
                f"recent_output={recent_output_summary(recent_lines, runtime_secrets)}"
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
                f"recent_output={recent_output_summary(recent_lines, runtime_secrets)}"
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
        log_event(stage, f"callback forwarded to gpclient; length={len(callback)}")
        waiting_for_auth_url = False
        auth_url_deadline = None

    raise RuntimeError(
        f"VPN readiness was not detected within {VPN_READY_TIMEOUT_SECONDS}s; "
        f"recent_output={recent_output_summary(recent_lines, runtime_secrets)}"
    )


def start_vpn() -> int:
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
            wait_for_vpn_ready(context, gpclient, output_queue)
            run_ssh_probe()
            return 0
        finally:
            try:
                stop_gpclient(gpclient, output_thread)
            finally:
                context.close()
                browser.close()
                log_event("cleanup", "browser and VPN resources released")


if __name__ == "__main__":
    raise SystemExit(start_vpn())
