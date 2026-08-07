from html import unescape
import os
import re
import subprocess
import sys
from time import monotonic
from urllib.parse import urlsplit

import pyotp
from playwright.sync_api import BrowserContext, Locator, Page, sync_playwright


CALLBACK_PREFIX = "globalprotectcallback:"
CALLBACK_PATTERN = re.compile(r"globalprotectcallback:[^\s\"'<>]+", re.IGNORECASE)
AUTH_TIMEOUT_SECONDS = 60
POLL_INTERVAL_MS = 250

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


def next_unused_otp(page: Page, previous_otp: str | None, deadline: float) -> str:
    totp = pyotp.TOTP(required_secret("POLIMI_TOTP"))
    announced_wait = False

    while True:
        otp_code = totp.now()
        if otp_code != previous_otp:
            return otp_code

        if monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for a fresh OTP value.")

        if not announced_wait:
            print(
                "Gateway requested another OTP in the same TOTP window; "
                "waiting for a fresh value.",
                file=sys.stderr,
                flush=True,
            )
            announced_wait = True
        page.wait_for_timeout(500)


def insert_otp(
    page: Page,
    otp_field: Locator,
    previous_otp: str | None,
    deadline: float,
) -> str:
    otp_code = next_unused_otp(page, previous_otp, deadline)
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
                print(
                    f"Completed {stage} browser authentication.",
                    file=sys.stderr,
                    flush=True,
                )
                return callback, used_otp
            raise RuntimeError(
                f"Could not load the {stage} authentication page; "
                f"{page_diagnostics(context, secrets)}"
            ) from exc

        while monotonic() < deadline:
            callback = observed_callbacks[-1] if observed_callbacks else None
            callback = callback or get_callback(context)
            if callback:
                print(
                    f"Completed {stage} browser authentication.",
                    file=sys.stderr,
                    flush=True,
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
                print(
                    f"Submitting credentials for {stage} authentication.",
                    file=sys.stderr,
                    flush=True,
                )
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
                print(
                    f"Submitting OTP for {stage} authentication.",
                    file=sys.stderr,
                    flush=True,
                )
                used_otp = insert_otp(
                    otp_page,
                    otp_field,
                    previous_otp,
                    deadline,
                )
                secrets = (username, password, used_otp)
                otp_submitted = True
                continue

            confirmation_control = find_confirmation_control(context)
            if confirmation_control is not None and not confirmation_submitted:
                _, confirmation_button = confirmation_control
                print(
                    f"Confirming {stage} browser authentication.",
                    file=sys.stderr,
                    flush=True,
                )
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


def start_vpn():
    # gpauth = subprocess.Popen(
    #     ["gpauth", required_secret("POLIMI_VPN"), "--browser", "remote"],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     stdin=subprocess.PIPE,
    #     text=True,
    #     bufsize=1,
    # )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
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
            start_new_session=True,
        )

        try:
            assert gpclient.stdout is not None
            assert gpclient.stdin is not None

            url_pattern = re.compile(r"https?://[^\s)\]]+")
            waiting_for_auth_url = False
            auth_round = 0
            previous_otp: str | None = None

            for line in gpclient.stdout:
                # Show the initial program's output.
                print(line, end="", flush=True)

                if "Manual Authentication Required" in line:
                    waiting_for_auth_url = True

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
                    stage = f"authentication round {auth_round}"

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

                # Keep stdin open so the gateway can request another remote login.
                gpclient.stdin.write(callback.rstrip("\r\n") + "\n")
                gpclient.stdin.flush()
                waiting_for_auth_url = False

            return_code = gpclient.wait()
            print(f"gpclient exited with status {return_code}")
            return return_code
        except Exception:
            if gpclient.poll() is None:
                gpclient.terminate()
            raise
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    start_vpn()
