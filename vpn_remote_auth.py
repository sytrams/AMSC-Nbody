from playwright.sync_api import sync_playwright
import sys
import re
import subprocess
import pyotp
import os

def required_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required secret {name} is missing")
    return value

def get_callback(page):
    match = re.search(r'href\s*=\s*["\'](globalprotectcallback:[^"\']+)["\']', page.content())
    callback = match.group(1) if match else None
    return callback

def insert_otp(page):
    otp = page.locator('input[name*="otp"]').first
    totp = pyotp.TOTP(required_secret("POLIMI_TOTP")).now()
    otp.fill(totp)
    page.get_by_role(
            "button",
            name=re.compile(r"evn_continua|continua|Continua", re.I)).click()
            

def insert_credentials(page, username, password):
    username_field = page.locator(
        'input[autocomplete="username"], '
        'input[name*="login" i], '
        'input[type="email"]'
    ).first
    password_field = page.locator('input[type="password"]').first
    username_field.fill(username)
    password_field.fill(password)

    page.get_by_role(
        "button",
        name=re.compile(r"log in|sign in|Accedi", re.I)
    ).click()

def browser_session(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        response = page.goto(url, wait_until="domcontentloaded")

        # Execute JavaScript inside the page
        title = page.evaluate("() => document.title")
        insert_credentials(page, required_secret("POLIMI_USERNAME"),  required_secret("POLIMI_PASSWORD"))
        page.evaluate("() => document.title")
        page.wait_for_timeout(2000)
        insert_otp(page)
        page.wait_for_timeout(2000)
        callback =  get_callback(page)
        return callback

def start_vpn():
    # gpauth = subprocess.Popen(
    #     ["gpauth", required_secret("POLIMI_VPN"), "--browser", "remote"],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     stdin=subprocess.PIPE,
    #     text=True,
    #     bufsize=1,
    # )

    gpclient = subprocess.Popen(
        [
            "sudo",
            "gpclient",
            "connect",
            required_secret("POLIMI_VPN"),
            "--browser",
            "remote"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,

        text=True,
        start_new_session = True,
    )

    url_pattern = re.compile(r"https?://\S+")

    for line in gpclient.stderr:
        # Show the initial program's output
        print(line, end="", flush=True)

        match = url_pattern.search(line)
        if not match:
            continue

        url = match.group(0).rstrip(".,);]")

        # helper.py receives the URL and prints the required response
        callback = browser_session(url).strip()
        if not callback or not callback.startswith("globalprotectcallback:"):
            gpclient.terminate()
            raise RuntimeError(
                "Expected globalprotectcallback:, "
                f"but browser_session returned {callback!r}"
            )
            break
        #response = result.stdout.rstrip("\n")

        # Feed the string back to the still-running initial program
        #initial.stdin.write(response + "\n")
        #initial.stdin.flush()

    gpclient.stdin.write(callback.rstrip("\r\n") + "\n")
    gpclient.stdin.flush()
    gpclient.stdin.close()
    print(gpclient.pid)
    for line in gpclient.stdout:
            print(f"[gpclient] {line}", end="", flush=True)

    return_code = gpclient.wait()
    print(f"gpclient exited with status {return_code}")


if __name__ == "__main__":
    start_vpn() 
