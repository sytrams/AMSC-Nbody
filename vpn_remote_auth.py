#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from playwright.sync_api import sync_playwright
import pyotp

def main():
    link = sys.argv[1]
    username = os.environ["POLIMI_USERNAME"]
    password = os.environ["POLIMI_PASSWORD"]
    otp_secret = os.environ["POLIMI_OTP_SECRET"]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(link, wait_until="domcontentloaded")

        page.locator("#myform").evaluate("form => form.submit()")

        page.wait_for_url("**shibidp.polimi.it/**", timeout=30000)
        page.wait_for_load_state("domcontentloaded")

        print(page.url, file=sys.stderr, flush=True)
        page.screenshot(path="debug-login.png")

        username = page.locator(
            'input[name="username"], input[name="j_username"], input[id="username"], input[type="text"], input[type="email"]'
        ).first

        password = page.locator(
            'input[name="password"], input[name="j_password"], input[id="password"], input[type="password"]'
        ).first

        username.wait_for(timeout=30000)
        password.wait_for(timeout=30000)

        username.fill(os.environ["POLIMI_USERNAME"])
        password.fill(os.environ["POLIMI_PASSWORD"])
    body = page.text_content("body") or ""
    m = re.search(r"globalprotectcallback:[^\s]+", body)
    if not m:
        raise RuntimeError("Could not find callback in final page")

    callback = m.group(0)
    browser.close()
    print(callback)

if __name__ == "__main__":
    main()