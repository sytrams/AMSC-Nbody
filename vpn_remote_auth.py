#!/usr/bin/env python3

import os
import re
import sys

import pyotp
from playwright.sync_api import sync_playwright


def main():
    link = sys.argv[1]
    login_username = os.environ["POLIMI_USERNAME"]
    login_password = os.environ["POLIMI_PASSWORD"]
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

        username_input = page.locator(
            'input[name="username"], input[name="j_username"], input[id="username"], input[type="text"], input[type="email"]'
        ).first
        password_input = page.locator(
            'input[name="password"], input[name="j_password"], input[id="password"], input[type="password"]'
        ).first

        username_input.wait_for(timeout=30000)
        password_input.wait_for(timeout=30000)

        username_input.fill(login_username)
        password_input.fill(login_password)

        page.locator('button[type="submit"], input[type="submit"]').first.click()

        otp_code = pyotp.TOTP(otp_secret).now()

        otp_input = page.locator(
            'input[name="otp"], input[name="code"], input[name="totp"], input[id="otp"]'
        ).first
        otp_input.wait_for(timeout=30000)
        otp_input.fill(otp_code)

        page.locator('button[type="submit"], input[type="submit"]').first.click()

        page.wait_for_load_state("domcontentloaded")

        body = page.text_content("body") or ""
        match = re.search(r"globalprotectcallback:[^\s]+", body)
        if not match:
            raise RuntimeError("Could not find callback in final page")

        callback = match.group(0)
        print(callback, flush=True)

        browser.close()


if __name__ == "__main__":
    main()