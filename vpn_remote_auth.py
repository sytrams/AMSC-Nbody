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
        page.goto(link)

        page.locator('input[name="username"]').fill(username)
        page.locator('input[name="password"]').fill(password)
        page.locator('button[type="submit"], input[type="submit"]').click()

        otp = pyotp.TOTP(otp_secret).now()
        page.locator('input[name="otp"], input[name="code"]').fill(otp)
        page.locator('button[type="submit"], input[type="submit"]').click()
    body = page.text_content("body") or ""
    m = re.search(r"globalprotectcallback:[^\s]+", body)
    if not m:
        raise RuntimeError("Could not find callback in final page")

    callback = m.group(0)
    browser.close()
    print(callback)

if __name__ == "__main__":
    main()