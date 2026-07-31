#!/usr/bin/env python3

import argparse
import html
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL_RE = re.compile(r"https://[^\s]+")
CALLBACK_RE = re.compile(r"globalprotectcallback:[^\"'<>\s]+")


def read_lines(stream, output_queue):
    for line in iter(stream.readline, ""):
        output_queue.put(line)


def obtain_login_url(process, timeout=60):
    lines = queue.Queue()
    threading.Thread(
        target=read_lines,
        args=(process.stderr, lines),
        daemon=True,
    ).start()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if process.poll() is not None:
                raise RuntimeError("gpauth exited before producing a login URL")
            continue

        match = URL_RE.search(line)
        if match:
            return match.group(0)

    raise TimeoutError("Timed out waiting for the gpauth login URL")


def authenticate_in_browser(login_url):
    username = os.environ["POLIMI_USER"]
    password = os.environ["POLIMI_PASSWORD"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(login_url, wait_until="domcontentloaded")

        # Replace these locators with the actual Polimi labels.
        page.get_by_label(
            re.compile(r"username|user|codice persona", re.I)
        ).fill(username)

        page.get_by_label(
            re.compile(r"password", re.I)
        ).fill(password)

        page.get_by_role(
            "button",
            name=re.compile(r"login|sign in|accedi", re.I),
        ).click()

        # If authentication uses a phone push, simply wait here.
        # If there is an OTP input, fill it only using an approved mechanism.

        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:
            for frame in page.frames:
                content = html.unescape(frame.content())
                match = CALLBACK_RE.search(content)
                if match:
                    callback = match.group(0)
                    browser.close()
                    return callback

            page.wait_for_timeout(500)

        browser.close()
        raise TimeoutError("No globalprotectcallback value was found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal", required=True)
    parser.add_argument("--cookie-file", type=Path, required=True)
    args = parser.parse_args()

    process = subprocess.Popen(
        ["gpauth", args.portal, "--browser", "remote"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    login_url = obtain_login_url(process)
    callback = authenticate_in_browser(login_url)

    if "\n" in callback or "\r" in callback:
        raise RuntimeError("Malformed GlobalProtect callback")

    # Mask the derived credential in subsequent GitHub log output.
    print(f"::add-mask::{callback}", flush=True)

    process.stdin.write(callback + "\n")
    process.stdin.close()

    cookie = process.stdout.read().strip()
    return_code = process.wait(timeout=60)

    if return_code != 0 or not cookie:
        raise RuntimeError("gpauth did not return a VPN cookie")

    if "\n" in cookie or "\r" in cookie:
        raise RuntimeError("Malformed VPN cookie")

    print(f"::add-mask::{cookie}", flush=True)

    args.cookie_file.write_text(cookie + "\n", encoding="utf-8")
    args.cookie_file.chmod(0o600)


if __name__ == "__main__":
    main()