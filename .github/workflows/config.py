from __future__ import annotations

import os


def _env(*names: str, default: str | None = None, required: bool = True) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    if required:
        joined = ", ".join(names)
        raise RuntimeError(f"Missing required environment variable. Expected one of: {joined}")
    return default


POLIMI_USERNAME = _env("POLIMI_USERNAME", "CINECA_USERNAME", required=False)
POLIMI_PASSWORD = _env("POLIMI_PASSWORD", "CINECA_PASSWORD", required=False)
POLIMI_OTP_SECRET = _env("POLIMI_OTP_SECRET", "CINECA_OTP_SECRET", required=False)

PORTAL = _env("PORTAL", required=False)
LINK = _env("LINK", required=False)

GPCLIENT_BIN = _env("GPCLIENT_BIN", default="gpclient", required=False)
GPCLIENT_USE_SUDO = _env("GPCLIENT_USE_SUDO", default="1", required=False) != "0"
GPCLIENT_FIX_OPENSSL = _env("GPCLIENT_FIX_OPENSSL", default="0", required=False) == "1"

REQUEST_TIMEOUT_SECONDS = int(_env("REQUEST_TIMEOUT_SECONDS", default="20", required=False))


def require(value: str | None, *names: str) -> str:
    if value:
        return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable. Expected one of: {joined}")
