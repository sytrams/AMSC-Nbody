#!/usr/bin/env sh

set -eu

TOKEN_FILE=${1:-${NBODY_RELAY_TOKEN_FILE:-}}
if [ -z "$TOKEN_FILE" ]; then
  echo "Usage: $0 TOKEN_FILE" >&2
  echo "The file must be on storage shared by the login and compute nodes." >&2
  exit 2
fi
if [ -e "$TOKEN_FILE" ]; then
  echo "Refusing to overwrite existing token: $TOKEN_FILE" >&2
  exit 1
fi

TOKEN_PARENT=$(dirname -- "$TOKEN_FILE")
mkdir -p "$TOKEN_PARENT"
umask 077
TEMPORARY=$(mktemp "$TOKEN_PARENT/.nbody-stream-token.XXXXXX")
trap 'rm -f -- "$TEMPORARY"' EXIT HUP INT TERM

if command -v openssl >/dev/null 2>&1; then
  TOKEN=$(openssl rand -hex 32)
else
  TOKEN=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
fi
if [ "${#TOKEN}" -ne 64 ]; then
  echo "Failed to generate a 256-bit streaming token" >&2
  exit 1
fi

printf '%s\n' "$TOKEN" >"$TEMPORARY"
chmod 600 "$TEMPORARY"
if ! ln "$TEMPORARY" "$TOKEN_FILE"; then
  echo "Could not publish token without overwriting: $TOKEN_FILE" >&2
  exit 1
fi
rm -f -- "$TEMPORARY"
trap - EXIT HUP INT TERM
echo "Created private streaming token: $TOKEN_FILE"
