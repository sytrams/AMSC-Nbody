#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

tmp_dir="$(mktemp -d)"
vpn_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    set +e

    if [[ -n "$vpn_pid" ]]; then
        sudo -n gpclient disconnect >/dev/null 2>&1
        kill "$vpn_pid" >/dev/null 2>&1
        wait "$vpn_pid" >/dev/null 2>&1
    fi

    rm -rf -- "$tmp_dir"
    exit "$status"
}
trap cleanup EXIT INT TERM

: "${GP_PORTAL:?Missing GP_PORTAL}"
: "${SSH_HOST:?Missing SSH_HOST}"
: "${SSH_USER:?Missing SSH_USER}"
: "${SSH_PRIVATE_KEY:?Missing SSH_PRIVATE_KEY}"
: "${SSH_KNOWN_HOSTS:?Missing SSH_KNOWN_HOSTS}"

ssh_port="${SSH_PORT:-22}"
key_file="$tmp_dir/id_ed25519"
known_hosts_file="$tmp_dir/known_hosts"
cookie_file="$tmp_dir/gp-cookie"
vpn_log="$tmp_dir/gpclient.log"

# Create SSH files without exposing their contents in command arguments.
printf '%s\n' "$SSH_PRIVATE_KEY" | tr -d '\r' >"$key_file"
printf '%s\n' "$SSH_KNOWN_HOSTS" >"$known_hosts_file"
chmod 600 "$key_file" "$known_hosts_file"

# Python drives gpauth and the JS authentication flow.
python scripts/auth_vpn.py \
    --portal "$GP_PORTAL" \
    --cookie-file "$cookie_file"

# Remove IdP credentials from the shell environment when no longer needed.
unset POLIMI_USER POLIMI_PASSWORD POLIMI_TOTP_SEED

# gpclient consumes the cookie through stdin and then remains in the background.
sudo -n gpclient connect "$GP_PORTAL" \
    --cookie-on-stdin \
    <"$cookie_file" \
    >"$vpn_log" 2>&1 &

vpn_pid=$!

# The redirection was opened before the background command started, so the
# filename can be removed while gpclient retains its open file descriptor.
rm -f -- "$cookie_file"

# Wait until the private SSH endpoint is reachable.
vpn_ready=false
for _ in {1..60}; do
    if ! kill -0 "$vpn_pid" 2>/dev/null; then
        echo "gpclient terminated before the VPN became ready" >&2
        # Do not print the unfiltered VPN log: it can contain cookies.
        exit 1
    fi

    if nc -z -w 2 "$SSH_HOST" "$ssh_port"; then
        vpn_ready=true
        break
    fi

    sleep 2
done

if [[ "$vpn_ready" != true ]]; then
    echo "SSH endpoint did not become reachable through the VPN" >&2
    exit 1
fi

ssh -T \
    -i "$key_file" \
    -p "$ssh_port" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$known_hosts_file" \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    "$SSH_USER@$SSH_HOST" \
    -- 'hostname && /path/to/your/remote-command'