#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

BUILD_DIR=${NBODY_SERVER_BUILD_DIR:-"$PROJECT_DIR/build/simulation"}
SERVER=${NBODY_SERVER_EXECUTABLE:-"$BUILD_DIR/nbody_stream_server"}
SPOOL_DIR=${NBODY_STREAM_SPOOL_DIR:-"${SCRATCH:-$PROJECT_DIR}/nbody-frames-${SLURM_JOB_ID:-manual}"}
BIND_ADDRESS=${NBODY_STREAM_BIND:-127.0.0.1}
PORT=${NBODY_STREAM_PORT:-4747}
POLL_MS=${NBODY_STREAM_POLL_MS:-100}
RELAY_HOST=${NBODY_STREAM_RELAY_HOST:-}
RELAY_PORT=${NBODY_STREAM_RELAY_PORT:-4748}
RELAY_TOKEN_FILE=${NBODY_RELAY_TOKEN_FILE:-}
RECONNECT_MS=${NBODY_STREAM_RECONNECT_MS:-1000}

if [ "${NBODY_SKIP_BUILD:-0}" != "1" ]; then
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if command -v ninja >/dev/null 2>&1; then
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=ON \
        -DNBODY_BUILD_SIMULATION=ON \
        -DNBODY_BUILD_STREAMING=ON \
        -DNBODY_BUILD_VIEWER=OFF
    else
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=ON \
        -DNBODY_BUILD_SIMULATION=ON \
        -DNBODY_BUILD_STREAMING=ON \
        -DNBODY_BUILD_VIEWER=OFF
    fi
  fi

  cmake --build "$BUILD_DIR" --target nbody_stream_server --parallel
fi

if [ ! -x "$SERVER" ]; then
  echo "Server executable not found: $SERVER" >&2
  echo "Load the cluster CUDA toolchain, remove NBODY_SKIP_BUILD=1, or set NBODY_SERVER_EXECUTABLE." >&2
  exit 1
fi

mkdir -p "$SPOOL_DIR"

echo "Serving frames from $SPOOL_DIR"
if [ -n "$RELAY_HOST" ]; then
  if [ -z "$RELAY_TOKEN_FILE" ]; then
    echo "NBODY_RELAY_TOKEN_FILE is required with NBODY_STREAM_RELAY_HOST." >&2
    exit 2
  fi
  echo "Connecting outward to relay $RELAY_HOST:$RELAY_PORT"
  exec "$SERVER" \
    --spool-dir "$SPOOL_DIR" \
    --poll-ms "$POLL_MS" \
    --relay-host "$RELAY_HOST" \
    --relay-port "$RELAY_PORT" \
    --relay-token-file "$RELAY_TOKEN_FILE" \
    --reconnect-ms "$RECONNECT_MS" \
    "$@"
fi

echo "Listening on $BIND_ADDRESS:$PORT"
exec "$SERVER" \
  --spool-dir "$SPOOL_DIR" \
  --bind "$BIND_ADDRESS" \
  --port "$PORT" \
  --poll-ms "$POLL_MS" \
  "$@"
