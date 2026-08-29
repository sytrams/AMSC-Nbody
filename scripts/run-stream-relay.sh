#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

BUILD_DIR=${NBODY_RELAY_BUILD_DIR:-"$PROJECT_DIR/build/relay"}
RELAY=${NBODY_RELAY_EXECUTABLE:-"$BUILD_DIR/nbody_stream_relay"}
TOKEN_FILE=${NBODY_RELAY_TOKEN_FILE:-}
SOURCE_BIND=${NBODY_RELAY_SOURCE_BIND:-0.0.0.0}
SOURCE_PORT=${NBODY_RELAY_SOURCE_PORT:-4748}
CLIENT_BIND=${NBODY_RELAY_CLIENT_BIND:-127.0.0.1}
CLIENT_PORT=${NBODY_RELAY_CLIENT_PORT:-4747}

if [ -z "$TOKEN_FILE" ]; then
  echo "NBODY_RELAY_TOKEN_FILE is required." >&2
  echo "Create it once with ./scripts/create-stream-token.sh FILE." >&2
  exit 2
fi
if [ ! -r "$TOKEN_FILE" ]; then
  echo "Relay token is not readable: $TOKEN_FILE" >&2
  exit 1
fi

if [ "${NBODY_SKIP_BUILD:-0}" != "1" ]; then
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if command -v ninja >/dev/null 2>&1; then
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=OFF \
        -DNBODY_BUILD_SIMULATION=OFF \
        -DNBODY_BUILD_STREAMING=ON \
        -DNBODY_BUILD_VIEWER=OFF
    else
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=OFF \
        -DNBODY_BUILD_SIMULATION=OFF \
        -DNBODY_BUILD_STREAMING=ON \
        -DNBODY_BUILD_VIEWER=OFF
    fi
  fi
  cmake --build "$BUILD_DIR" --target nbody_stream_relay --parallel
fi

if [ ! -x "$RELAY" ]; then
  echo "Relay executable not found: $RELAY" >&2
  echo "Remove NBODY_SKIP_BUILD=1 or set NBODY_RELAY_EXECUTABLE." >&2
  exit 1
fi

echo "Accepting authenticated compute sources on $SOURCE_BIND:$SOURCE_PORT"
echo "Accepting SSH-forwarded Mac clients on $CLIENT_BIND:$CLIENT_PORT"
exec "$RELAY" \
  --token-file "$TOKEN_FILE" \
  --source-bind "$SOURCE_BIND" \
  --source-port "$SOURCE_PORT" \
  --client-bind "$CLIENT_BIND" \
  --client-port "$CLIENT_PORT" \
  "$@"
