#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

BUILD_DIR=${NBODY_CLIENT_BUILD_DIR:-"$PROJECT_DIR/build/client"}
CLIENT=${NBODY_CLIENT_EXECUTABLE:-"$BUILD_DIR/nbody_stream_client"}
HOST=${NBODY_STREAM_HOST:-127.0.0.1}
PORT=${NBODY_STREAM_PORT:-4747}
OUTPUT_DIR=${NBODY_FRAME_OUTPUT_DIR:-"$PROJECT_DIR/received-frames"}

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

  cmake --build "$BUILD_DIR" --target nbody_stream_client --parallel
fi

if [ ! -x "$CLIENT" ]; then
  echo "Client executable not found: $CLIENT" >&2
  echo "Remove NBODY_SKIP_BUILD=1 or set NBODY_CLIENT_EXECUTABLE." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Connecting to $HOST:$PORT"
echo "Writing complete frames to $OUTPUT_DIR"

# Additional arguments are placed last and can override the defaults. For
# example, pass --once to retrieve the current backlog and exit.
exec "$CLIENT" \
  --host "$HOST" \
  --port "$PORT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
