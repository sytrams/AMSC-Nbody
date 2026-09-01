#!/usr/bin/env sh

set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 FRAME_DIRECTORY [VIEWER_OPTIONS...]" >&2
  echo "Example: $0 /shared/nbody-frames --port 8080" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
FRAME_DIR=$1
shift

BUILD_DIR=${NBODY_BROWSER_BUILD_DIR:-"$PROJECT_DIR/build/browser-viewer"}
VIEWER=${NBODY_BROWSER_EXECUTABLE:-"$BUILD_DIR/nbody_browser_viewer"}

if [ "${NBODY_SKIP_BUILD:-0}" != "1" ]; then
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if command -v ninja >/dev/null 2>&1; then
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=OFF \
        -DNBODY_BUILD_SIMULATION=OFF \
        -DNBODY_BUILD_VIEWER=OFF \
        -DNBODY_BUILD_STREAMING=ON
    else
      cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DNBODY_ENABLE_CUDA=OFF \
        -DNBODY_BUILD_SIMULATION=OFF \
        -DNBODY_BUILD_VIEWER=OFF \
        -DNBODY_BUILD_STREAMING=ON
    fi
  fi
  cmake --build "$BUILD_DIR" --target nbody_browser_viewer --parallel
fi

if [ ! -x "$VIEWER" ]; then
  echo "Browser viewer executable not found: $VIEWER" >&2
  echo "Remove NBODY_SKIP_BUILD=1 or set NBODY_BROWSER_EXECUTABLE." >&2
  exit 1
fi

exec "$VIEWER" --frames-dir "$FRAME_DIR" "$@"
