#!/usr/bin/env bash
# Build and test hey-jev on the build host.
#
# The working copy lives wherever you edit it; this script syncs the sources to the
# build host and runs the real toolchain there. The build host is Loki (see README):
# it has the Rust toolchain and the C linker that cargo needs, and it is the only
# fleet host Marcin wants holding build load for this project.
#
# target/ on the build host is the build cache and is never overwritten by the sync.
set -euo pipefail

BUILD_HOST="${HEY_JEV_BUILD_HOST:-mb@100.64.82.76}"
BUILD_JUMP="${HEY_JEV_BUILD_JUMP-mb@100.98.212.49}"
BUILD_DIR="${HEY_JEV_BUILD_DIR:-/home/mb/hey-jev}"

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
if [ -n "$BUILD_JUMP" ]; then
  SSH+=(-J "$BUILD_JUMP")
fi

# Loki is owned by the user account, so the fleet jump host is the only route in.
"${SSH[@]}" "$BUILD_HOST" "mkdir -p '$BUILD_DIR'"

tar cz -C "$LOCAL" \
  --exclude=./target --exclude=./.git --exclude=./cache --exclude=./dist \
  . | "${SSH[@]}" "$BUILD_HOST" "tar xz -C '$BUILD_DIR'"

"${SSH[@]}" "$BUILD_HOST" "cd '$BUILD_DIR' && export PATH=\"\$HOME/.cargo/bin:\$PATH\" && export CARGO_INCREMENTAL=0 && cargo test --quiet && cargo build --release --quiet && echo CHECK_OK"
