#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$SCRIPT_DIR/cliff-video.mp4"
OUTPUT_DIR="$SCRIPT_DIR/frames"

mkdir -p "$OUTPUT_DIR"

ffmpeg -i "$INPUT" -vf "fps=1" "$OUTPUT_DIR/frame_%06d.png"
