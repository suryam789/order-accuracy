#!/bin/sh
set -eu

# =============================================================================
# On-Demand RTSP Streamer
# =============================================================================
# Uses MediaMTX with on-demand streaming. No 2PC sync, no ffmpeg pre-start.
#
# How it works:
#   1. MediaMTX starts and listens on RTSP port
#   2. When a GStreamer client connects to rtsp://rtsp-streamer:8554/<path>
#   3. MediaMTX spawns ffmpeg to stream /media/<path>.mp4
#   4. Client receives frame 0 immediately — no race condition
#   5. When all clients disconnect, ffmpeg exits after 30s timeout
#
# Stream naming:
#   rtsp://rtsp-streamer:8554/station_1  → streams /media/station_1.mp4
#   rtsp://rtsp-streamer:8554/test       → streams /media/test.mp4
# =============================================================================

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
MEDIAMTX_BIN=${MEDIAMTX_BIN:-/opt/rtsp-streamer/mediamtx}

if [ ! -d "$MEDIA_DIR" ]; then
  echo "Media directory $MEDIA_DIR does not exist" >&2
  exit 1
fi

if [ ! -x "$MEDIAMTX_BIN" ]; then
  echo "mediamtx binary $MEDIAMTX_BIN not found or not executable" >&2
  exit 1
fi

# Verify at least one video file exists
set -- "$MEDIA_DIR"/*.mp4
if [ ! -e "$1" ]; then
  echo "No .mp4 files found in $MEDIA_DIR" >&2
  exit 1
fi

# Find the source video (first .mp4 or RTSP_STREAM_NAME if set)
SOURCE_VIDEO=${RTSP_STREAM_NAME:-}
if [ -n "$SOURCE_VIDEO" ] && [ -f "$MEDIA_DIR/${SOURCE_VIDEO}.mp4" ]; then
  source_file="$MEDIA_DIR/${SOURCE_VIDEO}.mp4"
else
  set -- "$MEDIA_DIR"/*.mp4
  source_file="$1"
fi

source_name=$(basename "$source_file" .mp4)

echo "=== On-Demand RTSP Streamer ==="
echo "Media directory: $MEDIA_DIR"
echo "Source video: $source_file"
echo "Available streams:"
for f in "$MEDIA_DIR"/*.mp4; do
  name=$(basename "$f" .mp4)
  echo "  rtsp://rtsp-streamer:${RTSP_PORT}/${name}"
done
echo "Streams start on-demand when a client connects."
echo "================================"

# Start MediaMTX — it handles everything via mediamtx.yml config
exec "$MEDIAMTX_BIN"
