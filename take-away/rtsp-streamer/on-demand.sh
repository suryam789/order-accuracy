#!/bin/sh
# On-demand ffmpeg launcher called by MediaMTX runOnDemand.
# MediaMTX sets MTX_PATH as an environment variable (e.g. "station_1").
# Resolves /media/$MTX_PATH.mp4, falls back to first .mp4 in /media.

MEDIA_DIR=/media

# Logging helper — prefixes every message with ISO timestamp and station path
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') [rtsp-streamer][on-demand][${MTX_PATH}] $*"
}

log "On-demand stream requested for path='${MTX_PATH}' (PID=$$)"

f="$MEDIA_DIR/$MTX_PATH.mp4"

if [ -f "$f" ]; then
  log "Found exact video match: $f"
else
  log "Video '$f' not found, falling back to first .mp4 in $MEDIA_DIR"
  f=$(ls "$MEDIA_DIR"/*.mp4 2>/dev/null | head -1)
fi

if [ -z "$f" ] || [ ! -f "$f" ]; then
  log "ERROR: no video file found for path '$MTX_PATH'" >&2
  exit 1
fi

log "Streaming $f → rtsp://localhost:8554/$MTX_PATH (ffmpeg PID=$$ starting)"

# LOOP_COUNT controls how many times the video plays:
#   -1 = infinite loop (default)
#    1 = play once then exit
#    N = play N times
LOOP_COUNT=${LOOP_COUNT:--1}
if [ "$LOOP_COUNT" -eq -1 ]; then
  stream_loop_arg="-1"
  log "Loop mode: infinite"
elif [ "$LOOP_COUNT" -eq 1 ]; then
  stream_loop_arg="0"
  log "Loop mode: play once"
else
  stream_loop_arg="$((LOOP_COUNT - 1))"
  log "Loop mode: play ${LOOP_COUNT} times"
fi

exec ffmpeg -hide_banner -loglevel warning -re -stream_loop "$stream_loop_arg" -fflags +genpts -i "$f" -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -keyint_min 10 -sc_threshold 0 -an -f rtsp "rtsp://localhost:8554/$MTX_PATH"
