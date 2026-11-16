#!/bin/bash
# Persistent SSH tunnel to the exp8 service (localhost:8800). Reconnects forever (VPN drops, server restarts).
# The runner (remote.py) reuses any tunnel already listening on 8800, so this one becomes the shared tunnel.
while true; do
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ConnectTimeout=10 -N -L 8800:localhost:8800 user@gpu-server
  sleep 5
done
