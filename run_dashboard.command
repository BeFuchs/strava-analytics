#!/bin/bash
# Double-click starter for the Ride Analytics dashboard (macOS).
# Starts the local server and opens the browser at http://localhost:8000.
cd "$(dirname "$0")"
( sleep 2; open "http://localhost:8000" ) &
exec .venv/bin/ride-analytics serve
