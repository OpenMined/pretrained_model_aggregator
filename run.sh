#!/bin/sh

set -e

uv venv -p 3.12 .venv
uv pip install -U -r requirements.txt
. .venv/bin/activate

while true; do
    echo "Running 'pretrained_model_aggregator' with $(python3 --version) at '$(which python3)'"
    python3 main.py

    echo "Sleeping for 10 seconds..."
    sleep 10
done

deactivate
