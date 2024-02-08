#!/bin/sh
isort /app/transformer/*.py
black /app/transformer/*.py

python3 -B /app/transformer/run.py "$@"
