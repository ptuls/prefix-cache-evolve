#!/bin/sh
set -eu

candidate=/candidate/policy.py
if [ ! -f "$candidate" ]; then
  echo "candidate must be mounted read-only at $candidate" >&2
  exit 2
fi

cp "$candidate" /tmp/candidate.py
exec prefix-cache-evolve "$@" --candidate-program /tmp/candidate.py
