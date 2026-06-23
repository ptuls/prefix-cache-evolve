#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 CANDIDATE.py [prefix-cache-evolve arguments...]" >&2
  exit 2
fi

candidate=$1
shift
candidate_dir=$(CDPATH= cd -- "$(dirname -- "$candidate")" && pwd)
candidate_name=$(basename -- "$candidate")
image=${PREFIX_CACHE_SANDBOX_IMAGE:-prefix-cache-evolve-sandbox}
repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ "$#" -eq 0 ]; then
  set -- --baseline-report --quick
fi

docker build --tag "$image" --file "$repository_root/docker/sandbox/Dockerfile" "$repository_root"
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 64 \
  --memory 1g \
  --cpus 2 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --mount "type=bind,src=$candidate_dir/$candidate_name,dst=/candidate/policy.py,readonly" \
  "$image" \
  "$@"
