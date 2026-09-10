# Security

## Candidate Code

Candidate policies are Python programs and must be treated as untrusted code.
The evaluator's static checks, subprocess boundary, timeouts, and resource
limits protect benchmark integrity and availability. They are not a security
sandbox.

Linux workers additionally receive an address-space resource limit. On macOS,
where the equivalent resource limit is unsupported, the parent process samples
worker resident memory and terminates workers that exceed their configured
growth budget. Candidate execution fails closed when isolation or memory
monitoring cannot be established.

Do not evaluate untrusted candidates directly on a workstation containing
credentials or sensitive files. Use the container profile under
`docker/sandbox`, which runs as a non-root user with:

- no network access;
- a read-only root filesystem;
- all Linux capabilities dropped;
- `no-new-privileges`;
- CPU, memory, process, and temporary-storage limits;
- the candidate mounted read-only.

Run:

```bash
docker/sandbox/run.sh path/to/candidate.py
```

The profile is defense in depth. For hostile multi-tenant workloads, use a
stronger VM or microVM boundary and isolate the Docker daemon itself.

For automated source evolution, `problem.settings.sandbox_image` selects an
already-built image (prefer its immutable image ID). The source evaluator stages
only selected trace inputs, runs a non-root networkless container, and does not
fall back to host execution. Search cleanup removes its own remaining containers;
the container also enforces a deadline if its client is interrupted. The
worker discards candidate stdout/stderr and converts candidate-triggered exits
to failed evaluations, keeping the result channel owned by the verifier. The
source contract permits attribute writes only on candidate-owned `self` state,
and permitted imports resolve to read-only candidate facades instead of shared
module namespaces. This prevents aliasing from mutating trusted scoring modules
or verifier inputs. Direct runtime-type lookup and unbound candidate-class access
are rejected so a helper cannot relabel shared class state as `self`. Selecting
the Docker backend enforces this grammar even if a caller's trusted host
configuration disables optional source-pattern checks. The
host streams container output with limits of 32 MiB for results and 64 KiB for
errors; exceeding either limit terminates the client and removes the container.
The automated Docker backend supports Linux and macOS hosts. The `.dockerignore`
allowlist excludes credentials, traces, and other workspace files
from image builds. Trusted built-in baseline helpers remain available for local
simulation. See the production evolution workflow in `docs/reproducibility.md`.

## Reporting Vulnerabilities

Report vulnerabilities privately through the repository host's security
advisory mechanism. Include the affected commit, reproduction steps, and the
security boundary that was crossed. Do not include API keys, production traces,
or raw prompt content.
