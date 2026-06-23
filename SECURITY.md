# Security

## Candidate Code

Candidate policies are Python programs and must be treated as untrusted code.
The evaluator's static checks, subprocess boundary, timeouts, and resource
limits protect benchmark integrity and availability. They are not a security
sandbox.

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

## Reporting Vulnerabilities

Report vulnerabilities privately through the repository host's security
advisory mechanism. Include the affected commit, reproduction steps, and the
security boundary that was crossed. Do not include API keys, production traces,
or raw prompt content.
