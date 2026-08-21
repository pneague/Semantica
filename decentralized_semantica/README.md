# Membership-Free Decentralized Semantica v2

This package runs Semantica as isolated peer processes. Each peer binds an
operating-system-assigned TCP port, owns only its private shard, and performs all
peer and custodian interactions through length-prefixed MessagePack RPCs. Peers
receive one root-custodian endpoint rather than a membership list. The root
discovers joining endpoints at runtime and resolves unknown peer IDs on demand.

## Installation and source dependencies

From a clean clone, install the bounded runtime dependency set with:

```bash
python -m pip install -r requirements-decentralized.txt
```

For local test execution, use `requirements-decentralized-dev.txt`. The package
intentionally uses the repository's existing centralized implementation as its
independent validation oracle. Consequently, the source dependency closure for
this implementation includes `unified_retrieval_benchmark.py`,
`peer_graph_comparison.py`, `pubmedqa_overlay_benchmark.py`, and
`retrieval_benchmark.py`; these files must be committed together with the
`decentralized_semantica/` package.

```mermaid
flowchart LR
    L["Launcher<br/>setup only"] -->|"configure one endpoint"| R["Root custodian peer"]
    P["Joining peer processes"] -->|"JOIN"| R
    R -->|"lookup on cache miss"| P
    P <-->|"successor tokens, tree RPCs,<br/>expansion, query hops"| P
    P <-->|"one custodian per node"| C["Distributed centroid tree"]
    P --> O["Per-peer state and network logs"]
    O --> V["Centralized oracle validator"]
```

## Commands

On DAS-6, the acceptance configuration uses the defaults:

```bash
python -m decentralized_semantica prepare \
  --output-dir /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001/prepared

python -m decentralized_semantica run \
  --prepared-dir /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001/prepared \
  --run-dir /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001/run

python -m decentralized_semantica validate \
  --prepared-dir /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001/prepared \
  --run-dir /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001/run
```

The default `run` command uses deterministic synchronous compatibility mode:
the root starts first, the remaining processes join in staged peer order, and a
successor token serializes tree insertion, round 0, expansion rounds, queries,
and export. Use `--keep-active` to leave the overlay online. An independent peer
can then join with only its private shard and `bootstrap.json`:

```bash
python -m decentralized_semantica join \
  --bootstrap RUN_ROOT/run/bootstrap.json \
  --peer-id 100 \
  --shard /path/to/private-peer-100.npz \
  --run-dir RUN_ROOT/run

python -m decentralized_semantica finalize \
  --bootstrap RUN_ROOT/run/bootstrap.json
```

Submit the complete prepare/run/validate workflow after syncing the source to
the login node:

```bash
scripts/submit_decentralized_semantica_100.sh \
  /var/scratch/pmneague/Low_Coord_Semantic_DHT/decentralized_semantica/run_001
```

The submission helper creates the scratch and Slurm log directories and checks
that Slurm node metadata is available. The job requests one regular 64-CPU node,
starts 100 peer processes, and pins BLAS/OpenMP libraries to one thread per
process.

## Outputs

The run directory contains a peer-list-free `bootstrap.json`, a run manifest,
the final discovered registry and pinned custody assignments, one JSONL network
log and one MessagePack state file per peer, a canonical tree snapshot,
profile arrays and CSV metadata, contacts for every round, query paths and
results (one bounded MessagePack artifact per source peer), message/byte totals,
phase timings, and `validation_report.json`.

The validator first enforces protocol and graph invariants. Its default full
mode reconstructs the same 100-peer fixture with the existing unified benchmark,
runs the centralized query chain sequentially in the same peer order, and
compares fitted profiles, canonical leaf membership, split centers,
per-centroid contacts for rounds 0–20, exact hop sequences, exact per-hop and
merged document ordering, terminal reasons, and scores within the declared
float32 tolerance. `--reference-run` additionally requires bitwise-identical
algorithm state against a previous decentralized run while excluding physical
ports, custodian placement, and wire totals. `--structural-only` is available
for dynamic joins and handcrafted fixtures that do not have a matching static
centralized fixture. For a kept-active run, `--prejoin-run` compares the
serialized contact tables from an initial state checkpoint with the finalized
run and requires them to remain byte-identical after late admissions.

## Deliberate v2 limits

The explicit root endpoint is learned out of band and remains a single bootstrap
and lookup authority. A tree node has one pinned rendezvous-selected custodian.
Late admission organizes the newcomer only; existing peers discover it during a
future expansion cycle. TCP failure or timeout aborts the run. Departures,
custodian replication, reassignment, malicious peers, authentication, recovery,
and protected random links are outside this version.
