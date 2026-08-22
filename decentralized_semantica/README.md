# Fully Decentralized Semantica v3

This package runs Semantica as isolated Python processes communicating only
through TCP ports. Every peer owns its private document shard, profile,
contacts, query state, descriptor cache, and the tree nodes for which it is a
custodian.

The active implementation has no membership registry or privileged
coordinator. The bootstrap peer is only the custodian of the genesis root
node. It cannot list the overlay, assign custodians, advance rounds, admit
peers, resolve peer IDs, export other peers, or stop other peers.

Two deliberately separate modes are available:

- **Active decentralized mode:** start one genesis peer and independently add
  peers using only `bootstrap.json` and a private shard. Concurrent and late
  joins are supported. Existing contacts are not automatically rebuilt.
- **Compatibility experiment:** an external, non-peer coordinator passes a
  successor token through a staged cohort. This is an equivalence harness and
  reproduces the sequential centralized simulator exactly.

## Install

From the repository root:

```bash
python -m pip install -r requirements-decentralized.txt
```

For development:

```bash
python -m pip install -r requirements-decentralized-dev.txt
```

Peers currently bind `127.0.0.1`, so all processes in an experiment must run
on one host.

## Custodian selection

The bootstrap peer holds the empty root as the one unavoidable genesis
assignment. The root has custody seed 0. Every child has its parent's seed plus
one.

When a leaf splits, the eligible custodians for each child are exactly the
unique owners of centroid entries placed in that child. The splitting
custodian cannot choose among them. Candidates are sorted by numeric peer ID
and selected with:

```text
SHA256(
    "semantica-custody-v1"
    || epoch
    || uint64_be(child_seed)
    || child_node_id
    || length-prefixed uint64_be(candidate IDs)
) mod candidate_count
```

`CREATE_LEAF` carries a `CustodyProof`. The selected peer recomputes the node
ID, seed, candidate set, digest, selected peer, and selected endpoint before
accepting the node. Existing node custody never changes after later joins.

## Prepare PubMedQA shards

The DAS-6 defaults use the existing seed-42 PubMedQA embedding bundle and
overlay:

```bash
export RUN_ROOT="$PWD/decentralized-run"

python -m decentralized_semantica prepare \
  --output-dir "$RUN_ROOT/prepared" \
  --peer-count 100 \
  --seed 42
```

For other locations:

```bash
python -m decentralized_semantica prepare \
  --output-dir "$RUN_ROOT/prepared" \
  --data-root /path/to/pubmedqa/embedding-bundle \
  --overlay-root /path/to/pubmedqa/overlay-cache \
  --peer-count 100 \
  --seed 42
```

Preparation writes one `shards/peer_XXXXX.npz` per peer. The prepared manifest
contains algorithm parameters and provenance, not runtime endpoints or a peer
directory. `central_fixture.npz` belongs only to the trusted validator.

## Run the exact compatibility experiment

Use this mode when checking the decentralized protocol against the centralized
simulator:

```bash
python -m decentralized_semantica compat-run \
  --prepared-dir "$RUN_ROOT/prepared" \
  --run-dir "$RUN_ROOT/compat-run" \
  --root-peer-id 0 \
  --rpc-timeout-seconds 30
```

The launcher starts peers sequentially, but the peers never receive the full
cohort. Each receives only its successor and the external coordinator endpoint.
The token serializes centroid insertion, tree statistics, round 0, every
expansion round, queries, and export. The root follows the same token handlers
as every other peer and holds no barrier state.

Validate against a freshly executed centralized oracle:

```bash
python -m decentralized_semantica validate \
  --prepared-dir "$RUN_ROOT/prepared" \
  --run-dir "$RUN_ROOT/compat-run"
```

Acceptance uses exact IDs and ordering plus bitwise-equal canonical float32
values. It compares profiles, quotas, canonical tree contents, all contact
rounds, query paths, per-hop rankings, final rankings, and terminal states.
Physical custodians, TCP endpoints, timings, and message totals are excluded.

Compare with an earlier decentralized run as an additional gate:

```bash
python -m decentralized_semantica validate \
  --prepared-dir "$RUN_ROOT/prepared" \
  --run-dir "$RUN_ROOT/compat-run" \
  --reference-run /path/to/reference/run
```

To turn the ordered cohort into a coordinator-free active overlay after the
equivalence checkpoint, add `--release-active`. The command closes the external
coordinator, rewrites `bootstrap.json` with compatibility control disabled,
and writes `overlay_released.json`. It then remains as a process supervisor so
standalone `join` commands can use the released overlay. Request a local stop
without messaging any peer:

```bash
touch "$RUN_ROOT/compat-run/stop_requested"
```

The exact compatibility checkpoint remains under `state/`; the post-release
states, including late peers, are retained under `released_state/`.

## Run the active decentralized overlay

Start peer 0 in one terminal. It publishes `bootstrap.json` only after its TCP
listener and seed-0 root exist:

```bash
python -m decentralized_semantica start-root \
  --prepared-dir "$RUN_ROOT/prepared" \
  --peer-id 0 \
  --run-dir "$RUN_ROOT/active-run"
```

Then start any peer in another terminal or process. It needs no manifest of
other peers:

```bash
python -m decentralized_semantica join \
  --bootstrap "$RUN_ROOT/active-run/bootstrap.json" \
  --peer-id 1 \
  --shard "$RUN_ROOT/prepared/shards/peer_00001.npz" \
  --run-dir "$RUN_ROOT/active-run"
```

Repeat `join` independently for the remaining shards. Joins may overlap. Each
peer traverses from the root descriptor, inserts through custodians, builds its
own contacts, performs its expansion rounds, runs its query, and remains online
for peer and custodian work.

If the reachable tree cannot yet fill the configured degree, the peer publishes
no partial contact table. It reports `contacts_pending` under `status/` and
retries locally after 1, 2, 4, 8, 16, and 30 seconds, then every 30 seconds,
with deterministic jitter of at most 250 ms.

A later peer uses the same `join` command. Existing peers' contacts remain
unchanged; exact centralized equality is intentionally not required for an
asynchronously changing tree.

Stop each process locally with `SIGINT` or `SIGTERM`. The process exports its
own state before exiting:

```bash
kill -TERM "$PEER_PID"
wait "$PEER_PID"
```

There is intentionally no overlay-wide `finalize` or `shutdown` RPC.

After every peer has stopped and exported locally, validate active-mode
invariants without asking for centralized equality:

```bash
python -m decentralized_semantica validate \
  --prepared-dir "$RUN_ROOT/prepared" \
  --run-dir "$RUN_ROOT/active-run" \
  --structural-only
```

The validator derives an audit inventory from the collected state files; peers
never read that inventory. Add `--prejoin-run /path/to/checkpoint` to require
byte-identical contacts for all peers present before a late join.

## DAS-6

The synchronous acceptance job requests one regular 64-CPU node and uses one
BLAS/OpenMP thread per peer process:

```bash
scripts/submit_decentralized_semantica_100.sh \
  /var/scratch/$USER/decentralized_semantica/v3-compat-001
```

Or submit directly:

```bash
sbatch \
  --export=ALL,PROJECT_DIR="$PWD",REFERENCE_RUN=/path/to/reference/run \
  scripts/run_decentralized_semantica_100.sbatch \
  /var/scratch/$USER/decentralized_semantica/v3-compat-001
```

The separate active acceptance job starts the root and initial peers as
independent commands, admits a late peer, checks that earlier contacts remain
byte-identical, and terminates every PID locally.

## Outputs

| Path | Purpose |
|---|---|
| `bootstrap.json` | Epoch, parameters, and the seed-0 root descriptor; no peer list |
| `status/peer_XXXXX.json` | Locally published lifecycle status for one peer |
| `state/peer_XXXXX.msgpack` | One peer's exported algorithmic and custodian state |
| `logs/peer_XXXXX.jsonl` | Every protocol send and receive for one peer |
| `experiment_peer_inventory.json` | Compatibility-harness evidence; never runtime input |
| `custody_audit.msgpack` | Canonical custody seeds and proofs |
| `tree_snapshot.msgpack` | Canonical tree, excluding physical-path assumptions |
| `centroid_profiles.npz` | Compact fitted profiles |
| `contacts_rounds_0_20.msgpack` | Compatibility contact snapshots |
| `query_paths_results/` | Per-source query paths and results |
| `network_summary.json` | Protocol message and byte totals |
| `validation_report.json` | Structural and exact centralized-parity evidence |

Active mode does not create an experiment inventory unless an external audit
tool explicitly constructs one from collected state. Such inventory is never
read by a peer.

## Tests

```bash
python -m pytest -q tests/test_decentralized_semantica.py
```

The suite covers framing, request correlation, deterministic custody and proof
rejection, endpoint conflicts, absence of root power state and RPCs, contact
pending behavior, descriptor-based expansion, process-isolated exact parity,
concurrent active joining, late joining, and existing-contact stability.

## Deferred work

- Custodian replication, reassignment, churn recovery, and failed-custodian
  routing are not implemented.
- Authentication, Sybil resistance, and malicious-custodian defenses are not
  implemented.
- Peer IDs and endpoints must remain unique and stable; observed conflicts are
  rejected.
- The bootstrap root endpoint must be learned out of band and stay available.
- Existing peers discover newcomers only through a later explicitly initiated
  expansion cycle; newcomers do not rewrite existing contacts automatically.
