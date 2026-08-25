# Decentralized Semantica

This repository contains a process-isolated implementation of the
Semantica peer-to-peer algorithm and a synthetic local simulation. Every peer
runs in its own process, listens on its own TCP port, owns a private document
shard, and learns other peers only through protocol messages.

The genesis peer holds the initial empty tree node but has no membership list
or coordinator privileges. When a tree leaf splits, its child custodian is
selected deterministically from the child centroid owners and verifies the
custody proof before accepting the node.

## Run the local simulation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-decentralized.txt

python -m decentralized_semantica simulate \
  --work-dir demo-run \
  --peer-count 4
```

The smoke simulation generates deterministic synthetic private shards, starts
four real peer processes, verifies that every peer is listening and has joined
the distributed tree, lets them run briefly, and then stops them. `demo-run/`
is generated locally, safe to reuse, and ignored by Git.

## Run the centralized simulation

The centralized simulator uses the same shard format and algorithm kernels but
keeps every profile, tree node, contact list, and query in one process:

```bash
python -m decentralized_semantica centralized-simulate \
  --work-dir demo-run/centralized \
  --peer-count 4
```

It can also use a compact AOL bundle containing `document_ids` with shape
`[peers, documents]` and `document_embeddings` with shape
`[peers, documents, dimensions]`:

```bash
python -m decentralized_semantica centralized-simulate \
  --work-dir demo-run/aol-centralized \
  --aol-bundle /path/to/aol_peer_bundle.npz \
  --peer-count 100
```

The command prints aggregate completion counts and does not save rankings or
experiment results.

## Start peers manually

Prepare the synthetic shards:

```bash
python -m decentralized_semantica prepare-demo \
  --output-dir demo-run/prepared \
  --peer-count 4
```

Start the genesis peer in one terminal:

```bash
python -m decentralized_semantica start-root \
  --prepared-dir demo-run/prepared \
  --run-dir demo-run/run
```

After `demo-run/run/bootstrap.json` appears, start another peer in a second
terminal:

```bash
python -m decentralized_semantica join \
  --bootstrap demo-run/run/bootstrap.json \
  --peer-id 1 \
  --shard demo-run/prepared/shards/peer_00001.npz \
  --run-dir demo-run/run
```

Repeat `join` for the remaining shard IDs. Stop each process locally with
Ctrl-C.

## Algorithm flow

1. A peer fits one or more centroids from its private document embeddings.
2. It traverses the distributed centroid tree from the public genesis
   descriptor and inserts its centroid entries.
3. Custodians split overflowing leaves and assign child custody by a
   deterministic, independently verified hash proof.
4. Each peer discovers nearby owners through tree traversal, builds a bounded
   contact set, and performs one expansion round.
5. A semantic query follows the best unvisited contact and merges local top-k
   document responses.

## Code layout

- `decentralized_semantica/algorithms.py`: centroid fitting, tree splitting,
  custody proofs, contacts, and local retrieval.
- `decentralized_semantica/peer.py`: one isolated TCP peer and its protocol
  handlers.
- `decentralized_semantica/protocol.py`: framed MessagePack RPC.
- `decentralized_semantica/dataset.py`: deterministic synthetic private shards
  and compact AOL bundle conversion.
- `decentralized_semantica/simulation.py`: local multi-process startup smoke
  simulation.
- `decentralized_semantica/centralized_simulation.py`: equivalent in-process
  tree, contact, and query simulation.
- `tests/test_decentralized_semantica.py`: focused unit and startup tests.

## Tests

```bash
python -m pip install -r requirements-decentralized-dev.txt
python -m pytest -q
```

The implementation is intentionally small. It binds to loopback, assumes
stable unique peer IDs and endpoints, and does not implement authentication,
Sybil resistance, custodian replication, churn recovery, or durable storage.
