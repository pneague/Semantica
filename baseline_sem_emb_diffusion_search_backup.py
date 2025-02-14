import os
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.sparse import eye
from Construct_avg_embedding import construct_anon_embedding

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csc_matrix
from multiprocessing import Pool, cpu_count  # <-- Add these
from functools import partial

graph_type = 'barabasi'  # or 'small_world'
barabasi_graph_m_value = 86

# Small-world constants
small_world_k = 10
small_world_p = 0.5

nbr_train_recs = 20
nbr_test_recs = 10
sample_rdn_test = True

filter_out_users_with_few_docs = False
num_docs_to_test = 500000

if graph_type == 'small_world':
    experiment_name = f'diffusionBaseline_{graph_type}_results_smallUserFilter{filter_out_users_with_few_docs}_p{small_world_p}_k{small_world_k}'
else:
    experiment_name = f'diffusionBaseline_{graph_type}_results_smallUserFilter{filter_out_users_with_few_docs}_m{barabasi_graph_m_value}'

logging.basicConfig(
    filename=f'{experiment_name}.log',  # <-- Change this to your preferred log filename/path
    level=logging.INFO,  # INFO level will record info, warnings, errors, etc.
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True
)

logging.info("Loading large data...")

df_avgemb, data, test_data, cosine_sim_df = construct_anon_embedding(
    nbr_train_recs=nbr_train_recs,
    nbr_test_records=nbr_test_recs,
    sample_rdn_test=sample_rdn_test,
    filter_out_users_with_few_docs=filter_out_users_with_few_docs
)
df_avgemb = df_avgemb.rename(columns={'embedding': 'AnonEmbedding'})
query_embs = pd.read_pickle(
    'embedded_queries_df_BERT.pkl'
).rename(columns={'embedding': 'QueryEmbedding'})

# Ensure each embedding is a np.array
df_avgemb['AnonEmbedding'] = df_avgemb['AnonEmbedding'].apply(lambda x: np.array(x))
query_embs['QueryEmbedding'] = query_embs['QueryEmbedding'].apply(lambda x: np.array(x))

# Build doc_to_users
doc_to_users = {}
for idx, row in data.iterrows():
    doc_id = row['DocIndex']
    user_id = row['AnonID']
    doc_to_users.setdefault(doc_id, set()).add(user_id)

# Build Barabási-Albert Graph
unique_users = df_avgemb['AnonID'].unique()
num_users = len(unique_users)

# Create the base graph
if graph_type == 'barabasi':
    G_temp = nx.barabasi_albert_graph(n=num_users, m=barabasi_graph_m_value)
elif graph_type == 'small_world':
    G_temp = nx.watts_strogatz_graph(n=num_users, k=small_world_k, p=small_world_p)
else:
    raise ValueError(f"Unknown graph_type: {graph_type}")

# Map integer -> actual user
idx_to_user = dict(enumerate(unique_users))
user_to_idx = {v: k for k, v in idx_to_user.items()}
G = nx.relabel_nodes(G_temp, idx_to_user)

# Build initial node_embeddings
node_embeddings = {}
for idx, row in df_avgemb.iterrows():
    node_embeddings[row['AnonID']] = row['AnonEmbedding']

if len(node_embeddings) > 0:
    zero_vec = np.zeros_like(next(iter(node_embeddings.values())))
else:
    zero_vec = np.zeros(768)

for user_id in G.nodes():
    if user_id not in node_embeddings:
        node_embeddings[user_id] = zero_vec

logging.info("Data loaded successfully!")


###############################################################################
#                         2) Define your functions                            #
###############################################################################
def personalized_pagerank(embeddings, adj_matrix, alpha, nodes, max_iter=50):
    num_nodes = adj_matrix.shape[0]
    initial_embeddings_matrix = np.array([embeddings[node] for node in nodes])
    embeddings_matrix = initial_embeddings_matrix.copy()

    P = adj_matrix.copy()
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0] = 1
    P = P.multiply(1 / row_sums)

    for _ in range(max_iter):
        embeddings_matrix = (1 - alpha) * P.dot(embeddings_matrix) + alpha * initial_embeddings_matrix

    return {nodes[i]: embeddings_matrix[i] for i in range(num_nodes)}


def fast_pagerank(orig_emb, A, alpha, node_list):
    """
    Computes the diffused embeddings according to:
      E = alpha * (I - (1-alpha) * P)^(-1) * E(0)
    where:
      - E(0) is the initial embedding matrix (one row per node)
      - A is the (sparse) adjacency matrix.
      - P is the row-normalized version of A.
      - alpha is the teleportation probability.
      - node_list is a list of nodes (in the same order as A).

    Parameters:
      orig_emb (dict): Mapping from node to its initial embedding (E(0)).
      A (scipy.sparse matrix): Adjacency matrix of the graph.
      alpha (float): Teleportation (restart) probability (between 0 and 1).
      node_list (list): List of nodes corresponding to rows/columns of A.

    Returns:
      dict: Mapping from node to its diffused embedding.
    """
    num_nodes = len(node_list)
    # Build initial embedding matrix E(0) as a (num_nodes x d) array
    E0 = np.array([orig_emb[node] for node in node_list])

    # Normalize A row-wise to form the transition matrix P.
    # (Make a copy to avoid modifying the original A.)
    A_norm = A.copy()
    row_sums = np.array(A_norm.sum(axis=1)).flatten()
    # Avoid division by zero in isolated nodes
    row_sums[row_sums == 0] = 1.0
    # Multiply each row by the reciprocal of its sum
    A_norm = A_norm.multiply(1 / row_sums[:, np.newaxis])

    # Create identity matrix I (sparse)
    I = eye(num_nodes, format='csr')

    # Form the matrix M = I - (1 - alpha) * A_norm
    M = I - (1 - alpha) * A_norm

    # For a small graph, convert M to a dense array and invert it.
    # (For large graphs, one should use a linear solver instead.)
    M_dense = M.toarray()
    M_inv = np.linalg.inv(M_dense)

    # Compute the diffused embedding matrix:
    # E = alpha * M_inv * E0
    E = alpha * M_inv.dot(E0)

    # Return the result as a dictionary mapping node to its embedding
    return {node_list[i]: E[i] for i in range(num_nodes)}

def execute_query(query_embedding, start_node, golden_nodes, G, ppr_embeddings, max_hops=50):
    visited = set()
    current_node = start_node

    for hop in range(max_hops):
        if current_node in golden_nodes:
            return True, hop
        visited.add(current_node)

        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            return False, hop

        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            unvisited = neighbors

        best_score = -1e9
        best_neighbor = None
        for nbr in unvisited:
            emb = ppr_embeddings[nbr]
            if np.isnan(emb).any():
                sim = -1e6
            else:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    emb.reshape(1, -1)
                )[0, 0]
            if sim > best_score:
                best_score = sim
                best_neighbor = nbr

        current_node = best_neighbor

    return False, max_hops


def run_experiment(alpha, number_docs_to_test):
    """
    Returns final_results for a given alpha
    """
    logging.info(f"[Process α={alpha}] Building adjacency matrix...")
    node_list = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, dtype=np.float32).tocsc()

    # Local copy so we don't mutate global embeddings
    orig_emb = {user: node_embeddings[user].copy() for user in node_list}

    logging.info(f"[Process α={alpha}] Running PPR diffusion...")
    ppr_embs = personalized_pagerank(orig_emb, A, alpha, node_list, max_iter=500)
    ppr_embs2 = fast_pagerank(orig_emb, A, alpha, node_list)

    distance_buckets = {}
    golden_min_distances = []
    # Sample if you only want a few records, or remove sample() to test everything
    test_subset = test_data.sample(n=min(number_docs_to_test, len(test_data)))

    for i, (_, row) in enumerate(test_subset.iterrows()):
        # Log every 1000 rows
        if i % 1000 == 0 and i > 0:
            logging.info(f"[Process α={alpha}] Processed {i} rows out of {len(test_subset)}")

        golden_doc = row['DocIndex']
        if golden_doc not in doc_to_users:
            dist_val = None
            golden_min_distances.append(dist_val)
            continue
        golden_nodes = doc_to_users[golden_doc]
        if not golden_nodes:
            continue

        qindex = row['QueryIndex']
        q_row = query_embs[query_embs['QueryIndex'] == qindex]
        if q_row.empty:
            continue
        query_embedding = q_row.iloc[0]['QueryEmbedding']

        start_node = row['AnonID']
        if start_node not in G:
            continue

        bfs_map = nx.single_source_shortest_path_length(G, start_node)
        distances_to_golden = [bfs_map[gn] for gn in golden_nodes if gn in bfs_map]
        dist_val = np.min(distances_to_golden) if distances_to_golden else 999999

        if dist_val == 1:
            print(1)

        hit, _ = execute_query(query_embedding, start_node, golden_nodes, G, ppr_embs, max_hops=50)
        distance_buckets.setdefault(dist_val, []).append(1 if hit else 0)
        golden_min_distances.append(dist_val)

    final_acc = {}
    max_plot_dist = 2000
    for dist_val, hits_list in distance_buckets.items():
        if dist_val > max_plot_dist:
            continue
        accuracy = np.mean(hits_list)
        final_acc[dist_val] = accuracy

    if len(distance_buckets) == 0:
        overall_accuracy = 0.0
    else:
        overall_accuracy = np.sum([sum(lst) for lst in distance_buckets.values()]) / \
                           np.sum([len(lst) for lst in distance_buckets.values()])

    return overall_accuracy, final_acc


###############################################################################
#            3) Main guard & running with multiprocessing Pool               #
###############################################################################

if __name__ == '__main__':
    # List of α values
    alpha_values = [0.9]

    final_results = {}

    # Use all available CPU cores
    n_cores = 1
    logging.info(f"Using {n_cores} processes...")

    # Prepare the arguments for each alpha in a list of tuples
    args_for_pool = [(alpha, num_docs_to_test) for alpha in alpha_values]

    results = run_experiment(alpha_values[0], num_docs_to_test)

    # # Create a pool of processes; by default, 'processes=n_cores'
    # with Pool(processes=n_cores) as pool:
    #     # starmap will feed each tuple in args_for_pool to run_experiment
    #     results = pool.starmap(run_experiment, args_for_pool)

    # results is a list of (overall_accuracy, acc_map) in the same order as alpha_values
    for alpha, (overall_accuracy, acc_map) in zip(alpha_values, results):
        final_results[alpha] = (overall_accuracy, acc_map)
        logging.info(f"[Process α={alpha}] Completed successfully with accuracy {overall_accuracy}.")

    # Save results
    with open(
            os.path.join(
                'results',
                f'diffusionBaseline_results_smallUserFilter{filter_out_users_with_few_docs}_m{barabasi_graph_m_value}.pkl'
            ),
            'wb'
    ) as file:
        pkl.dump(final_results, file)

    # Plot
    plt.figure(figsize=(10, 6))
    max_dist_to_plot = 1000

    for alpha in alpha_values:
        if alpha not in final_results:
            continue
        overall_accuracy, acc_map = final_results[alpha]
        distances_to_plot = sorted(d for d in acc_map.keys() if d <= max_dist_to_plot)
        accuracies = [acc_map[d] for d in distances_to_plot]

        plt.plot(distances_to_plot, accuracies, marker='o', label=f'α={alpha}')

    plt.xlabel('Distance from start_node to golden node(s)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. distance (sample_rdn = {sample_rdn_test})')
    plt.legend()
    plt.show()
