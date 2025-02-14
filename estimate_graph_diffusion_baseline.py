
from scipy.sparse import csr_matrix, eye
import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pkl

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csc_matrix
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# Custom data loading code (as per your original script):
from Construct_avg_embedding import construct_anon_embedding

###############################################################################
#                          0) Global references                               
###############################################################################
# We'll store doc_to_users, query_embs, G, ppr_embs, etc. in module-level 
# globals, so each worker loads them just once (read-only).
doc_to_users_glob = None
query_embs_glob   = None
G_glob            = None
ppr_embs_glob     = None
max_hops_glob     = None
counter_glob      = None
lock_glob         = None

###############################################################################
#                 A) Worker initializer to share big read-only data           
###############################################################################
def init_pool(
    doc_to_users, 
    query_embs, 
    G, 
    ppr_embs, 
    max_hops, 
    counter, 
    lock
):
    """
    This initializer is run *once* per worker process. We copy references to
    large read-only data structures into global variables so we don't have
    to pass them repeatedly (which can bloat RAM usage).
    """
    global doc_to_users_glob, query_embs_glob, G_glob, ppr_embs_glob
    global max_hops_glob, counter_glob, lock_glob

    doc_to_users_glob = doc_to_users
    query_embs_glob   = query_embs
    G_glob            = G
    ppr_embs_glob     = ppr_embs
    max_hops_glob     = max_hops
    counter_glob      = counter
    lock_glob         = lock

###############################################################################
#            B) Worker function referencing the global variables              
###############################################################################
def process_single_query(row):
    """
    Handles one test row/query. 
    Increments a shared counter after finishing, 
    and logs progress every 100 queries.
    """
    global doc_to_users_glob, query_embs_glob, G_glob, ppr_embs_glob
    global max_hops_glob, counter_glob, lock_glob

    doc_to_users = doc_to_users_glob
    query_embs   = query_embs_glob
    G            = G_glob
    ppr_embs     = ppr_embs_glob
    max_hops     = max_hops_glob
    counter      = counter_glob
    lock         = lock_glob

    golden_doc = row['DocIndex']
    if golden_doc not in doc_to_users:
        return None

    golden_nodes = doc_to_users[golden_doc]
    if not golden_nodes:
        return None

    qindex = row['QueryIndex']
    q_row = query_embs[query_embs['QueryIndex'] == qindex]
    if q_row.empty:
        return None

    query_embedding = q_row.iloc[0]['QueryEmbedding']
    start_node = row['AnonID']
    if start_node not in G:
        return None

    # BFS distance
    bfs_map = nx.single_source_shortest_path_length(G, start_node)
    distances_to_golden = [bfs_map[gn] for gn in golden_nodes if gn in bfs_map]
    dist_val = min(distances_to_golden) if distances_to_golden else 999999

    # Execute the walk
    hit, _ = execute_query(
        query_embedding=query_embedding,
        start_node=start_node,
        golden_nodes=golden_nodes,
        G=G,
        ppr_embeddings=ppr_embs,
        max_hops=max_hops
    )

    # Increment the shared counter and log every 100 queries
    with lock:
        counter.value += 1
        if counter.value % 100 == 0:
            pid = os.getpid()
            logging.info(f"[PID={pid}] Processed {counter.value} queries so far...")

    return (dist_val, 1 if hit else 0)

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

def personalized_pagerank_past(embeddings, adj_matrix, alpha, nodes, max_iter=1000):
    """
    Simple personalized PageRank/diffusion on a set of node embeddings.
    This function remains largely unchanged from your original code.
    """
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

def personalized_pagerank(orig_emb, A, alpha, node_list, max_iter=100):
    """
    Iterative diffusion of node embeddings using the update:
      X^(t+1) = (1 - alpha) * P * X^(t) + alpha * X^(0)
    where P is the row-normalized version of A.
    """
    # Build initial embeddings matrix from dictionary (shape: [num_nodes x embedding_dim])
    initial_embeddings_matrix = np.array([orig_emb[node] for node in node_list])
    embeddings_matrix = initial_embeddings_matrix.copy()
    # Normalize A row-wise: get a stochastic matrix P
    row_sums = np.array(A.sum(axis=1)).flatten()
    # Replace zeros to avoid division by zero
    row_sums[row_sums == 0] = 1.0
    P = A.multiply(1 / row_sums[:, np.newaxis])
    # Run the iterative update for max_iter iterations
    for _ in range(max_iter):
        embeddings_matrix = (1 - alpha) * P.dot(embeddings_matrix) + alpha * initial_embeddings_matrix
    # Return a dictionary mapping each node to its resulting embedding vector
    result = {node_list[i]: embeddings_matrix[i] for i in range(len(node_list))}
    return result


def fast_pagerank(orig_emb, A, alpha, node_list):
    """
    Direct method for computing the diffused embedding:
      X = alpha * (I - (1 - alpha) * P)^{-1} * X^(0)
    where P is the row-normalized version of A.
    """
    num_nodes = len(node_list)
    # Build the initial embeddings matrix from dictionary
    E0 = np.array([orig_emb[node] for node in node_list])
    # Normalize A row-wise to obtain P
    A_norm = A.copy()
    row_sums = np.array(A_norm.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = A_norm.multiply(1 / row_sums[:, np.newaxis])
    # Build (I - (1-alpha)*P) and invert it (convert to dense for inversion)
    I = eye(num_nodes, format='csr')
    M = I - (1 - alpha) * A_norm
    M_dense = M.toarray()
    M_inv = np.linalg.inv(M_dense)
    # Compute the closed-form solution for the diffused embeddings
    E = alpha * M_inv.dot(E0)
    result = {node_list[i]: E[i] for i in range(num_nodes)}
    return result



###############################################################################
#                    C) Main experiment code, using Pool(init_pool)           
###############################################################################
def run_experiment(
    max_hops,
    number_docs_to_test,
    alpha,
    doc_to_users,
    query_embs,
    test_data,
    node_embeddings,
    G,
    counter,
    lock
):
    """
    Runs PPR once (for alpha=0.9, say), then processes each query in parallel.
    Also uses the shared counter and lock for logging progress.
    """
    logging.info(f"[max_hops={max_hops}] Building adjacency matrix...")
    node_list = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, dtype=np.float32).tocsc()

    # Local copy so we don't mutate global embeddings
    orig_emb = {user: node_embeddings[user].copy() for user in node_list}

    logging.info(f"[max_hops={max_hops}] Running PPR diffusion (alpha={alpha})...")
    ppr_embs = personalized_pagerank(orig_emb, A, alpha, node_list)
    ppr_embs2 = fast_pagerank(orig_emb, A, alpha, node_list)


    # We'll sample from test_data
    test_subset = test_data.sample(n=min(number_docs_to_test, len(test_data))).copy()
    rows_for_pool = [row for _, row in test_subset.iterrows()]

    logging.info(f"[max_hops={max_hops}] Starting parallel query processing...")
    n_cores = cpu_count()
    logging.info(f"[max_hops={max_hops}] Using {n_cores} CPU cores.")

    # Instead of partial(...) with big objects, we use initializer=init_pool
    with Pool(
        processes=n_cores,
        initializer=init_pool,
        initargs=(doc_to_users, query_embs, G, ppr_embs, max_hops, counter, lock)
    ) as pool:
        results = pool.map(process_single_query, rows_for_pool, chunksize=500)

    # Aggregate results
    distance_buckets = {}
    for res in results:
        if res is None:
            continue
        dist_val, hit_flag = res
        distance_buckets.setdefault(dist_val, []).append(hit_flag)

    # Summarize results
    if not distance_buckets:
        overall_accuracy = 0.0
        final_acc = {}
    else:
        total_hits = sum(sum(hits_list) for hits_list in distance_buckets.values())
        total_count = sum(len(hits_list) for hits_list in distance_buckets.values())
        overall_accuracy = total_hits / total_count

        max_plot_dist = 2000
        final_acc = {}
        for dist_val, hits_list in distance_buckets.items():
            if dist_val <= max_plot_dist:
                final_acc[dist_val] = np.mean(hits_list)

    return overall_accuracy, final_acc

###############################################################################
#                         D) Running the entire script                         
###############################################################################
if __name__ == '__main__':
    # Example parameter
    m = 104

    # Configure logging
    logging.basicConfig(
        filename=f'graphDiffusion_baseline_m{m}_alpha01_n600.log',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        force=True
    )

    logging.info("Loading data...")

    # 1) Load your data frames
    df_avgemb, data, test_data, _ = construct_anon_embedding(
        nbr_train_recs=20,
        nbr_test_records=10,
        sample_rdn_test=True,
        filter_out_users_with_few_docs=True
    )
    logging.info(f'{data.shape}, {test_data.shape}, {df_avgemb.shape}')
    logging.info(f"Performing analysis of {test_data.shape[0]} queries")

    # 2) Rename columns and load query embeddings
    df_avgemb = df_avgemb.rename(columns={'embedding': 'AnonEmbedding'})
    query_embs = pd.read_pickle('embedded_queries_df_BERT.pkl')
    query_embs = query_embs.rename(columns={'embedding': 'QueryEmbedding'})

    # 3) Convert to NumPy arrays
    df_avgemb['AnonEmbedding'] = df_avgemb['AnonEmbedding'].apply(lambda x: np.array(x))
    query_embs['QueryEmbedding'] = query_embs['QueryEmbedding'].apply(lambda x: np.array(x))

    # 4) Build doc_to_users mapping
    doc_to_users = {}
    for idx, row in data.iterrows():
        doc_id = row['DocIndex']
        user_id = row['AnonID']
        doc_to_users.setdefault(doc_id, set()).add(user_id)

    # 5) Build a graph
    unique_users = df_avgemb['AnonID'].unique()
    G_temp = nx.barabasi_albert_graph(n=len(unique_users), m=m)
    idx_to_user = dict(enumerate(unique_users))
    G = nx.relabel_nodes(G_temp, idx_to_user)

    # 6) Build node embeddings map
    node_embeddings = {}
    for idx, row in df_avgemb.iterrows():
        node_embeddings[row['AnonID']] = row['AnonEmbedding']

    # 7) Fill any missing embeddings with zero vectors
    if node_embeddings:
        zero_vec = np.zeros_like(next(iter(node_embeddings.values())))
    else:
        zero_vec = np.zeros(768)  # Fallback if empty
    for user_id in G.nodes():
        if user_id not in node_embeddings:
            node_embeddings[user_id] = zero_vec

    logging.info("Data loaded!")





    ######################## Experiment parameters
    alpha = 0.9
    max_hops_list = [600]
    num_docs_to_test = 500000
    ######################## Experiment parameters





    # Create a Manager for the shared counter and lock
    manager = Manager()
    counter = manager.Value('i', 0)  # shared integer counter
    lock = manager.Lock()            # shared lock

    final_results = {}
    for mh in max_hops_list:
        # Reset the counter for each experiment (optional)
        counter.value = 0
        logging.info(f"=== Starting experiment for max_hops={mh} ===")

        overall_accuracy, acc_map = run_experiment(
            max_hops=mh,
            number_docs_to_test=num_docs_to_test,
            alpha=alpha,
            doc_to_users=doc_to_users,
            query_embs=query_embs,
            test_data=test_data,
            node_embeddings=node_embeddings,
            G=G,
            counter=counter,
            lock=lock
        )
        final_results[mh] = (overall_accuracy, acc_map)
        logging.info(f"=== Finished max_hops={mh}, Accuracy={overall_accuracy:.4f} ===")

    # Save results for future reference
    with open(f"graphDiffusion_baseline_m{m}_alpha{str(alpha).replace('.','')}_n600_test.pkl", 'wb') as f:
        pkl.dump(final_results, f)

    logging.info("All done.")
