
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import dill


# def compute_distance(n1, n2):
#     path1 = get_path_to_root(n1)  # From n1 up to root
#     path2 = get_path_to_root(n2)  # From n2 up to root
#
#     # Convert path2 to a set for quick membership checks
#     ancestors_of_n2 = set(path2)
#
#     # Find the LCA: the first node in n1's path that is also in n2's path
#     LCA = None
#     for ancestor in path1:
#         if ancestor in ancestors_of_n2:
#             LCA = ancestor
#             break
#
#     # Distance from n1 to LCA
#     dist_n1_to_LCA = path1.index(LCA)
#     # Distance from n2 to LCA
#     dist_n2_to_LCA = path2.index(LCA)
#
#     # Total distance is sum of both distances
#     return dist_n1_to_LCA + dist_n2_to_LCA


def evaluate_routing_chain_for_chunk(
        chunk_df,
        anonid_to_embedding,
        user_docs_dict,
        tree_anon_closest_users_df,
        max_hops=50
):
    """
    Run chain-based routing for a chunk of queries (subset of user_docs_test).
    Returns a list of dicts: [{"AnonID": ..., "DocIndex": ..., "found": ...}, ...]
    """
    pid = os.getpid()
    chunk_size = len(chunk_df)
    log_every = 100
    results = []

    for i, row in enumerate(chunk_df.itertuples(index=False)):
        sender_id = row.AnonID
        query_emb = np.array(row.query_embedding)
        target_doc = row.DocIndex

        found_flag = False
        hops = 0
        visited = set([sender_id])
        current_user = sender_id

        while hops < max_hops:
            hops += 1

            # Get the list of accessible users for current_user
            if current_user not in tree_anon_closest_users_df.index:
                break
            contacts = tree_anon_closest_users_df.loc[current_user, "All_Shared"]
            if not contacts:
                break

            # Among these contacts, pick the single contact with highest similarity
            best_contact = None
            best_sim = -1.0
            for c in contacts:
                if c not in anonid_to_embedding or c in visited:
                    continue
                sim = cosine_similarity(
                    query_emb.reshape(1, -1),
                    anonid_to_embedding[c].reshape(1, -1)
                )[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_contact = c

            if best_contact is None:
                break

            # Check if best_contact has the doc
            if best_contact in user_docs_dict and target_doc in user_docs_dict[best_contact]:
                found_flag = True
                break
            else:
                visited.add(best_contact)
                current_user = best_contact

        results.append({
            "AnonID": sender_id,
            "DocIndex": target_doc,
            "found": found_flag
        })
        # Log progress for this subprocess every 100 queries (adjust as desired)
        if i % log_every == 0:
            with open("chain_progress.log", "a") as f:
                f.write(
                    f"[PID={pid}] Processed {i+1}/{chunk_size} queries in this chunk.\n"
                )
    # One final write at the end of the chunk
    with open("chain_progress.log", "a") as f:
        f.write(
            f"[PID={pid}] Completed all {chunk_size} queries in this chunk.\n"
        )

    return results

def parallel_evaluate_routing_chain(
        df_avgemb,
        user_train_docs,
        user_docs_test,
        tree_anon_closest_users_df,
        top_k=50
):
    """
    Parallel version of the chain-based routing:
    1) Build lookup structures.
    2) Partition user_docs_test.
    3) Use multiple processes to handle each partition.
    """
    # 1) Build dictionaries for quick lookups
    anonid_to_embedding = {}
    for row in df_avgemb.itertuples(index=False):
        anonid_to_embedding[row.AnonID] = np.array(row.embedding)

    user_docs_dict = {}
    for row in user_train_docs.itertuples(index=False):
        uid = row.AnonID
        doc_index = row.DocIndex
        if uid not in user_docs_dict:
            user_docs_dict[uid] = set()
        user_docs_dict[uid].add(doc_index)

    # 2) Partition user_docs_test
    #    Let's define how many chunks we'll use. Could be cpu_count() or a smaller number.
    n_cores = max(1, cpu_count() - 1)  # or something like min(8, cpu_count()) if memory is a concern
    chunks = np.array_split(user_docs_test, n_cores)

    # 3) For each chunk, process in parallel
    with Pool(n_cores) as pool:
        # We'll use a partial function or pass everything as arguments
        results_lists = pool.starmap(
            evaluate_routing_chain_for_chunk,
            [
                (chunk, anonid_to_embedding, user_docs_dict, tree_anon_closest_users_df, 50)
                for chunk in chunks
            ]
        )

    # Flatten the list of lists into one big list
    all_results = [res for sublist in results_lists for res in sublist]

    # Build final DataFrame and compute accuracy
    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()

    return results_df, accuracy


def evaluate_routing_for_chunk(
    chunk_df,
    anonid_to_embedding,
    user_docs_dict,
    tree_anon_closest_users_df,
    top_k=50
):
    """
    Process a subset (chunk) of `user_docs_test`.
    Returns a list of dicts: [{"AnonID": ..., "DocIndex": ..., "found": ...}, ...]
    """
    from sklearn.metrics.pairwise import cosine_similarity  # Import locally if needed
    import numpy as np

    pid = os.getpid()
    chunk_size = len(chunk_df)
    log_every = 100
    results = []

    for i, row in enumerate(chunk_df.itertuples(index=False)):
        sender_id = row.AnonID
        query_emb = np.array(row.query_embedding)
        target_doc = row.DocIndex

        # Get the list of accessible users for this sender
        # (assuming "All_Shared" is already a list of AnonIDs)
        if sender_id not in tree_anon_closest_users_df.index:
            # If the sender_id is missing in the DF index, skip
            results.append({
                "AnonID": sender_id,
                "DocIndex": target_doc,
                "found": False
            })
            continue

        accessible_users = tree_anon_closest_users_df.loc[sender_id, "All_Shared"]

        # Compute cosine similarities for these accessible users
        similarities = []
        for user_id in accessible_users:
            # If user_id is missing from df_avgemb, skip
            if user_id not in anonid_to_embedding:
                continue

            user_emb = anonid_to_embedding[user_id].reshape(1, -1)
            sim = cosine_similarity(query_emb.reshape(1, -1), user_emb)[0][0]
            similarities.append((user_id, sim))

        # Sort by similarity (descending), take top_k
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        top_user_ids = [u[0] for u in top_similar]

        # Check if target_doc is among those top_user_ids
        found_flag = False
        for u in top_user_ids:
            if u in user_docs_dict and target_doc in user_docs_dict[u]:
                found_flag = True
                break

        results.append({
            "AnonID": sender_id,
            "DocIndex": target_doc,
            "found": found_flag
        })        # Log progress for this subprocess every 100 queries (adjust as desired)
        if i % log_every == 0:
            with open("non_chain_progress.log", "a") as f:
                f.write(
                    f"[PID={pid}] Processed {i}/{chunk_size} queries in this chunk.\n"
                )
    # One final write at the end
    with open("non_chain_progress.log", "a") as f:
        f.write(
            f"[PID={pid}] Completed all {chunk_size} queries in this chunk.\n"
        )
    return results
def parallel_evaluate_routing(
    df_avgemb,
    user_train_docs,
    user_docs_test,
    tree_anon_closest_users_df,
    top_k=50,
    n_cores= max(1, cpu_count() - 1)
):
    """
    Parallel version of the 'evaluate_routing' function:
      - Splits user_docs_test into multiple chunks.
      - Each chunk is processed by `evaluate_routing_for_chunk`.
      - Aggregates results and computes accuracy.
    """

    import numpy as np
    import pandas as pd
    from multiprocessing import Pool
    from collections import defaultdict

    # 1) Build dictionaries for quick lookups
    anonid_to_embedding = {}
    for row in df_avgemb.itertuples(index=False):
        anonid_to_embedding[row.AnonID] = row.embedding  # or np.array(row.embedding)

    user_docs_dict = defaultdict(set)
    for row in user_train_docs.itertuples(index=False):
        user_docs_dict[row.AnonID].add(row.DocIndex)

    # 2) Partition user_docs_test
    chunks = np.array_split(user_docs_test, n_cores)

    # 3) For each chunk, process in parallel
    with Pool(n_cores) as pool:
        # We'll starmap the helper function
        results_lists = pool.starmap(
            evaluate_routing_for_chunk,
            [
                (chunk, anonid_to_embedding, user_docs_dict, tree_anon_closest_users_df, top_k)
                for chunk in chunks
            ]
        )

    # Flatten the list of lists into one big list
    all_results = []
    for res_list in results_lists:
        all_results.extend(res_list)

    # 4) Build final DataFrame and compute accuracy
    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()

    return results_df, accuracy

def calculate_queryAsNode_for_chunk(
    chunk_df,
    path_to_tree_pickle,
    user_train_docs_dict,
    M,
    N,
    log_filename="querynode_progress.log"
):
    """
    Processes a subset (chunk) of `user_docs_test`, loading the tree from pickle.
    Returns a list of dicts: [{"AnonID": ..., "DocIndex": ..., "QueryFound": ...}, ...]
    """
    import os
    import pickle

    pid = os.getpid()
    chunk_size = len(chunk_df)
    results = []
    log_every = 100  # how often to log progress

    # 1) Load the tree from disk (not from memory!)
    with open(path_to_tree_pickle, "rb") as f:
        tree = dill.load(f)

    # 2) Process each row
    for i, row in enumerate(chunk_df.itertuples(index=False)):
        sender_id = row.AnonID
        query_emb = row.query_embedding
        target_doc_index = row.DocIndex

        # Get the closest users
        closest_users = tree.get_closest_users_for_query(query_emb, M=M, N=N)

        # Union all their documents
        documents_from_these_users = set()
        for anon_id in closest_users:
            documents_from_these_users |= user_train_docs_dict.get(anon_id, set())

        # Check if target_doc_index is found
        found_flag = (target_doc_index in documents_from_these_users)
        results.append({
            "AnonID": sender_id,
            "DocIndex": target_doc_index,
            "QueryFound": found_flag
        })

        # Log progress every 'log_every' records
        if i % log_every == 0:
            with open(log_filename, "a") as lf:
                lf.write(f"[PID={pid}] Processed {i}/{chunk_size} queries in this chunk.\n")

    # One final log entry
    with open(log_filename, "a") as lf:
        lf.write(f"[PID={pid}] Completed all {chunk_size} queries in this chunk.\n")

    return results

def parallel_calculate_queryAsNode_overlap_percentage(
    path_to_tree_pickle,
    user_train_docs,
    user_docs_test,
    M,
    N,
    n_cores= max(1, cpu_count() - 1),
    log_filename="querynode_progress.log"
):
    """
    Parallel version of the 'calculate_queryAsNode_overlap_percentage' logic,
    but *without* passing the tree object directly. Instead, each worker loads
    the tree from a pickled file on disk.

    Parameters
    ----------
    path_to_tree_pickle : str
        Path to the pickle file where the tree is saved.
    user_train_docs : pd.DataFrame
        DataFrame with columns ["AnonID", "DocIndex"].
    user_docs_test : pd.DataFrame
        DataFrame with columns ["AnonID", "query_embedding", "DocIndex"].
    M, N : int
        Arguments for `tree.get_closest_users_for_query`.
    n_cores : int
        Number of processes to use in the pool.
    log_filename : str
        Name of the log file for progress messages.

    Returns
    -------
    results_df : pd.DataFrame
        Has columns ["AnonID", "DocIndex", "QueryFound"].
    accuracy : float
        Fraction of queries for which the target doc was found.
    """

    import numpy as np
    import pandas as pd
    from multiprocessing import Pool
    from collections import defaultdict


    # 1) Build a dictionary for quick doc lookups: {AnonID -> set(DocIndexes)}
    user_train_docs_dict = defaultdict(set)
    for row in user_train_docs.itertuples(index=False):
        user_train_docs_dict[row.AnonID].add(row.DocIndex)

    # 2) Split user_docs_test into chunks
    chunks = np.array_split(user_docs_test, n_cores)

    # 3) Create a pool of subprocesses
    with Pool(n_cores) as pool:
        # We'll starmap over our chunk-based function
        # Note: we pass only the chunk, the path, and the minimal needed data
        results_lists = pool.starmap(
            calculate_queryAsNode_for_chunk,
            [
                (chunk, path_to_tree_pickle, user_train_docs_dict, M, N, log_filename)
                for chunk in chunks
            ]
        )

    # 4) Flatten the results from all chunks
    all_results = []
    for res_list in results_lists:
        all_results.extend(res_list)

    # 5) Convert to DataFrame and compute accuracy
    results_df = pd.DataFrame(all_results)  # columns: [AnonID, DocIndex, QueryFound]
    accuracy = results_df["QueryFound"].mean()  # True == 1, False == 0 in Python

    return results_df, accuracy
