from Construct_avg_embedding import construct_anon_embedding
from Estimate_baseline_prob import calculate_baseline
from Construct_tree_verified_version  import construct_tree
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # You can also try 'Qt5Agg' or 'Agg'

import numpy as np
import networkx as nx
warnings.filterwarnings("ignore")

nbr_train_recs_list = [20]
nbr_test_records_list = [10]

record_limit_per_leafnode_list = [50]
small_user_filter_for_test_set = [True]
consider_contacts_options = [False]
perform_query_options = [None] # OPTIONS: "Query-as-Node", "Closest-Users", "Closest-User-Chain"
plot_results = True

# N = df_avgemb.shape[0]  # number of closest candidates to compare
N_list = [50]
M_multipliers = [1]
DELTA_list = [5e-4]
expansion_rounds_list = [0]

def calculate_min_distances_to_golden_nodes(tree_anon_closest_users_df,user_docs_train,user_docs_test, small_user_filter):
    G = nx.DiGraph()
    for anon_id, row in tree_anon_closest_users_df.iterrows():
        neighbors = row['All_Shared']  # This should be a list or set of AnonIDs
        for nbr in neighbors:
            G.add_edge(anon_id, nbr)
    doc_owners = {}
    for idx, row in user_docs_train.iterrows():
        doc = row['DocIndex']
        owner = row['AnonID']
        if doc not in doc_owners:
            doc_owners[doc] = set()
        doc_owners[doc].add(owner)
    # We'll create a list to store minimum distances
    min_distances = []
    # Iterate through each row in user_docs_test
    for i, row in user_docs_test.iterrows():
        query_anon_id = row['AnonID']
        query_doc_idx = row['DocIndex']
        # If nobody in user_docs_train has this DocIndex, distance is None/NaN
        if query_doc_idx not in doc_owners:
            min_distances.append(np.nan)
            continue
        # Potential target users (those who have query_doc_idx)
        targets = doc_owners[query_doc_idx]
        # If the query AnonID doesn't exist in the graph, distance is None/NaN
        if query_anon_id not in G:
            min_distances.append(np.nan)
            raise Exception
            # continue
        # Use shortest_path_length from the source = query_anon_id
        # to find distances to all nodes. Then pick the minimum among the targets.
        distances = nx.shortest_path_length(G, source=query_anon_id)
        # Filter distances to only those in the target set
        target_distances = [distances[node] for node in distances if node in targets]
        if target_distances:
            min_distances.append(np.min(target_distances))
        else:
            # If we can't reach any target, store None/NaN
            min_distances.append(np.inf)
    # Add these distances to user_docs_test
    user_docs_test['min_distance'] = min_distances
    user_docs_test.to_pickle(os.path.join('temp',f'user_docs_test_filter{small_user_filter}.pkl'))



# nbr_train_recs_list = [10]
# nbr_test_records_list = [10]
#
# record_limit_per_leafnode_list = [5]
# # N = df_avgemb.shape[0]  # number of closest candidates to compare
# N_list = [5]
# DELTA_list = [0]


def main():
    number_total_experiments = (len(nbr_train_recs_list)*len(nbr_test_records_list)*
                                len(record_limit_per_leafnode_list)*len(N_list)*len(DELTA_list)*
                                len(M_multipliers)*
                                len(small_user_filter_for_test_set)*len(consider_contacts_options)*
                                len(perform_query_options) * len(expansion_rounds_list))
    print (f"STARTING {number_total_experiments} EXPERIMENTS")
    current_exp = 0
    #

    results = []
    for perform_query in perform_query_options:
        for small_user_filter in small_user_filter_for_test_set:
            for nbr_train_recs in nbr_train_recs_list:
                result = {}
                for nbr_test_records in nbr_test_records_list:
                    # Construct the anonymous embedding
                    df_avgemb, user_docs_train, user_docs_test, cosine_sim_df = construct_anon_embedding(
                        nbr_train_recs=nbr_train_recs,
                        nbr_test_records=nbr_test_records,
                        sample_rdn_test = True,
                        filter_out_users_with_few_docs = small_user_filter
                    )
                    df_avgemb = df_avgemb.sample(frac = 1, random_state = 613626)
                    user_docs_train = user_docs_train.sample(frac = 1, random_state = 523414)
                    user_docs_test = user_docs_test.sample(frac = 1, random_state = 876996)
                    cosine_sim_df = cosine_sim_df.sample(frac = 1, random_state = 857494)

                    if plot_results:
                        plt.plot(cosine_sim_df.max().sort_values(ascending=False).values)
                        plt.title('Ranked users by cosine-similarity with best buddy')
                        plt.xlabel('User')
                        plt.ylabel('Cosine-similarity')
                        plt.show()

                        from itertools import combinations
                        # Assuming 'user_docs_train' is your DataFrame with columns ['AnonID', 'DocIndex', 'QueryIndex']
                        def count_common_docindexes(df):
                            # Initialize a default dictionary to store counts of pairs
                            pair_counts = defaultdict(int)
                            # Group by 'DocIndex' to get all AnonIDs associated with each DocIndex
                            grouped = df.groupby('DocIndex')['AnonID']
                            for doc, anon_ids in grouped:
                                # Convert to a sorted set to ensure uniqueness and consistency in pair ordering
                                unique_anon_ids = sorted(set(anon_ids))
                                # Generate all unique pairs of AnonIDs for the current DocIndex
                                for pair in combinations(unique_anon_ids, 2):
                                    pair_counts[pair] += 1
                            return pair_counts

                        # Usage
                        pair_overlap_counts = count_common_docindexes(user_docs_train)
                        # If you want to convert the result to a DataFrame for easier handling:
                        overlap_df = pd.DataFrame(
                            [(anon1, anon2, count) for (anon1, anon2), count in pair_overlap_counts.items()],
                            columns=['AnonID_1', 'AnonID_2', 'Common_DocIndex_Count']
                        )
                        overlap_df = overlap_df.groupby('AnonID_1')['Common_DocIndex_Count'].max()
                        plt.plot(overlap_df.sort_values(ascending=False).values)
                        plt.title('Ranked users by co-occurrence of documents with best buddy')
                        plt.xlabel('User')
                        plt.ylabel('Co-occurrence')
                        plt.show()




                    result['nbr_train_recs'] = nbr_train_recs
                    result['nbr_test_records'] = nbr_test_records

                    result_ds = result.copy()

                    for expansion_rounds in expansion_rounds_list:

                        for record_limit_per_leafnode in record_limit_per_leafnode_list:
                            for DELTA in DELTA_list:
                                for N in N_list:
                                    N_filtered = min(N, df_avgemb.AnonID.nunique())
                                    for M_multiplier in M_multipliers:
                                        for consider_contacts in consider_contacts_options:
                                            current_exp += 1
                                            M = N_filtered * M_multiplier
                                            result = result_ds.copy()
                                            print ()
                                            print (f'Running algorithm with:'
                                                   f'EXPERIMENT NUMBER: {current_exp},'
                                                   f'expansion_rounds" {expansion_rounds},'
                                                   f'DELTA: {DELTA},'
                                                   f'N: {N_filtered},'
                                                   f'M: {M},'
                                                   f'Small_user_filter: {small_user_filter},'
                                                   f'random_baseline_useContacts: {consider_contacts},'
                                                   f'record_limit_per_leafnode: {record_limit_per_leafnode},'
                                                   f'perform_query_analysis: {perform_query},'
                                                   f'nbr_train_recs: {nbr_train_recs},'
                                                   f'nbr_test_recs: {nbr_test_records},')

                                            # Construct the tree and store the results
                                            (cmn_counts_mean, cmn_counts_median, nbr_users_contacted, tree_overlap,
                                             absolute_overlap, tree_anon_closest_users_df, querylevel_accuracy) = construct_tree(
                                                record_limit_per_leafnode=record_limit_per_leafnode,
                                                N=N_filtered,
                                                DELTA=DELTA,
                                                df_avgemb = df_avgemb,
                                                user_docs_train = user_docs_train,
                                                user_docs_test = user_docs_test,
                                                cosine_sims = cosine_sim_df,
                                                M = M,
                                                perform_query_test = perform_query,
                                                EXPANSION_ROUNDS=expansion_rounds,
                                                print_hists=plot_results
                                            )

                                            # calculate_min_distances_to_golden_nodes(tree_anon_closest_users_df,
                                            #                                         user_docs_train, user_docs_test,
                                            #                                         small_user_filter)

                                            result['DELTA'] = DELTA
                                            result['N'] = N_filtered
                                            result['M'] = M
                                            result['small_user_filter'] = small_user_filter
                                            result['baseline_consider_contacts'] = consider_contacts
                                            # result[]
                                            result['reclim_per_leafnode'] = record_limit_per_leafnode
                                            result['train_recs'] = nbr_train_recs
                                            result['test_recs'] = nbr_test_records
                                            result['cmn_counts_mean'] = cmn_counts_mean
                                            result['cmn_counts_median'] = cmn_counts_median
                                            result['nbr_users_contacted'] = nbr_users_contacted
                                            result['tree_overlap'] = tree_overlap
                                            result['absolute_overlap'] = absolute_overlap
                                            result['expansion_rounds'] = expansion_rounds
                                            result['querylevel_accuracy'] = querylevel_accuracy


                                            # Calculate the baseline and store the results
                                            mean_random, std_random = calculate_baseline(number_experiments = 1,
                                                                                         tree_anon_closest_users_df = tree_anon_closest_users_df,
                                                                                         user_docs_train = user_docs_train,
                                                                                         user_docs_test = user_docs_test,
                                                                                         df_avgemb = df_avgemb,
                                                                                         consider_contacts=consider_contacts)

                                            result['mean_random_baseline'] = mean_random
                                            result['std_random_baseline'] = std_random
                                            print(result)

                                            results.append(result)


    print (results)
    results = pd.DataFrame.from_dict(results)
    results.to_pickle(os.path.join('results', 'all_results_df.pkl'))
    print (1)

if __name__ == "__main__":
    main()
