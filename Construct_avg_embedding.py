import pandas as pd
import os
from transformers import T5Tokenizer, T5EncoderModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import random



def construct_anon_embedding(nbr_train_recs, nbr_test_records = 10, sample_rdn_test = True,
                             filter_out_users_with_few_docs = False):
    # Set seed for reproducibility

    SEED = 42
    # number_closest_neighbors = 30
    # GETTING SUBSET OF USERS HERE
    # low_user_query_lim = 10
    # high_user_query_lim = 1200000
    # nbr_train_recs = 10
    # nbr_test_records = 10

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    data = pd.read_csv(os.path.join('dataset','data.csv'), sep='\t')
    doc = pd.read_csv(os.path.join('dataset','doc.csv'), sep='\t')
    doc = doc[(~doc['Title'].str.startswith('www'))]
    doc = doc[(~doc['Title'].str.contains('404|403|401 authorization|401 unauthorized|406 blocked|406 error|error 500|error 406|error|access denied|502 bad gateway'))]
    data = data[data['DocIndex'].isin(doc['DocIndex'])]
    query = pd.read_csv(os.path.join('dataset','query.csv'), sep='\t')


    # Split the 'CandiList' column by '\' and expand into separate columns
    split_columns = data['CandiList'].str.split('\t', expand=True)

    # Rename the new columns as 'Candi1', 'Candi2', etc.
    split_columns.columns = [f'Candi{i+1}' for i in range(split_columns.shape[1])]

    # Concatenate the original DataFrame with the new columns
    data = pd.concat([data, split_columns], axis=1)
    data.drop(columns = 'CandiList', inplace = True)

    # users = data.groupby('AnonID')['QueryIndex'].count().sort_values(ascending = True)


    # data_subset = data[data['AnonID'].isin(users_subset.index)]

    # docs_subset_list = data_subset.groupby('DocIndex')['QueryIndex'].count()
    # print (docs_subset_list.shape)

    # docs_subset = doc[doc['DocIndex'].isin(docs_subset_list.index)]
    # docs_under_50[docs_under_50>1]

    docs_subset = pd.read_pickle('embedded_docs_BERT_df.pkl')

    # TODO: CHECK WHETHER TAKING OUT QUERYTIME CHANGES PERFORMANCE
    data = data.sort_values('QueryTime',ascending = True).groupby(
        ['AnonID', 'DocIndex'], as_index = False).first()[['AnonID', 'DocIndex', 'QueryIndex']]

    # users_subset = users[(users>low_user_query_lim) & (users<high_user_query_lim)]
    # print (users_subset.shape)
    # data = data[data['AnonID'].isin(users_subset.index)]
    # data = data.groupby('AnonID', group_keys=False).apply(
    #     lambda x: x.sample(min(len(x), 5))
    # )

    # Apply the sampling and selecting logic
    def sample_and_select(group, nbr_train_recs = nbr_train_recs, nbr_test_records = nbr_test_records, sample_rdn_test = True):

        # if group['AnonID'].iloc[0] in [4462390, 4600674]:
        #     print (1)
        if len(group) <= nbr_train_recs + nbr_test_records:
            return None  # If the group has 5 or fewer records, return all records
        else:
            if sample_rdn_test:
                sample_idx = random.sample(group.index.tolist(), nbr_test_records)
                return group.loc[sample_idx]
            else:
                # Define the range for sampling
                max_start = len(group) - nbr_train_recs - nbr_test_records
                start_idx = np.random.randint(0, max_start + 1)  # Randomly sample a start index
                # start_idx = 0
                return group.iloc[start_idx: start_idx + nbr_train_recs + nbr_test_records]  # Take 5 consecutive rows starting from the sampled index

    if sample_rdn_test:
        test_data = data.groupby('AnonID', group_keys = False).apply(
            lambda group: sample_and_select(group, nbr_train_recs = nbr_train_recs, nbr_test_records = nbr_test_records, sample_rdn_test = sample_rdn_test)).dropna()
        # Reset the index to get the original indices
        test_data = test_data.reset_index(drop=False)

        # Drop the sampled rows from the original data
        if filter_out_users_with_few_docs: # TO KEEP OR NOT TO KEEP SMALL USERS AS NODES (excluded from test set)
            data = data[data['AnonID'].isin(test_data['AnonID'].unique())]
        data = data.drop(test_data['index'])


    else:
        data = data.groupby('AnonID', group_keys=False).apply(
            lambda group: sample_and_select(group, nbr_train_recs, sample_rdn_test = sample_rdn_test)).dropna()
        test_data = data.groupby('AnonID').apply(
            lambda x: x.iloc[nbr_train_recs:nbr_train_recs + nbr_test_records]).reset_index(drop=True)
        data = data.groupby('AnonID').head(nbr_train_recs)

    nbr_users = data['AnonID'].nunique()

    print(f'Number of users for {nbr_train_recs} train records and {nbr_test_records} test records: {nbr_users}')

    df = pd.merge(data, docs_subset[['DocIndex','embedding']], on = 'DocIndex', how = 'left')

    df_avgemb = df.groupby('AnonID')['embedding'].mean().reset_index()

    df_avgemb.to_pickle(os.path.join('results', 'average_embedding_small.pkl'))
    data.to_pickle(os.path.join('results', 'traindata.pkl'))
    test_data.to_pickle(os.path.join('results','testdata.pkl'))

    ####################### CONSTRUCTING HIST ##########################
    # Extract user IDs and embeddings
    user_ids = df_avgemb['AnonID'].tolist()
    embeddings = np.array(df_avgemb['embedding'].tolist())

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embeddings)

    num_users = cosine_sim_matrix.shape[0]
    # Exclude self-similarities (diagonal) by setting them to NaN
    np.fill_diagonal(cosine_sim_matrix, np.nan)

    # Convert to DataFrame for easier manipulation
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_ids, columns=user_ids)
    cosine_sim_df.to_pickle(os.path.join('results', 'cosine_sims.pkl'))


    # Sort similarities for each user from high to low
    sorted_similarities = np.sort(cosine_sim_matrix, axis=1)[:, ::-1]

    # Create a DataFrame where each column represents the rank (1st, 2nd, etc.)
    ranked_similarities = pd.DataFrame(sorted_similarities, index=user_ids)

    # For each user, find the highest similarity with other users
    closest_user_cosine = []
    for i, row in enumerate(cosine_sim_matrix):
        # Exclude self-similarity by setting diagonal element to -1
        row[i] = -1
        # Find the maximum similarity value
        max_sim = np.max(row)
        closest_user_cosine.append(max_sim)

    # Add the closest_user_cosine column to the DataFrame
    df_avgemb['closest_user_cosine'] = closest_user_cosine

    def calculate_baseline_recall_of_topN(num_rounds):
        import pandas as pd
        import random
        import numpy as np

        nbr_cons = pd.read_csv(os.path.join('temp','nbr_connections_per_user_delta001.csv'))
        # ------------------------------------------------------------
        # 1) LOAD OR HAVE YOUR COSINE SIMILARITY MATRIX (6900x6900)
        #    Suppose it's called cosine_sims_df (pandas DataFrame)
        # ------------------------------------------------------------
        # Example:
        # cosine_sims_df = pd.read_csv('my_cosine_sims.csv', index_col=0)
        # Make sure the DataFrame is 6900 x 6900.

        # Convert to a NumPy array for faster operations
        cosine_sim_matrix = cosine_sim_df.values
        N = cosine_sim_matrix.shape[0]

        # ------------------------------------------------------------
        # 2) FOR EACH USER, FIND THE TOP 50 MOST SIMILAR USERS
        #    AND STORE THEIR SIMILARITY THRESHOLD
        # ------------------------------------------------------------
        top_50_indices = [None] * N  # top_50_indices[i] will be a set of the user IDs
        similarity_thresholds = np.zeros(N)

        for i in range(N):
            # Copy out the row so we can modify the self-sim
            row = cosine_sim_matrix[i].copy()
            # Exclude self by making it very low (so it won't appear in top 50)
            row[i] = -1

            # Get indices of the top 50 similarities
            top_50_for_i = np.argsort(row)[::-1][:50]  # descending sort, take first 50
            top_50_indices[i] = set(top_50_for_i)

            # The 50th best similarity (the smallest among the top 50)
            similarity_thresholds[i] = row[top_50_for_i[-1]]

        # ------------------------------------------------------------
        # 3) INITIALIZE CONNECTIONS FOR EACH USER (208 RANDOM EACH)
        # ------------------------------------------------------------
        connections = [set() for _ in range(N)]
        all_users = np.arange(N)

        for i in range(N):
            x = int(nbr_cons.iloc[i]['0'])
            # print (x)
            # pick 208 distinct users (excluding i)
            possible = np.delete(all_users, i)  # or set(range(N)) - {i}
            chosen = np.random.choice(possible, size=x, replace=False)
            connections[i] = set(chosen)

        # ------------------------------------------------------------
        # 4) RUN 20 ROUNDS OF THE "ASK FRIENDS FOR BETTER CONTACTS"
        # ------------------------------------------------------------

        for round_idx in range(num_rounds):
            # Shuffle the user order if you want randomness in iteration
            user_order = np.random.permutation(N)
            for i in user_order:
                if len(connections[i]) == 0:
                    continue

                # Pick one random friend j of i
                j = random.sample(sorted(connections[i]), 1)[0]

                # Check j's connections to see if there's someone k
                # with similarity > i's 50th best known similarity
                thresh_i = similarity_thresholds[i]
                for k in connections[j]:
                    # Skip if k == i
                    if k == i:
                        continue

                    # If similarity to i is above i's 50th best global threshold, add to i's connections
                    if cosine_sim_matrix[i, k] > thresh_i:
                        connections[i].add(k)

        # ------------------------------------------------------------
        # 5) EVALUATE HOW MANY OF THE TRUE TOP-50 WE FOUND
        # ------------------------------------------------------------
        hits_per_user = np.zeros(N)

        for i in range(N):
            # how many of i's true top-50 ended in i's connections?
            hits_per_user[i] = len(top_50_indices[i].intersection(connections[i]))

        # e.g. compute the average number of top-50 found
        avg_hits = hits_per_user.mean()

        print(f"Average coverage of each user's top-50: {avg_hits:.2f}")
        return {'avg_hits':avg_hits, 'nbr_connections':np.mean([len(i) for i in connections])}

    # recall_per_random_expansion = []
    # for r in range(21):
    #     recall_per_random_expansion.append(calculate_baseline_recall_of_topN(r))
    # pd.Series(recall_per_random_expansion).to_csv(os.path.join('csvs_and_graphs','recall_baseline_random.csv'))

    # df_avgemb['closest_user_cosine'].hist(bins = 100)
    # plt.title('Closest user by T5-large embedding')
    # plt.xlabel('Highest cosine similarity of a user')
    # plt.ylabel('Number of users')
    # plt.xlim(0.55, 1)
    # plt.show()


    # # # Generate boxplot
    # plt.figure(figsize=(20, 8))
    # plt.boxplot(ranked_similarities.values, showfliers=False)
    # plt.xlabel("Rank (1 = Top Similarity)", fontsize=14)
    # plt.ylabel("Cosine Similarity", fontsize=14)
    # plt.title("Boxplot of Cosine Similarities by Rank Across Users", fontsize=16)
    # # Keep all ticks but reduce labels
    # ticks = range(1, num_users + 1, 1)  # Keeps all ticks
    # label_step = 100  # Adjust this for fewer labels (e.g., every 100th label)
    # labels = [tick if tick % label_step == 0 else '' for tick in ticks]  # Empty labels for most ticks
    # plt.xticks(ticks=ticks, labels=labels, rotation=90)  # Rotate labels if necessary for better visibility
    # plt.show()




    # ranked_similarities[list(range(1,number_closest_neighbors + 1))].mean(axis = 1).hist(bins = 100)
    # plt.xlabel(f"Cosine Similarity", fontsize=14)
    # plt.ylabel("Number of users", fontsize=14)
    # plt.xlim(0.55, 1)
    # plt.title(f"Mean Similarity of top {number_closest_neighbors}", fontsize=16)
    # plt.show()
    #
    # ranked_similarities.mean(axis = 1).hist(bins = 100)
    # plt.xlabel(f"Cosine Similarity", fontsize=14)
    # plt.ylabel("Number of users", fontsize=14)
    # plt.title(f"Mean Similarity of all neighbors", fontsize=16)
    # plt.show()

    print ('Train data per user: ',data.groupby('AnonID')['DocIndex'].count().iloc[0])
    print ('Test data per user: ',test_data.groupby('AnonID')['DocIndex'].count().iloc[0])
    print ('Finished constructing user embeddings')
    max_acc_attainable = test_data[test_data['DocIndex'].isin(data['DocIndex'].unique())].shape[0]/test_data.shape[0]
    print (f'Maximum accuracy attainable with current graph: {max_acc_attainable}')
    print ()
    return df_avgemb, data, test_data, cosine_sim_df
