import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_baseline(number_experiments = 5, tree_anon_closest_users_df = None,
                       user_docs_train = None,user_docs_test = None,
                       df_avgemb = None, consider_contacts = True, N = None, M = None):
    if user_docs_train is None:
        user_docs_train = pd.read_pickle(os.path.join('results', 'traindata.pkl'))
    if user_docs_test is None:
        user_docs_test = pd.read_pickle(os.path.join('results','testdata.pkl'))


    if tree_anon_closest_users_df is None:
        tree_anon_closest_users_df = pd.read_pickle(os.path.join('results','tree_anon_closest_users_df.pkl'))

    nbr_docs_test_per_user = user_docs_test.groupby('AnonID')['DocIndex'].count().max()
    assert nbr_docs_test_per_user == user_docs_test.groupby('AnonID')['DocIndex'].count().min()

    # Function to sample 10 other AnonIDs for each AnonID
    def sample_anonids(exclude_id, all_ids, n=None):
        # Exclude the current ID and sample from the rest
        available_ids = all_ids[all_ids != exclude_id]
        return np.random.choice(available_ids, size=n, replace=False)

    # Get the unique AnonIDs as a NumPy array
    unique_anonids = user_docs_train['AnonID'].unique()


    estimated_prob = []
    number_users_queried = int(round(tree_anon_closest_users_df['Number_Queried'].mean()))
    number_to_contact = int(round(tree_anon_closest_users_df['Connected_Users'].mean()))

    if N is not None:
        number_users_queried = N
    if M is not None:
        number_to_contact = M
    print ()
    print (f'BASELINE STATS')
    print (f'Number of contacted people {number_to_contact}')
    print (f'Number of queried people {number_users_queried}')
    print ()

    for experiment in range(number_experiments):
        # Generate the samples and perform lookups
        prop_found = []
        for i, anon_id in enumerate(unique_anonids):
            # Sample 10 other AnonIDs
            # number_users_queried = tree_anon_closest_users_df.loc[anon_id, 'Number_Queried']
            if consider_contacts:
                sampled_ids = sample_anonids(anon_id, unique_anonids, n=number_to_contact)

                sampled_ids = df_avgemb[df_avgemb['AnonID'].isin(sampled_ids)][['AnonID', 'embedding']]
                point_emb = np.array(
                    df_avgemb[df_avgemb['AnonID'] == anon_id]['embedding'].iloc[0].tolist()
                ).reshape(1, -1)
                embs = np.array(sampled_ids['embedding'].tolist())
                cosine_sim_alltimeclosest = cosine_similarity(point_emb, embs).squeeze(0)
                sampled_ids['similarity'] = cosine_sim_alltimeclosest
                sampled_ids.sort_values('similarity', ascending=False, inplace=True)
                sampled_ids = set(sampled_ids['AnonID'].iloc[:number_users_queried])
            else:
                sampled_ids = sample_anonids(anon_id, unique_anonids, n=number_users_queried-1)
            
            
            

            personal_docs = set(user_docs_train[user_docs_train['AnonID'].isin(sampled_ids)]['DocIndex'])
            target_docs = set(user_docs_test[user_docs_test['AnonID'] == anon_id]['DocIndex'])
            nbr_overlap = target_docs.intersection(personal_docs)
            prop_found.append(len(nbr_overlap)/nbr_docs_test_per_user)

        df = pd.DataFrame({
            'AnonID': unique_anonids,
            'Hits': prop_found
        })
        estimated_prob.append(df['Hits'].mean()*100)




    mean_res = np.mean(estimated_prob)
    if len(estimated_prob)>1:
        std_res = np.std(estimated_prob)
    else:
        std_res = -1

    print (f'mean: {mean_res}, std: {std_res}')
    return mean_res, std_res

