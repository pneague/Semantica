import pandas as pd
import os
from Estimate_baseline_prob import calculate_baseline
from Construct_avg_embedding import construct_anon_embedding
import logging



nbr_train_recs = 20
nbr_test_records = 10
small_user_filter = True

tree_anon_closest_users_df = pd.read_pickle(os.path.join('results','tree_anon_closest_users_df.pkl'))
df_avgemb, user_docs_train, user_docs_test, cosine_sim_df = construct_anon_embedding(
                    nbr_train_recs=nbr_train_recs,
                    nbr_test_records=nbr_test_records,
                    sample_rdn_test = True,
                    filter_out_users_with_few_docs = small_user_filter
                )
experiment_name = f'all_baseline_results_trainRecs{nbr_train_recs}_testRecs{nbr_test_records}_smallUserFilter{small_user_filter}'

logging.basicConfig(
    filename=f'{experiment_name}.log',        # <-- Change this to your preferred log filename/path
    level=logging.INFO,           # INFO level will record info, warnings, errors, etc.
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True
)

baseline_results = {}
Ns = range (500)

logging.info("Embeddings Calculated, proceeding to estimate baselines.")
for N in Ns:
    logging.info(f"Esimating baseline with N = {N}")
    tree_anon_closest_users_df['Number_Queried'] = N
    tree_anon_closest_users_df['Connected_Users'] = N
    mean_random, std_random = calculate_baseline(number_experiments=5,consider_contacts=False,
                                                 N = N, M = N )

    baseline_results[N] = (mean_random, std_random)
    baseline_results_df = pd.DataFrame.from_dict(baseline_results, orient='index')
    baseline_results_df.to_pickle(os.path.join('results',experiment_name + '.pkl'))
    print (f'{N}: {round(mean_random,3), round(std_random, 3)}')


baseline_results = pd.DataFrame.from_dict(baseline_results, orient='index')
baseline_results.to_pickle(os.path.join('results',experiment_name + '.pkl'))
