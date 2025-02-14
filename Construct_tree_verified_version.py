import pandas as pd
import copy
from sklearn.cluster import KMeans
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import copy
import dill
from collections import defaultdict
import random
from helper import *

co_occurrences = defaultdict(set)
random.seed(42)


def store_tree_locally(tree, path_to_tree_pickle="temp/tree.pkl"):
    # Make sure the directory 'temp' exists
    os.makedirs(os.path.dirname(path_to_tree_pickle), exist_ok=True)
    with open(path_to_tree_pickle, "wb") as f:
        dill.dump(tree, f)
    print(f"Tree saved to {path_to_tree_pickle}")

def construct_tree(record_limit_per_leafnode, N, DELTA,
                   df_avgemb, user_docs_train, user_docs_test, cosine_sims, print_hists = False, M = 25,
                   perform_query_test = True, EXPANSION_ROUNDS = 10):
    # df_avgemb = pd.read_pickle(os.path.join('results','average_embedding_small.pkl'))
    # user_docs_test = pd.read_pickle(os.path.join('results','testdata.pkl'))
    # user_docs_train = pd.read_pickle(os.path.join('results','traindata.pkl'))
    # cosine_sims = pd.read_pickle(os.path.join('results','cosine_sims.pkl'))

    global user_train_docs
    df_avgemb['embedding'] = list(normalize(np.vstack(df_avgemb['embedding']), axis=1))


    class Node:
        node_counter = 0

        def __init__(self, depth, parent=None, path_from_root=None, tree_ref=None):
            self.depth = depth
            self.parent = parent
            self.path_from_root = path_from_root if path_from_root is not None else []
            self.nodes_dict = {}
            self.node_id = Node.node_counter
            Node.node_counter += 1

            # If we have a reference to the tree, store this node in tree's dictionary
            if tree_ref is not None:
                tree_ref.nodes_dict[self.node_id] = self

    class SplitNode(Node):
        def __init__(self, depth, centroid_left, centroid_right, parent=None, path_from_root=None, tree_ref=None):
            super().__init__(depth, parent, path_from_root, tree_ref=tree_ref)
            self.centroid_left = centroid_left
            self.centroid_right = centroid_right
            self.left_child = None
            self.right_child = None

        def get_children(self):
            return [self.left_child, self.right_child]

    class LeafNode(Node):
        def __init__(self, depth, parent=None, path_from_root=None, tree_ref=None):
            super().__init__(depth, parent, path_from_root, tree_ref=tree_ref)
            self.records = []

        def add_record(self, record):


            # For each existing record in this leaf,
            # record the co-occurrence between them and the new record.
            for existing_rec in self.records:
                co_occurrences[existing_rec['AnonID']].add(record['AnonID'])
                co_occurrences[record['AnonID']].add(existing_rec['AnonID'])

            self.records.append(record)


    class Tree:
        def __init__(self, record_limit_per_leafnode=record_limit_per_leafnode):
            # The root starts as a leaf node at depth 0.
            self.queries_done = []
            self.nodes_dict = {}
            self.root = LeafNode(depth=0, parent=None, path_from_root=[], tree_ref=self)
            self.root.path_from_root = [self.root.node_id]
            self.maximum_attained_depth = 0
            self.record_limit_per_leafnode = record_limit_per_leafnode

        def get_all_nodes(self):
            all_nodes = []
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                all_nodes.append(node)
                if isinstance(node, SplitNode):
                    queue.extend(node.get_children())
            return all_nodes

        def get_split_nodes(self, depth):
            """
            Retrieves all nodes at a given depth. If depth = maximum_attained_depth,
            we might return leaf nodes. Otherwise, return split nodes at that depth.
            """
            nodes_at_depth = []
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                if node.depth == depth:
                    nodes_at_depth.append(node)
                elif node.depth < depth:
                    if isinstance(node, SplitNode):
                        queue.extend(node.get_children())
            return nodes_at_depth

        def turn_leafnode_into_splitnode(self, leaf_node, delta=DELTA):
            """Perform K-means clustering on the leaf node's records and turn it into a split node."""
            # Extract embeddings
            embeddings = np.array([rec['embedding'] for rec in leaf_node.records])  # Assuming record = (AnonID, embedding)

            # Perform K-means with k=2
            kmeans = KMeans(n_clusters=2, random_state=0)
            # kmeans = SphericalKMeans(n_clusters=2, random_state=0)
            labels = kmeans.fit_predict(embeddings)
            centroid_left = kmeans.cluster_centers_[0]
            centroid_right = kmeans.cluster_centers_[1]


            # Create the new split node
            split_node = SplitNode(depth=leaf_node.depth,
                                   centroid_left=centroid_left,
                                   centroid_right=centroid_right,
                                   parent=leaf_node.parent,
                                   path_from_root=leaf_node.path_from_root,
                                   tree_ref=self)

            # Assign records to two new leaf nodes
            left_leaf = LeafNode(depth=leaf_node.depth + 1,
                                 parent=split_node,
                                 path_from_root=split_node.path_from_root + [split_node.node_id],
                                 tree_ref=self)
            right_leaf = LeafNode(depth=leaf_node.depth + 1,
                                  parent=split_node,
                                  path_from_root=split_node.path_from_root + [split_node.node_id],
                                  tree_ref=self)

            for rec, lbl in zip(leaf_node.records, labels):
                # print (rec['AnonID'])

                emb = rec['embedding']
                dist_left = np.linalg.norm(emb - centroid_left)
                dist_right = np.linalg.norm(emb - centroid_right)
                # dist_left = cosine_similarity([emb, centroid_left])[0][1]
                # dist_right = cosine_similarity([emb, centroid_right])[0][1]

                # Record distances at this new split
                rec['split_distances'].append((split_node.node_id, round(dist_left,3), round(dist_right,3), round(abs(dist_left - dist_right),3)))
                rec['path'].append(split_node.node_id)

                # Check how close the record is to the border
                if abs(dist_left - dist_right) < delta:
                    # It's close to the boundary, so duplicate
                    rec['split_distances'].append((f'Delta too small, duplicating to nodes: '
                                                   f'{left_leaf.node_id}, {right_leaf.node_id}'))
                    left_leaf.add_record(rec)
                    right_leaf.add_record(rec)
                else:
                    if lbl == 0:
                        left_leaf.add_record(rec)
                    else:
                        right_leaf.add_record(rec)

            # Update maximum_attained_depth if needed
            if left_leaf.depth > self.maximum_attained_depth:
                self.maximum_attained_depth = left_leaf.depth
            if right_leaf.depth > self.maximum_attained_depth:
                self.maximum_attained_depth = right_leaf.depth

            # Connect children to the split node
            split_node.left_child = left_leaf
            split_node.right_child = right_leaf

            return split_node

        def position_record(self, record):
            self._insert_record_downstream(self.root, record)

        def _insert_record_downstream(self, current_node, record):
            """
            Traverse the tree starting from the root:
              - If current node is a split node, determine which centroid is closer
                and record distances.
                If distance to centroids too small, copy record and send it both ways
              - If it's a leaf node, add the record there.
                If the leaf node exceeds the limit, split it.
            """
            while True:
                if isinstance(current_node, SplitNode):
                    embedding = record['embedding']
                    dist_left = np.linalg.norm(embedding - current_node.centroid_left)
                    dist_right = np.linalg.norm(embedding - current_node.centroid_right)
                    # dist_left = cosine_similarity([embedding, current_node.centroid_left])[0][1]
                    # dist_right = cosine_similarity([embedding, current_node.centroid_right])[0][1]

                    diff = abs(dist_left - dist_right)
                    if diff < DELTA:
                        # Duplicate record to both sides
                        record_copy = copy.deepcopy(record)
                        self._insert_record_downstream(current_node.left_child, record)
                        self._insert_record_downstream(current_node.right_child, record_copy)
                        record['split_distances'].append((current_node.node_id, round(dist_left,3), round(dist_right,3), round(diff,3), 'DUPLICATING'))
                        return
                    else:
                        record['split_distances'].append((current_node.node_id, round(dist_left,3), round(dist_right,3), round(diff,3)))
                        # Proceed with the closer branch
                        if dist_left < dist_right:
                            current_node = current_node.left_child
                        else:
                            current_node = current_node.right_child

                elif isinstance(current_node, LeafNode):
                    record['path'] = current_node.path_from_root[:]
                    current_node.add_record(record)
                    if len(current_node.records) > self.record_limit_per_leafnode:
                        self.replace_leaf_with_splitnode(current_node)
                    return

        def replace_leaf_with_splitnode(self, leaf_node, delta = DELTA):
            """Replace a leaf node in the tree with a split node after KMeans."""
            # If the root is the leaf to be replaced
            if self.root is leaf_node:
                self.root = self.turn_leafnode_into_splitnode(leaf_node, delta = delta)
                return

            parent, is_left_child = self._find_parent(self.root, leaf_node)
            if parent is not None:
                new_split = self.turn_leafnode_into_splitnode(leaf_node, delta = delta)
                if is_left_child:
                    parent.left_child = new_split
                else:
                    parent.right_child = new_split

        def _find_parent(self, current_node, target_node):
            """
            Helper function to find the parent of target_node.
            Returns a tuple (parent_node, is_left_child) or (None, None) if not found.
            """
            if isinstance(current_node, SplitNode):
                if current_node.left_child is target_node:
                    return current_node, True
                if current_node.right_child is target_node:
                    return current_node, False

                left_result = self._find_parent(current_node.left_child, target_node)
                if left_result[0] is not None:
                    return left_result

                right_result = self._find_parent(current_node.right_child, target_node)
                if right_result[0] is not None:
                    return right_result

            return None, None

        def get_all_leaf_nodes(self):
            """
            Returns a list of all leaf nodes in the tree.
            """
            leaf_nodes = []
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                if isinstance(node, LeafNode):
                    leaf_nodes.append(node)
                elif isinstance(node, SplitNode):
                    queue.extend(node.get_children())
            return leaf_nodes

        def get_anon_ids_from_leaf(self, leaf_node):
            """
            Given a leaf node, return a list of AnonIDs present in that leaf node.
            """
            return [rec['AnonID'] for rec in leaf_node.records]

        def get_path_to_node(self, node):
            """
            Returns the path_from_root for a given node, i.e., the sequence of "left"/"right" moves.
            """
            return node.path_from_root

        def nodes_at_distance(self, start_node, distance):
            """
            Returns all nodes exactly 'distance' edges away from start_node.
            distance=0 would return [start_node] itself.
            """
            # if 125981 in [i[0] for i in start_node.records]:
            #     print (1)
            if distance == 0:
                return [start_node]

            # Perform a BFS from start_node
            visited = set()
            queue = [(start_node, 0)]
            result = []
            while queue:
                current, dist = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    # Get neighbors: parent and children
                    neighbors = []
                    if current.parent is not None:
                        neighbors.append(current.parent)
                    if isinstance(current, SplitNode):
                        neighbors.extend(current.get_children())

                    for n in neighbors:
                        if n not in visited:
                            if dist + 1 == distance:
                                result.append(n)
                            else:
                                queue.append((n, dist + 1))
            return result

            # ---------------------------------------------------------------------
            # ---------------------- NEW QUERY FUNCTIONALITY ----------------------
            # ---------------------------------------------------------------------

        def get_leaf_nodes_for_query(self, query_embedding, delta=DELTA):
            """
            Routes a query embedding from the root to the appropriate leaf node(s).
            If the distance to both child centroids is less than delta, the query
            is duplicated. Returns *all* leaf nodes where this query would land.
            """
            # We do this similarly to _insert_record_downstream, but we do NOT store anything.
            leaf_nodes = []
            stack = [(self.root, query_embedding)]

            while stack:
                current_node, q_emb = stack.pop()
                if isinstance(current_node, SplitNode):
                    dist_left = np.linalg.norm(q_emb - current_node.centroid_left)
                    dist_right = np.linalg.norm(q_emb - current_node.centroid_right)
                    diff = abs(dist_left - dist_right)

                    if diff < delta:
                        # Duplicate the query to both subtrees
                        stack.append((current_node.left_child, q_emb))
                        stack.append((current_node.right_child, q_emb))
                    else:
                        if dist_left < dist_right:
                            stack.append((current_node.left_child, q_emb))
                        else:
                            stack.append((current_node.right_child, q_emb))

                elif isinstance(current_node, LeafNode):
                    leaf_nodes.append(current_node)

            return leaf_nodes

        def get_closest_users_for_query(self, query_embedding, M=25, delta=DELTA, N = 10):
            """
            1. Route the query to one or more leaf nodes.
            2. For each leaf node, do a BFS outward (using nodes_at_distance) until
               we've collected >= M AnonIDs.
            3. Compute the distances from the query to these candidate AnonIDs.
            4. Return the top M closest AnonIDs by L2 distance (or you could
               switch to cosine similarity if desired).

            The query is *not* stored in the tree; this is ephemeral.
            """
            # Step 1: find leaf nodes
            target_leaf_nodes = self.get_leaf_nodes_for_query(query_embedding, delta=delta)

            # We'll gather candidates from these leaf nodes plus neighbors
            candidate_anon_ids = set()
            max_distance = self.maximum_attained_depth * 4

            # BFS from each leaf node outward
            for leaf_node in target_leaf_nodes:
                for dist in range(max_distance + 1):
                    nodes_at_dist = self.nodes_at_distance(leaf_node, dist)
                    for n in nodes_at_dist:
                        if isinstance(n, LeafNode) and len(candidate_anon_ids)< M:
                            candidate_anon_ids.update(r['AnonID'] for r in n.records)
                    # Once we have at least M candidates from BFS, we can break
                    if len(candidate_anon_ids) >= M:
                        break

            # Step 3: compute distances to the query
            # We'll gather embeddings for each candidate_anon_id from the df (or a stored dictionary)
            # In practice, you'd want a dictionary: anonid_to_embedding[AnonID] -> np.array
            # For example, let's suppose you have stored them in "self.anonid_to_embedding".
            # But here, let's just assume we rely on an external variable or pass it in.
            # As an example, let's just create a placeholder:
            #   self.anonid_to_embedding = ...
            # For now, we rely on the global scope. In real usage, you'd adapt accordingly.

            # We'll do a mock access for demonstration. We assume you pass in a
            # dictionary: self.anonid_to_embedding = { ... }
            # If not, you need to create it in your code before building the tree.
            # Let's forcibly require it for this demonstration:
            try:
                anonid_to_embedding = self.anonid_to_embedding
            except AttributeError:
                raise ValueError(
                    "You must set 'self.anonid_to_embedding' as a dict: {AnonID: embedding_array} "
                    "before calling get_closest_users_for_query()."
                )

            # Filter out any candidate that doesn't exist in our embedding dictionary
            candidate_anon_ids = [
                cid for cid in candidate_anon_ids if cid in anonid_to_embedding
            ]

            query_embedding = query_embedding.reshape(1, -1)
            distances = []
            for cid in candidate_anon_ids:
                cand_emb = anonid_to_embedding[cid].reshape(1, -1)
                # dist = np.linalg.norm(query_embedding - cand_emb)
                dist = 1 - cosine_similarity(query_embedding,cand_emb)[0][0]
                distances.append((cid, dist))

            # Sort by distance ascending
            distances.sort(key=lambda x: x[1])
            # Return the top M
            closest_anon_ids = [cid for cid, _ in distances[:N]]
            return closest_anon_ids

            # ---------------------------------------------------------------------
            # ---------------------- NEW QUERY FUNCTIONALITY ----------------------
            # ---------------------------------------------------------------------

    def get_path_to_root(node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path



    # Example usage:
    # Suppose df_avgemb is your dataframe with columns 'AnonID', 'embedding'
    embeddings = df_avgemb['embedding'].values  # This should be a list/array of 1024-dimensional arrays
    anon_ids = df_avgemb['AnonID'].values

    tree = Tree(record_limit_per_leafnode=record_limit_per_leafnode)
    for anon_id, emb in zip(anon_ids, embeddings):
        record = {
            'AnonID': anon_id,
            'embedding': emb,
            'path': [],
            'split_distances': []
        }
        tree.position_record(record)

    all_leafs = tree.get_all_leaf_nodes()
    leaf_cosine_sims = []
    nbr_users_per_leaf = []
    final_records = []
    for idx, lf in enumerate(all_leafs):
        for rec in lf.records:
            final_records.append(rec)

        # print("Path to leaf:", tree.get_path_to_node(lf))
        # print("AnonIDs in leaf:", tree.get_anon_ids_from_leaf(lf))
        embs = np.array(df_avgemb[df_avgemb['AnonID'].isin(tree.get_anon_ids_from_leaf(lf))]['embedding'].tolist())

        if embs.shape[0] == 0:
            continue
        nbr_users_per_leaf.append(embs.shape[0])
        # Calculate cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(embs)

        # Exclude self-similarities (diagonal) by setting them to NaN
        np.fill_diagonal(cosine_sim_matrix, np.nan)

        # print (cosine_sim_matrix)
        # print (np.mean(np.nanmean(cosine_sim_matrix, axis = 1)))
        leaf_cosine_sims.append(np.mean(np.nanmean(cosine_sim_matrix, axis = 1)))
    df_avgemb = pd.DataFrame.from_dict(final_records)



    # Create a dictionary of AnonID -> list of node_ids
    anonid_to_nodeids = {}
    for leaf in tree.get_all_leaf_nodes():
        for rec in leaf.records:
            anon_id = rec['AnonID']
            if anon_id not in anonid_to_nodeids:
                anonid_to_nodeids[anon_id] = []
            anonid_to_nodeids[anon_id].append(leaf.node_id)


    # CALCULATING DISTANCE TO EACH NODE FROM EACH NODE - TAKES A WHILE
    # You can load this pickle for the moment

    record_distance_map_list = []
    # For each anonID, consider each node where it resides
    for anon_id, node_ids in anonid_to_nodeids.items():
        # node_ids is now a list of node_ids where this anon_id ended up
        # We will process each node individually
        # if anon_id == 770356:
        #     print (1)
        for node_id in node_ids:
            node = tree.nodes_dict[node_id]

            # Initialize data structure for this particular copy of anon_id
            distance_map_for_copy = {}
            total_number_of_neighbors = []

            # Determine a maximum search distance
            max_distance = tree.maximum_attained_depth * 4

            for dist in range(max_distance + 1):
                nodes_at_dist = tree.nodes_at_distance(node, dist)
                # Gather all records from these nodes
                records_at_dist = []
                for n in nodes_at_dist:
                    if isinstance(n, LeafNode):
                        # Collect AnonIDs from this leaf node
                        # Note: Each record is a dict with 'AnonID' and 'embedding'
                        records_at_dist.extend([r['AnonID'] for r in n.records])

                if records_at_dist:
                    # Avoid duplicates by using a set
                    total_number_of_neighbors.extend(records_at_dist)
                    total_number_of_neighbors = list(set(total_number_of_neighbors))
                    distance_map_for_copy[dist] = records_at_dist

                # If we've found more than N neighbors, we can stop early
                if len(total_number_of_neighbors) > M:
                    break

            # Append the result for this copy of the anon_id
            record_distance_map_list.append({
                'AnonID': anon_id,
                'NodeID': node_id,
                'DistanceMap': distance_map_for_copy
            })
    # record_distance_map_file = os.path.join('results', 'record_distance_map_5docs_30maxnodesperleaf.pkl')
    # # Save the data to a pickle file
    # with open(record_distance_map_file, 'wb') as f:
    #     pkl.dump(record_distance_map, f)

    # with open(record_distance_map_file, 'rb') as f:
    #     record_distance_map = pkl.load(f)
    # print("record_distance_map successfully loaded.")


    # PLOTTING THE TOP N CLOSEST NODES IN TREE
    anonid_to_embedding = dict(zip(df_avgemb['AnonID'], df_avgemb['embedding']))
    mean_similarities = {}
    tree_closest_users_dict = {}
    tree.anonid_to_embedding = anonid_to_embedding

    for entry in record_distance_map_list:
        target_anon_id = entry['AnonID']
        node_id = entry['NodeID']
        dist_map = entry['DistanceMap']

        # if target_anon_id == 770356:
        #     print (1)

        # The rest of the logic remains similar,
        # but now you'll store results keyed by (AnonID, NodeID) if you want.
        candidates = [target_anon_id]
        for d in sorted(dist_map.keys()):
            for candidate_anon_id in dist_map[d]:
                if candidate_anon_id not in candidates:
                    candidates.append(candidate_anon_id)
                    if len(candidates) == M:
                        break
            if len(candidates) == M:
                break

        # Compute mean similarity if candidates found
        if len(candidates) == 0:
            mean_similarities[(target_anon_id, node_id)] = np.nan
            continue

        target_emb = anonid_to_embedding[target_anon_id].reshape(1, -1)
        candidate_embs = np.array([anonid_to_embedding[c] for c in candidates])
        sim_matrix = cosine_similarity(target_emb, candidate_embs)

        tree_closest_users_dict[(target_anon_id, node_id)] = candidates
        mean_similarities[(target_anon_id, node_id)] = np.mean(sim_matrix)

    def expand_neighbors_via_referral_all_users(
            user_ids,
            tree_anon_closest_users_df,
            cosine_sims,
            absolute_closest_users_df,
            DELTA,
            num_rounds=5,
            K=20
    ):
        """
        This function interleaves rounds across all users:
            Round 1: user1 asks once, user2 asks once, ...
            Round 2: user1 asks once, user2 asks once, ...
            etc.

        Returns a dict mapping each user -> (final_topk_set, all_shared_set).
        """

        # --------------------------
        # Helper functions (same as before)
        # --------------------------
        def build_initial_topk(u_id, neighbors, cos_sims, max_k=20):
            """
            Quickly extract the top-K neighbors for user u_id from cos_sims,
            restricted to 'neighbors', using a vectorized approach.
            """
            # If the user isn't in cos_sims' rows, return empty
            if u_id not in cos_sims.index:
                return []

            # Intersect the given neighbors with cos_sims' columns so we don't get KeyErrors
            valid_neighbors = [n for n in neighbors if n in cos_sims.columns]
            if not valid_neighbors:
                return []

            # Slice cos_sims to get the row for u_id but only the columns for valid_neighbors
            # This gives us a Series indexed by those valid_neighbors
            row_series = cos_sims.loc[u_id, valid_neighbors]

            # Use nlargest to grab the top K values in descending order
            top_series = row_series.nlargest(max_k)

            # Convert to a list of (neighbor, similarity)
            return list(top_series.items())

        def add_candidate_if_better_than_Kth(u_id, cand_id, topk_list_local, cos_sims, max_k, all_shared):
            if (u_id not in cos_sims.index) or (cand_id not in cos_sims.columns):
                return topk_list_local
            cand_sim = cos_sims.loc[u_id, cand_id]

            if len(topk_list_local) < max_k:
                topk_list_local.append((cand_id, cand_sim))
                topk_list_local.sort(key=lambda x: x[1], reverse=True)
                return topk_list_local

            kth_sim = topk_list_local[max_k - 1][1]
            if cand_sim > kth_sim:
                all_shared.add(cand_id)
                topk_list_local.append((cand_id, cand_sim))
                topk_list_local.sort(key=lambda x: x[1], reverse=True)
                topk_list_local = topk_list_local[:max_k]
            return topk_list_local
        def scatterplot_commoncounts(user_to_topk, round_idx, DELTA, user_to_all_shared):
            ### PRINT COMMON COUNTS EVERY ROUND ####
            x = []
            list_contacts = []
            for u in user_ids:
                x.append(len(set(absolute_closest_users_df.loc[u].values).intersection(
                    set([i[0] for i in user_to_topk[u]]))))
                list_contacts.append(len(user_to_all_shared[u]))
            plt.scatter(range(len(x)), x, s=0.1)
            plt.suptitle(f'Number of absolute 50 closest users caught in "known-users" list')
            plt.title(f'Expansion round {round_idx}, delta {DELTA}')
            plt.ylabel('Number of neighbors in absolute closest 50 list')
            plt.xlabel('User ID')
            plt.ylim(-2,52)
            plt.show()
            print(f'Mean, std and median common counts for round {round_idx}: {np.mean(x)}, {np.std(x)}, {np.median(x)} '
                  f'and mean, std and median number of connected-users: {np.mean(list_contacts)}, {np.std(list_contacts)}, {np.median(list_contacts)}')
            ### PRINT COMMON COUNTS EVERY ROUND ####

        # ---------------------------------------------------
        # 1) Initialize data structures for ALL users
        # ---------------------------------------------------
        # user_to_topk stores each user's current top-K list of (nbr, sim)
        user_to_topk = {}
        # user_to_all_shared stores the set of anyone introduced so far for that user
        user_to_all_shared = {}
        # user_to_asked_set tracks neighbors that have already been "asked" by that user
        user_to_asked_set = {}
        # current_neighbors_dict stores initial neighbors
        current_neighbors_dict = {}

        # Build initial topK for each user
        for u in user_ids:
            current_neighbors_dict[u]= tree_anon_closest_users_df[u]
            initial_topk_list = build_initial_topk(u, current_neighbors_dict[u], cosine_sims, max_k=K)
            user_to_topk[u] = initial_topk_list
            user_to_all_shared[u] = set(current_neighbors_dict[u])
            user_to_asked_set[u] = set()

        # ---------------------------------------------------
        # 2) Run multiple rounds, but interleave across users
        # ---------------------------------------------------
            # For each round, EVERY user gets exactly ONE ask
        for round_idx in range(num_rounds):
            scatterplot_commoncounts(user_to_topk, round_idx=round_idx, DELTA = DELTA,
                                     user_to_all_shared=user_to_all_shared)
            tree_anon_closest_users_df = copy.deepcopy(user_to_all_shared)


            # print (f'Going through user-sharing in round {round_idx+1} out of {num_rounds}')

            for u in user_ids:
                topk_list = user_to_topk[u]
                asked_set = user_to_asked_set[u]
                all_shared = user_to_all_shared[u]

                # If no top-k to ask from, skip
                if not topk_list:
                    continue

                # Potential referrers = top-k neighbors minus those we already asked
                potential_referrers = [(nid, sim) for (nid, sim) in topk_list if nid not in asked_set]
                if not potential_referrers:
                    # We have top-k, but have asked them all, so skip
                    continue

                # Pick a random referrer
                referrer_id, _ = random.choice(potential_referrers)
                asked_set.add(referrer_id)  # Mark as asked this round

                # If referrer not in tree_anon_closest_users_df, skip
                if referrer_id not in tree_anon_closest_users_df:
                    continue

                # The set of possible referrals is everyone the referrer co-occurs with
                # minus the user themselves, minus known topK IDs
                possible_referrals = [
                    c for c in tree_anon_closest_users_df[referrer_id]
                    if c != u and c not in all_shared
                ]

                # Attempt to add each candidate if it's better than the Kth best
                for cand_id in possible_referrals:
                    if (cand_id in cosine_sims.index) and (cand_id in cosine_sims.columns):
                        # all_shared.add(cand_id)
                        topk_list = add_candidate_if_better_than_Kth(u, cand_id, topk_list,
                                                                     cosine_sims, max_k=K,
                                                                     all_shared=all_shared)

                # Update our records for user u
                user_to_topk[u] = topk_list
                user_to_asked_set[u] = asked_set
                user_to_all_shared[u] = all_shared

        # ---------------------------------------------------
        # 3) Convert final top-K lists to sets + return
        # ---------------------------------------------------
        results = {}
        for u in user_ids:
            final_topk_set = {n_id for (n_id, _) in user_to_topk[u]}
            results[u] = (final_topk_set, user_to_all_shared[u])
        scatterplot_commoncounts(user_to_topk, round_idx = num_rounds, DELTA = DELTA,
                                 user_to_all_shared=user_to_all_shared)
        return results



    np.fill_diagonal(cosine_sims.values, -np.inf)
    # Find the 30 closest points for each row
    absolute_closest_users = {}
    for user in cosine_sims.index:
        absolute_closest_users[user] = cosine_sims.loc[user].nlargest(N).index.tolist()
    # Convert the result into a DataFrame
    absolute_closest_users_df = pd.DataFrame.from_dict(absolute_closest_users, orient='index', columns=[f'Closest_{i+1}' for i in range(N)])

    ############### Calculating number of common closest between tree and absolute neighbor orders ###############
    # Convert the tree_closest_users dictionary into a DataFrame
    tree_clone_closest_users_df = pd.DataFrame.from_dict(
        tree_closest_users_dict, orient='index',
        columns=[f'Closest_{i+1}' for i in range(len(next(iter(tree_closest_users_dict.values()))))])
    # Reset the index and split the tuple into two separate columns
    tree_clone_closest_users_df.index = pd.MultiIndex.from_tuples(tree_clone_closest_users_df.index, names=["AnonID", "NodeID"])

    # Calculate the number of common items for each user
    common_counts = {}
    tree_anon_closest_users_df = {}
    total_nbr_connected = defaultdict(list) # measures number of AnonID's of which each AnonID knows the embedding
    for i, user in enumerate(absolute_closest_users_df.index):
        tree_closest_users_list = tree_clone_closest_users_df.loc[user]
        tree_closest_users_set = []
        for node_id in tree_closest_users_list.index.tolist():
            tree_closest_users_set.extend(list(tree_clone_closest_users_df.loc[(user, node_id)]))
        tree_closest_users_set = set(tree_closest_users_set)
        tree_anon_closest_users_df[user] = tree_closest_users_set

    # Collect all users
    user_list = list(absolute_closest_users_df.index)

    # Build a dict: user -> initial neighbors (the 'tree_anon_closest_users_df[user]' sets)


    # Now call the new "all_users" function once
    results_all_users = expand_neighbors_via_referral_all_users(
        user_ids=user_list,
        tree_anon_closest_users_df=tree_anon_closest_users_df,  # same structure, user->set
        cosine_sims=cosine_sims,
        num_rounds=EXPANSION_ROUNDS,
        DELTA = DELTA,
        K=N,
        absolute_closest_users_df=absolute_closest_users_df
    )

    # results_all_users[u] = (final_topk_set, all_shared_set)

    for u in user_list:
        final_topk_u, all_shared_u = results_all_users[u]
        total_nbr_connected[u] = len(all_shared_u)
        common_counts[u] = len(set(absolute_closest_users_df.loc[u].values).intersection(final_topk_u))

        tree_anon_closest_users_df[u] = {'Closest_Users': list(final_topk_u), 'All_Shared': list(all_shared_u)}



        if i%1000 == 0:
            print (i)

    # Convert common counts into a DataFrame
    common_counts_df = pd.DataFrame.from_dict(common_counts, orient='index', columns=['Common_Count'])
    tree_anon_closest_users_df = pd.DataFrame.from_dict(tree_anon_closest_users_df, orient = 'index')
    total_nbr_connected = pd.DataFrame.from_dict(total_nbr_connected, orient = 'index', columns = ['Connected_Users'])
    # common_counts_df.index = pd.MultiIndex.from_tuples(common_counts_df.index, names=["AnonID", "node_id"])
    # common_counts_df = common_counts_df.groupby('AnonID')['Common_Count'].max().reset_index()

    ############### Calculating number of common closest between tree and absolute neighbor orders ###############



    ##### CALCULATING ACCURACY #####
    def calculate_userlevel_overlap_percentage(user_docs_train, user_docs_test, tree_clone_closest_users_df):
        """
        Calculates the percentage of documents that users are looking for, held by their closest users.

        Parameters:
            user_docs_train (pd.DataFrame): DataFrame with columns 'AnonID', 'DocIndex' representing documents users hold.
            user_docs_test (pd.DataFrame): DataFrame with columns 'AnonID', 'DocIndex' representing documents users are looking for.
            tree_clone_closest_users_df (pd.DataFrame): DataFrame with columns 'Closest1', 'Closest2', ..., 'ClosestN' representing the closest users.

        Returns:
            pd.DataFrame: A DataFrame with columns 'AnonID' and 'OverlapPercentage'.
        """
        # Prepare dictionaries for quick lookup
        user_train_docs = user_docs_train.groupby('AnonID')['DocIndex'].apply(set).to_dict()
        user_test_docs = user_docs_test.groupby('AnonID')['DocIndex'].apply(set).to_dict()

        # Initialize a list to store results
        results = []

        # Iterate through tree_clone_closest_users_df
        for user, closest_users in tree_clone_closest_users_df.iterrows():
            if isinstance(user,tuple):
                u = user[0]
                node_id = user[1]
            else:
                u = user
                node_id = None

            if closest_users.shape[0] == 1: # In case of the results_df_tree_anon
                closest_users = closest_users.tolist()[0]

            if u not in user_test_docs:
                # Skip users with no test documents
                continue

            # Documents the user is looking for
            test_docs = user_test_docs[u]

            # Collect all documents held by closest users
            closest_users_docs = set()
            for closest_user in closest_users:
                closest_users_docs.update(user_train_docs.get(closest_user, set()))

            # Calculate overlap
            overlap_docs = test_docs.intersection(closest_users_docs)
            overlap_percentage = (len(overlap_docs) / len(test_docs)) * 100 if test_docs else 0

            # Append to results
            results.append({'AnonID': u, 'EndNodeID': node_id, 'OverlapPercentage': overlap_percentage})

        # Create and return a dataframe for the results
        return pd.DataFrame(results)

    results_df_abs = calculate_userlevel_overlap_percentage(user_docs_train, user_docs_test, absolute_closest_users_df)
    # results_df_tree_clone = calculate_userlevel_overlap_percentage(user_docs_train, user_docs_test, tree_clone_closest_users_df)
    results_df_tree_anon = calculate_userlevel_overlap_percentage(user_docs_train, user_docs_test, tree_anon_closest_users_df[['Closest_Users']])
    tree_anon_closest_users_df['Number_Queried'] = tree_anon_closest_users_df['Closest_Users'].apply(lambda l: len(l))

    tree_anon_closest_users_df = pd.merge(total_nbr_connected, tree_anon_closest_users_df, left_index=True,
                                          right_index=True)
    tree_anon_closest_users_df.to_pickle(os.path.join('results','tree_anon_closest_users_df.pkl'))


    results_df_tree = results_df_tree_anon.groupby('AnonID')['OverlapPercentage'].max().reset_index()
    results_df_abs.drop(columns = ['EndNodeID'], inplace = True)

    train_doc_nbr = user_docs_train.groupby('AnonID')['DocIndex'].count().iloc[0]
    test_doc_nbr = user_docs_test.groupby('AnonID')['DocIndex'].count().iloc[0]


    cmn_counts_median = common_counts_df.Common_Count.median()
    cmn_counts_mean = common_counts_df.Common_Count.mean()
    nbr_users_contacted = np.mean(tree_anon_closest_users_df['All_Shared'].apply(lambda i: len(i)).mean())
    tree_overlap = results_df_tree["OverlapPercentage"].mean()
    absolute_overlap = results_df_abs["OverlapPercentage"].mean()



    if float(DELTA) == 0.:
        assert tree_anon_closest_users_df['Number_Queried'].min() == tree_anon_closest_users_df['Number_Queried'].max()

    print (f'Median Common Closest {N} Users: {cmn_counts_median}')
    print (f'Mean Common Closest {N} Users: {cmn_counts_mean}')
    print ()
    print (f'Accuracy with {train_doc_nbr} train and {test_doc_nbr}:')
    print (f'Absolute: {absolute_overlap}')
    print (f'Tree: {tree_overlap}')
    print (f'Number of users contacted mean: ', nbr_users_contacted)



    querylevel_accuracy = -1
    if perform_query_test == "Query-as-Node":
        store_tree_locally(tree, path_to_tree_pickle="temp/tree.pkl")
        df_q = pd.read_pickle('embedded_queries_df_BERT.pkl').rename(columns={'embedding': 'query_embedding'})
        # user_train_docs = user_docs_train.groupby('AnonID')['DocIndex'].apply(set).to_dict()
        user_docs_test = pd.merge(df_q, user_docs_test, on='QueryIndex')
        number_queries_done = 0
        results_df, querylevel_accuracy = parallel_calculate_queryAsNode_overlap_percentage(
            path_to_tree_pickle="temp/tree.pkl",
            user_train_docs=user_docs_train,
            user_docs_test=user_docs_test,
            M=M,
            N=N,
            # n_cores=4,  # or however many cores you want
            log_filename="querynode_progress.log"
        )

        print("QueryAsNode accuracy:", querylevel_accuracy * 100)

    elif perform_query_test == 'Closest-Users':
        df_q = pd.read_pickle('embedded_queries_df_BERT.pkl').rename(columns={'embedding': 'query_embedding'})
        user_docs_test = pd.merge(user_docs_test, df_q, on='QueryIndex')
        results_df, querylevel_accuracy = parallel_evaluate_routing(df_avgemb,
                                                user_docs_train,
                                                user_docs_test,
                                                tree_anon_closest_users_df,
                                                top_k=50)
        print (f'QueryLevel with Neighbor similarity: {querylevel_accuracy * 100}')

    elif perform_query_test == 'Closest-User-Chain':
        # test_data = pd.read_csv(os.path.join('dataset', 'data.csv'), sep='\t')
        # test_data = test_data[test_data['AnonID']]
        # user_docs_test = pd.merge(test_data[['AnonID','DocIndex']], user_docs_test,
        #                           left_on=['AnonID','DocIndex'], right_on = ['AnonID','DocIndex'])
        # del(test_data)
        df_q = pd.read_pickle('embedded_queries_df_BERT.pkl').rename(columns={'embedding': 'query_embedding'})
        user_docs_test = pd.merge(user_docs_test, df_q, on='QueryIndex')
        results_df, querylevel_accuracy = parallel_evaluate_routing_chain(df_avgemb,
                                                user_docs_train,
                                                user_docs_test,
                                                tree_anon_closest_users_df,
                                                top_k=50)
        print (f'QueryLevel with Neighbor similarity: {querylevel_accuracy * 100}')

    if print_hists:
        print (f'Number of leaf nodes: {len(leaf_cosine_sims)}')
        plt.hist(total_nbr_connected)
        plt.title(f"number of other-AnonID-embeddings known by AnonID's")
        plt.xlabel(f"number of other-AnonID-embeddings known")
        plt.ylabel(f"number of AnonID's")
        plt.show()


        plt.hist(mean_similarities.values(), bins=100)
        plt.title(f'mean similarity of closest {N} users in the tree')
        plt.xlim(0.8, 1.1)
        plt.show()

        plt.hist(leaf_cosine_sims, bins = int(len(leaf_cosine_sims)/50.0), density = True)
        plt.suptitle(f'Mean similarity between nodes in same leaf')
        plt.title(f'Delta = {DELTA}')
        plt.xlim(0.85, 1.05)
        plt.show()

        pd.Series(nbr_users_per_leaf).to_csv(os.path.join('csvs_and_graphs',f'nbr_users_per_leaf_delta{DELTA}.csv'))
        plt.hist(nbr_users_per_leaf, bins = 100)
        plt.suptitle(f'Number of users per leaf node')
        plt.title(f'Delta = {DELTA}')
        plt.show()

        common_counts_df['Common_Count'].hist(bins = 60)
        plt.suptitle(f'Absolutely Closest {N} Users for Each User in Tree')
        plt.title(f'Delta = {DELTA}')
        plt.xlabel(f'Number of Absolutely Closest {N}')
        plt.ylabel(f'Number of Users')
        plt.show()

        results_df_abs['OverlapPercentage'].hist(bins = 20)
        plt.suptitle(f'% docs users looked for, held by abs-closest {N} other users')
        plt.title(f'Users hold min 20 docs, are looking for next 10 docs')
        plt.xlabel('Percentage')
        plt.xlim(-5,105)
        plt.ylabel('Number of users')
        plt.show()

        results_df_tree['OverlapPercentage'].hist(bins = 20)
        plt.suptitle(f'% docs users looked for, held by tree-closest {N} other users')
        plt.title(f'Users hold min 20 docs, are looking for next 10 docs')
        plt.xlabel('Percentage')
        plt.xlim(-5,105)
        plt.ylabel('Number of users')
        plt.show()

        clones_per_user = tree_clone_closest_users_df.reset_index().groupby('AnonID')['NodeID'].count()
        clones_per_user.to_csv(os.path.join('csvs_and_graphs', f'clones_per_user_delta{DELTA}.csv'))
        plt.hist(clones_per_user, bins = clones_per_user.max() * 2)
        plt.suptitle("Number of clones per user")
        plt.title(f"Delta: {DELTA}")
        plt.xlabel('Number of clones')
        plt.ylabel('Number of users')
        plt.show()

    return (cmn_counts_mean, cmn_counts_median, nbr_users_contacted, tree_overlap,
            absolute_overlap, tree_anon_closest_users_df, querylevel_accuracy * 100)


