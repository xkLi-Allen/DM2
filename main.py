import numpy as np

COSINE = 'cosine' 
CORRELATION = 'correlation' 
JACCARD = 'jaccard'
ENABLE_WEIGHTING = False
USED_DISTANCE = COSINE


class PagesSimilarityMatrix:
    def __init__(self, pages=None, pages_vists_only=None, users_visited_pages_ids=None):
        self._pages = pages
        self._pages_vists_only = pages_vists_only
        self._users_visited_pages_ids = users_visited_pages_ids
        self._pages_similarities = {}
    def compute_matrix(self):
        weighted_vectors_cache = {}
        users_weights_cache = {}
        def weighted_vect(page_id, page_users):
            if page_id in weighted_vectors_cache:
                return weighted_vectors_cache[page_id]
            weighted_page_users = []
            for i, visit in enumerate(page_users):
                user_id = self._users_visited_pages_ids.keys()[i]
                if user_id in users_weights_cache:
                    weighted_page_users.append(users_weights_cache[user_id]*visit)
                    continue
                user_nbr_visits = len(self._users_visited_pages_ids[user_id])
                import math
                w = math.log(len(self._pages.keys())*1.0/user_nbr_visits)
                users_weights_cache[user_id] = w
                weighted_page_users.append(w*visit)
            assert len(page_users) == len(weighted_page_users)
            weighted_vectors_cache[page_id] = weighted_page_users
            return weighted_page_users
        import datetime
        print('Calculating sparse pages visits...', datetime.datetime.now())
        pages_visits = {}
        for page_id, page_users in self._pages_vists_only.items():
            pages_visits[page_id] = [1 if user_id in page_users else 0 for user_id in self._users_visited_pages_ids.keys()]
        for page_id in self._pages.keys():
            if page_id not in pages_visits:
                pages_visits[page_id] = [0]*len(self._users_visited_pages_ids.keys())
        print('Done, ',datetime.datetime.now())
        print('Calculating similarity between pages ...', datetime.datetime.now())
        self._pages_similarities = {}
        simitry_tracking = {}
        for page_id, page_users in pages_visits.items():
            sim_vector = []
            for tmp_page_id, tmp_page_users in pages_visits.items():
                if (tmp_page_id, page_id) in simitry_tracking:
                    # check for symmetric entry
                    sim_vector.append((tmp_page_id, {COSINE: simitry_tracking[(tmp_page_id, page_id)][COSINE], 
                                                     CORRELATION: simitry_tracking[(tmp_page_id, page_id)][CORRELATION], 
                                                     JACCARD: simitry_tracking[(tmp_page_id, page_id)][JACCARD]}))
                    continue
                pair_analysis = VectorsPairAnalysis(a_id=page_id,
                                                    a=page_users if not ENABLE_WEIGHTING else weighted_vect(page_id, page_users), 
                                                    b_id=tmp_page_id,
                                                    b=tmp_page_users if not ENABLE_WEIGHTING else weighted_vect(tmp_page_id, tmp_page_users))
                correlation_value = 0 if USED_DISTANCE  != CORRELATION else pair_analysis.pearson_correlation()
                cosine_similarity = 0 if USED_DISTANCE  !=  COSINE else pair_analysis.cosine_similarity()
                jaccard_index = 0 if USED_DISTANCE  !=  JACCARD else pair_analysis.jaccard_index()
                simitry_tracking[(page_id, tmp_page_id)] = {COSINE: cosine_similarity,
                                                            CORRELATION: correlation_value,
                                                            JACCARD: jaccard_index}
                sim_vector.append((tmp_page_id, {COSINE: cosine_similarity, 
                                                 CORRELATION: correlation_value, 
                                                 JACCARD: jaccard_index}))
            self._pages_similarities[page_id] = sim_vector
        print('Done, ',datetime.datetime.now())
        print('Sim matrix dimentions: ', len(list(self._pages_similarities.keys())), len(list(self._pages_similarities.values())[0]))
        return self._pages_similarities 
    def dump_matrix(self):
        import codecs
        tmp = []
        records_file = codecs.open('pages_similarity_matrix.csv', 'w')
        records_file.write(';%s\n'%','.join(map(str, [page_id for page_id, sim_vector in self._pages_similarities.items()])))
        for page_id, sim_vector in self._pages_similarities.items():
            line = [page_id] + list(map(str, [x[1][USED_DISTANCE] for x in sim_vector]))
            tmp.append([x[1][USED_DISTANCE] for x in sim_vector])
            records_file.write('%s\n'%';'.join(map(str, line)))
        records_file.close()
        records_file = codecs.open('pages_similarity_matrix_orange.csv', 'w')
        records_file.write('%d\n'%len(self._pages_similarities.keys()))
        for page_id, sim_vector in self._pages_similarities.items():
            line = [page_id] + list(map(str, [x[1][USED_DISTANCE] for x in sim_vector]))
            tmp.append([x[1][USED_DISTANCE] for x in sim_vector])
            records_file.write('%s\n'%'\t'.join(map(str, line)))
        records_file.close()
    def hierarchical_cluster(self,similarities=None):
        from sklearn.metrics import silhouette_score
        if similarities is None:
            distances = []
            for page_id, sim_vector in self._pages_similarities.items():
                distances.append([1-x[1][USED_DISTANCE] for x in sim_vector])
        else:
            distances = []
            for x in similarities:
                distances.append([1 - a for a in x])
        np_distances = np.asarray(distances)
        import scipy.cluster
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import squareform
        distances = squareform(np_distances)
        ddgm = scipy.cluster.hierarchy.linkage(distances, method='single')
        nodes = scipy.cluster.hierarchy.fcluster(ddgm, t=70, criterion='maxclust')
        print('nodes: ', len(set(nodes)), nodes)
        res = silhouette_score(np_distances , nodes, metric='precomputed')        
        print('Res: ', res)

    @staticmethod
    def load(file='pages_similarity_matrix.csv'):
        import codecs
        file = codecs.open(file, 'r')
        for line in file:
            break
        res = []
        for line in file:
            x = line.split(';')
            res.append(map(float, x[1:]))
        file.close()
        return res   

class VectorsPairAnalysis:
    def __init__(self, a_id, a, b_id, b):
        assert len(a) == len(b)
        self._a = a
        self._b = b
        self._a_id = a_id
        self._b_id = b_id
        self._sims = {COSINE: self.cosine_similarity,
                      CORRELATION: self.pearson_correlation,
                      JACCARD: self.jaccard_index
                      }
    def cosine_similarity(self):
        # -1 -> 1
        if sum([x > 0 for x in self._a]) == 0 or sum([x > 0 for x in self._b]) == 0:
            return 1 if self._a_id == self._b_id and not self._a_id is None  else 0
        from scipy.spatial.distance import cosine
        val = 1.0 - cosine(self._a, self._b)
        import math
        if math.isnan(val):
            print('Check: ', sum([x > 0 for x in self._a]), sum([x > 0 for x in self._b]))
            exit(1)
        return val
    def pearson_correlation(self):
        # -1 -> 1
        from scipy.stats.stats import pearsonr 
        if sum([x > 0 for x in self._a]) == 0 or sum([x > 0 for x in self._b]) == 0:
            return 1 if self._a_id == self._b_id and not self._a_id is None else 0
        val = pearsonr(self._a, self._b)[0]
        import math
        if math.isnan(val):
            print('Check: ', sum([x > 0 for x in self._a]), sum([x > 0 for x in self._b]))
            exit(1)
        return val
    def jaccard_index(self):
        from scipy.spatial.distance import jaccard
        if sum([x > 0 for x in self._a]) == 0 and sum([x > 0 for x in self._b]) == 0:
            return 1 if self._a_id == self._b_id and not self._a_id is None  else 0
        val = 1- jaccard(self._a, self._b)
        import math
        if math.isnan(val):
            print('Check: ', sum([x > 0 for x in self._a]), sum([x > 0 for x in self._b]))
            exit(1)
        return val
    def similarity(self):
        return self._sims[USED_DISTANCE]()

def case_amplification(w):
    import math
    return w*math.pow(abs(w), 1.5)

def evaluate_allbut1_item_based_recommendation(test_pages, test_users_visited_pages_ids, test_pages_vists,
                                               train_pages, train_users_visited_pages_ids, train_pages_vists):
    def get_up_votes_positions(test_pages_votes):    
        upvotes_positions = []
        for i, j in enumerate(test_pages_votes):
            if j == 1:
                upvotes_positions.append(i)
        return upvotes_positions
    assert test_pages.keys() == train_pages.keys()
    pcs = PagesSimilarityMatrix(pages=train_pages, 
                                pages_vists_only=train_pages_vists, 
                                users_visited_pages_ids=train_users_visited_pages_ids)
    train_pages_similarities = pcs.compute_matrix()
    print('Dumping pages similarity matrix ...')
    pcs.dump_matrix()
    print('Done ...')
    nbr_exact_recommendations = 0
    test_pages_ids = list(test_pages.keys())
    nbr_of_considered_users = 0
    for test_user_id, test_visited_pages_ids in test_users_visited_pages_ids.items():
        # current user votes with respect to the test pages 
        test_pages_votes = [1 if page_id in test_visited_pages_ids else 0 for page_id in test_pages_ids]
        # get user up-votes 
        upvotes_positions = get_up_votes_positions(test_pages_votes)
        # skip users if they up-voted only on page
        if len(upvotes_positions) == 1:
            continue
        nbr_of_considered_users += 1
        # apply the all but one policy
        excluded_vote_index = upvotes_positions[0]
        excluded_vote_value = test_pages_votes[excluded_vote_index]
        test_pages_votes[excluded_vote_index] = 0
        print('\n\ntest_user_id: ', test_user_id)
        print('Upvotes_positions: ', upvotes_positions)
        print('Excluding position: ', excluded_vote_index, ', vote: ', test_pages_votes[excluded_vote_index])
        # go through the user votes vector and try to identify some similar 
        # pages to up-voted ones and predict the most likely to be visited next
        
        page_similarities = []
        for i, vote in enumerate(test_pages_votes):
            page_id = test_pages_ids[i]
            if page_id not in train_pages_similarities:
                # page with 0 visits
                page_similarities.append([0]*len(test_pages_votes))
                continue
            page_similarities.append([x[1][USED_DISTANCE] for x in train_pages_similarities[page_id]])

        test_pages_votes = np.asarray(test_pages_votes)
        scores = []
        for i, vote in enumerate(test_pages_votes):
            # skip if already visited 
            if i in upvotes_positions and i != excluded_vote_index:
                scores.append(0)
                continue
            # calculate expected score to visit i
            page_id = test_pages_ids[i]
            page_pages_sim = [x[1][USED_DISTANCE] for x in train_pages_similarities[page_id]]
            score = 0
            sum = 0
            for j, sim in enumerate(page_pages_sim):
                sim = sim
                score += sim*test_pages_votes[j]
                sum += sim
            avg_score = 0
            if sum > 0:
                avg_score  = score/sum
            else:
                avg_score = 0
            scores.append(avg_score)
        best_score = max(scores) 
        recommended_page_index = scores.index(best_score)
        if recommended_page_index == excluded_vote_index:
            print('Good recommendation found')
            nbr_exact_recommendations += 1
        print('Recommendation, position: ', recommended_page_index,', page_id: ', test_pages_ids[recommended_page_index], ', score: ', best_score)
    print('\nNbr_exact_recommendations: ', nbr_exact_recommendations, ', Nbr of test cases: ', nbr_of_considered_users)
    


def evaluate_allbut1_user_based_recommendation(test_pages, test_users_visited_pages_ids, test_pages_vists,
                                               train_pages, train_users_visited_pages_ids, train_pages_vists):
    def get_up_votes_positions(test_pages_votes):    
        upvotes_positions = []
        for i, j in enumerate(test_pages_votes):
            if j == 1:
                upvotes_positions.append(i)
        return upvotes_positions
    assert test_pages.keys() == train_pages.keys()

    nbr_exact_recommendations = 0
    test_pages_ids = list(test_pages.keys())
    nbr_of_considered_users = 0
    nbr_top_n = 10
    for test_user_id, test_visited_pages_ids in test_users_visited_pages_ids.items():
        # current user votes with respect to the test pages 
        test_pages_votes = [1 if page_id in test_visited_pages_ids else 0 for page_id in test_pages_ids]

        # get user up-votes 
        upvotes_positions = get_up_votes_positions(test_pages_votes)
        # skip users if they up-voted only on page
        if len(upvotes_positions) == 1:
            continue
        nbr_of_considered_users += 1
        # apply the all but one policy
        excluded_vote_index = upvotes_positions[0]
        excluded_vote_value = test_pages_votes[excluded_vote_index]
        test_pages_votes[excluded_vote_index] = 0
        
        # calculate the similarity between the test user and all the training ones
        train_users_similarities = []
        for train_user_id, train_visited_pages_ids in train_users_visited_pages_ids.items():
            # current user votes with respect to the test pages 
            train_pages_votes = [1 if page_id in train_visited_pages_ids else 0 for page_id in test_pages_ids]
            pair_analysis = VectorsPairAnalysis(a_id=None,
                                                a=test_pages_votes, 
                                                b_id=None,
                                                b=train_pages_votes
                                                )
            sim = pair_analysis.similarity()
            if sim > 0:
                train_users_similarities.append((train_user_id, sim))
        # sort similar user by decreasing similarity
        train_users_similarities = sorted(train_users_similarities, key=lambda x:x[1], reverse=True)
        print('\nUser_id: ', test_user_id)
        print('\ntrain_users_similarities: ', train_users_similarities[:nbr_top_n])
        train_users_similarities = train_users_similarities[:nbr_top_n]
        

        # go through the user votes vector and try to identify some similar 
        # pages to up-voted ones and predict the most likely to be visited next
        scores = []
        for i, vote in enumerate(test_pages_votes):
            # skip if already visited 
            if i in upvotes_positions and i != excluded_vote_index:
                scores.append(0)
                continue
            
            # calculate expected score to visit i
            page_id = test_pages_ids[i]
            avg_score = 0
            sum = 0
            for sim_user_id, sim in train_users_similarities:
                if page_id in train_users_visited_pages_ids[sim_user_id]:
                    sum += sim
            avg_score = sum/len(train_users_similarities)
            scores.append(avg_score)
            
        best_score = max(scores)
        recommended_page_index = scores.index(best_score)
        if recommended_page_index == excluded_vote_index:
            print('Good recommendation found')
            nbr_exact_recommendations += 1
        print('Recommendation, position: ', recommended_page_index,', page_id: ', test_pages_ids[recommended_page_index], ', score: ', best_score)
    print('\nNbr_exact_recommendations: ', nbr_exact_recommendations, ', Nbr of test cases: ', nbr_of_considered_users)
    
    
def dump_sparse_matrix(pages, users_visited_pages_ids):
    import codecs
    records_file = codecs.open('sparse_matrix.csv', 'w')
    records_file.write(';%s\n'%','.join(map(str, [details.description for id, details in pages.items()])))
    for user_id, visited_pages_ids in users_visited_pages_ids.items():
        line = [user_id] + [1 if page_id in visited_pages_ids else 0 for page_id in pages.keys()]
        records_file.write('%s\n'%';'.join(map(str, line)))
    records_file.close()

def parse_and_load_ms_web_data(input_file):
    import codecs
    from functools import reduce
    file = codecs.open(input_file, 'r')
    import collections
    page = collections.namedtuple('page', ['id', 'description', 'url'])
    pages = {}
    users = {}
    current_user_pages_ids = []
    current_user_user_id = None
    users_visited_pages_ids = {}
    pages_vists = {}
    for line in file:
        chunks = line.split(',')
        type = chunks[0]
        if type == 'A':
            type, id, ignored, description, url = chunks
            pages[int(id)] = page(id=int(id), description=description, url=url)
            continue
        if type == 'C':
            if not current_user_user_id is None:
                users_visited_pages_ids[current_user_user_id] = set(current_user_pages_ids)
                current_user_pages_ids = []
            current_user_user_id = int(chunks[2])
        if type == 'V':
            page_id = int(chunks[1])
            current_user_pages_ids.append(page_id)
            pages_vists.setdefault(page_id, [])
            pages_vists[page_id].append(current_user_user_id)
    
    nbr_visits_list = map(len, users_visited_pages_ids.values())
    nbr_visits_list_ = map(len, users_visited_pages_ids.values())
    average_visits_nbr = reduce(lambda x, y: x + y, nbr_visits_list) / len(list(nbr_visits_list_))
    nbr_users_with_0_visits = sum(x == 0 for x in nbr_visits_list)
    pages_visits_list = map(len, pages_vists.values())
    pages_visits_list_ = map(len, pages_vists.values())
    average_pages_visits_nbr = reduce(lambda x, y: x + y, pages_visits_list) / len(list(pages_visits_list_))
    print('Nbr of pages: ', len(pages.keys()))
    print('Nbr of users: ', len(users_visited_pages_ids.keys()))
    print('Average nbr of visits: ', average_visits_nbr)
    print('average_pages_visits_nbr: ', average_pages_visits_nbr)
    return pages, users_visited_pages_ids, pages_vists

def transform_data(pages_visited):
    itemsets = []
    for user_pages in pages_visited.values():
        itemsets.append(set(user_pages))
    return itemsets

def get_support(itemsets, itemset):
    count = 0
    for transaction in itemsets:
        if itemset.issubset(transaction):
            count += 1
    return count

def apriori(itemsets, min_support):
    from itertools import combinations
    def generate_candidates(prev_candidates, k):
        candidates = set()
        for i, itemset1 in enumerate(prev_candidates):
            for itemset2 in prev_candidates[i+1:]:
                union_set = itemset1 | itemset2
                if len(union_set) == k and all(combo in prev_candidates for combo in combinations(union_set, k-1)):
                    candidates.add(union_set)
        return candidates

    frequent_itemsets = []
    candidates = [frozenset([item]) for item in set.union(*itemsets)]
    k = 2
    while candidates:
        frequent_candidates = []
        for candidate in candidates:
            support = get_support(candidate) / len(itemsets)
            if support >= min_support:
                frequent_itemsets.append((candidate, support))
                frequent_candidates.append(candidate)
        candidates = generate_candidates(frequent_candidates, k)
        k += 1
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    from itertools import combinations
    association_rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    confidence = support / get_support(antecedent)
                    if confidence >= min_confidence:
                        association_rules.append((antecedent, consequent, confidence))
    return association_rules

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--test-src", dest="test_src", default="/home/xkli/dm/dm2/anonymous-msweb.data",
                      help="Test data source file. Default: %default", type="string")
    parser.add_option("--train-src", dest="train_src", default="/home/xkli/dm/dm2/anonymous-msweb.data",
                      help="train data source file. Default: %default", type="string")
    parser.add_option('--item-based', dest="item_based", default=True, help='Run item based collaborative filtering', action='store_true') 
    parser.add_option('--user-based', dest="user_based", default=True, help='Run user based collaborative filtering', action='store_true')
    parser.add_option('--cluster', dest="cluster", default=False, help='Run clustering on already generated distance matrix', action='store_true')
    parser.add_option('--dump-weka-files', dest="dump_weka_files", default=False, help='Dump weka input files', action='store_true')

    options, args_left = parser.parse_args()
    
    if options.test_src is None or options.train_src is None \
      or (not options.item_based and not options.user_based and not options.cluster and not options.dump_weka_files):
        parser.print_help()
        exit(1)
    
    print('\nLoading test data')
    test_pages, test_users_visited_pages_ids, test_pages_vists = parse_and_load_ms_web_data(input_file=options.test_src)
    print('\nLoading train data')
    train_pages, train_users_visited_pages_ids, train_pages_vists = parse_and_load_ms_web_data(input_file=options.train_src)
    assert train_pages.keys() == test_pages.keys()
        
    if options.item_based:
        # item based recommendations
        print('\nRunning item based approach ...')
        evaluate_allbut1_item_based_recommendation(test_pages, test_users_visited_pages_ids, test_pages_vists,
                                                   train_pages, train_users_visited_pages_ids, train_pages_vists)
    elif options.user_based:
        # user based recommendations 
        print('\nRunning user based approach ...')
        evaluate_allbut1_user_based_recommendation(test_pages, test_users_visited_pages_ids, test_pages_vists,
                                                   train_pages, train_users_visited_pages_ids, train_pages_vists)
    elif options.cluster:
        print('\nClustering ...')
        pcs = PagesSimilarityMatrix()
        res = pcs.load()
        pcs.hierarchical_cluster(res)
    else:
        assert(False)
    
if __name__ == '__main__':
    main()