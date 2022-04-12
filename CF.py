# -*- coding = utf-8 -*-
"""
    User-User base Collaborative Filtering

    the :mod `CF` module provides the CF class
"""

from utls import *
from config import cfg


class CF:
    """
    User based Collaborative Filtering.
    Top-K recommendation.
    """

    def __init__(self, _cfg = cfg,_save_model=True):
        """
        Initiate the User-User CF model.
        :param _cfg: the configuration of the model.
        :param k_sim_user: the number of most similar users to consider for each user.
        :param n_rec_movie: the number of top-recommendation movies for each user.
        :param save_model: whether to save the model.
        """
        self.k = _cfg.MODEL.K
        self.n_rec_movie = _cfg.MODEL.N_MOVIE
        self._save_model = _save_model
        self.cfg = _cfg

        self.trainset = None
        self.testset = None
        self.user_map = None
        self.movie_map = None
        self.score_matrix = None
        self.movie_popular = None
        self.movie_count = None
        self.user_sim_mat = None
        self.saver = None
        

    def fit(self, trainset):
        """
        Build the User-User CF model. 
        fit the trainset by calculate user similarity matrix.
        :param trainset: the trainset, which is a 2D array of (user_id, item_id, rating, timestamp).
        """
        self.saver = ModelManager(self.cfg)
        self.trainset = trainset

        if not self.cfg.MODEL.TRAINING:
            self.user_map = self.saver.load_model("user_map")
            self.movie_map = self.saver.load_model("movie_map")
            self.score_matrix = self.saver.load_model('score_matrix')
            self.user_sim_mat = self.saver.load_model('user_sim_mat')
            self.movie_popular = self.saver.load_model('movie_popular')
            self.movie_count = self.saver.load_model('movie_count')
            self.trainset = self.saver.load_model('trainset')
            print('load model successfully')
        else:
            print('model not found, training a new model')
            self.score_matrix, self.user_sim_mat, self.movie_popular, self.movie_count,self.user_map,self.movie_map,_ = calculate_user_sim_matrix(self.trainset,self.cfg.MODEL.PRE_CALCULATE_SIMILARITY)
            if self._save_model:
                self.save_model()

    def save_model(self):
        """
        Save the model.
        """
        self.saver.save_config(self.cfg)
        self.saver.save_model(self.user_map, "user_map")
        self.saver.save_model(self.movie_map, "movie_map")
        self.saver.save_model(self.score_matrix,'score_matrix')
        self.saver.save_model(self.user_sim_mat,'user_sim_mat')
        self.saver.save_model(self.movie_popular,'movie_popular')
        self.saver.save_model(self.movie_count,'movie_count')
        self.saver.save_model(self.trainset,'trainset')
        print('save model successfully')

    def recommend(self,user):
        '''
        Find K similar users and recommend N movies for the user.
        :param user: the user to be recommended.
        :return: the top-N recommendation for the user.
        '''
        K = self.k
        N = self.n_rec_movie
        if self.user_sim_mat is None:
            raise Exception('Error: model not found')
        user_sim_vector = None 
        if self.cfg.MODEL.PRE_CALCULATE_SIMILARITY and self.user_sim_mat is not None:
            user_sim_vector = self.user_sim_mat[user]
        else:
            user_count = len(self.user_map)
            user_sim_vector = np.zeros(user_count)
            for i in range(user_count):
                user_sim_vector[i] = pearson_sim(self.score_matrix[i],self.score_matrix[user]) 

        score_matrix = self.score_matrix

        # sort the user similarity matrix by user similarity in descending order
        # neighbors, neighbors_sim = k_neighbors(user_sim_vector,K) 
        # all_sim = np.sum(neighbors_sim)
        # if all_sim == 0:
        #     neighbors_sim = np.ones(K) / K
        # else:
        #     neighbors_sim = neighbors_sim / all_sim
        # neighbors_sim = neighbors_sim.reshape(1, -1)
        # neighbors_scores = score_matrix[neighbors].reshape(len(neighbors), -1)
        # movie_scores = np.dot(neighbors_sim,neighbors_scores)[0]

        # sort the user similarity matrix by user similarity in descending order
        # only care the user has rated the movies
        neighbors, neighbors_sim = k_neighbors(user_sim_vector,K) 
        movie_scores = np.zeros(self.movie_count)
        for i in range(self.movie_count):
            neighbors_scores = score_matrix[neighbors,i].reshape(1, -1)[0]
            mask = neighbors_scores != 0
            if np.sum(mask) == 0:   # if no one has rated the movie, then the predicted score is 0
                continue
            tmp_sim = neighbors_sim * mask
            all_sim = np.sum(tmp_sim)
            if all_sim == 0:
                tmp_sim = np.ones(mask.sum()) / mask.sum()
            else:
                tmp_sim = neighbors_sim / all_sim
            movie_scores[i] = np.sum(tmp_sim*neighbors_scores)

        # set the movies that have not been rated by the user to 0, get the mask of the movies that have been rated by the user
        mask = score_matrix[user,:] != 0
        # print("mask:",mask)
        # print("movie_scores:",movie_scores)
        movie_scores*= ~mask
        # print("movie_scores:",movie_scores)

        # sort the movies in rating in descending order
        movie_rank = np.argsort(-movie_scores)
        return movie_rank[:N]

    def predict(self,user,movie):
        '''
        Predict the rating of the user for the movie.
        :param user: the user to be predicted.
        :param movie: the movie to be predicted.
        :return: the predicted rating.
        '''
        K = self.k
        N = self.n_rec_movie
        if self.user_sim_mat is None:
            raise Exception('Error: model not found')
        user_sim_vector = None 
        if self.cfg.MODEL.PRE_CALCULATE_SIMILARITY and self.user_sim_mat is not None:
            user_sim_vector = self.user_sim_mat[user]
        else:
            user_count = len(self.user_map)
            user_sim_vector = np.zeros(user_count)
            for i in range(user_count):
                user_sim_vector[i] = pearson_sim(self.score_matrix[i],self.score_matrix[user]) 

        score_matrix = self.score_matrix

        # sort the user similarity matrix by user similarity in descending order
        neighbors, neighbors_sim = k_neighbors(user_sim_vector,K) 
        movie_scores = np.zeros(self.movie_count)
        for i in range(self.movie_count):
            neighbors_scores = score_matrix[neighbors,i].reshape(1, -1)[0]
            mask = neighbors_scores != 0
            if np.sum(mask) == 0:   # if no one has rated the movie, then the predicted score is 0
                continue
            tmp_sim = neighbors_sim * mask
            all_sim = np.sum(tmp_sim)
            if all_sim == 0:
                tmp_sim = np.ones(mask.sum()) / mask.sum()
            else:
                tmp_sim = neighbors_sim / all_sim
            movie_scores[i] = np.sum(tmp_sim*neighbors_scores)
        return movie_scores[movie]


    def test(self,testset):
        """
        Test the model.
        :param testset: the testset, a tuple of (trainset,user_map,movie_map,type_map)
        :return: the precision and recall of the model.
        """
        self.testset = testset
        if self.user_sim_mat is None:
            raise Exception('Error: model not found')
        if self.testset is None:
            raise Exception('Error: testset not found')
        true_ratings = []
        predict_ratings =[]
        test_rating, user_map, movie_map, _ = testset
        print(">>>> begin to test the model")
        begin_time = time.time()  # record the start time
        tqdm_process = tqdm(total=test_rating.shape[0])
        pred_result = ['user_id,movie_id,rating,predict_rating']
        for row in test_rating.itertuples(index=True,name="Pandas"):
            tqdm_process.update(1)
            user = user_map[getattr(row,'userId')]
            movie = movie_map[getattr(row,'movieId')]
            rating = getattr(row,'rating')
            pred_rating = self.predict(user,movie)
            true_ratings.append(rating)
            predict_ratings.append(pred_rating)
            pred_result.append([user,movie,rating,pred_rating])
        tqdm_process.close()
        print(">>>> test finished")
        end_time = time.time()  # record the end time
        print('@ time cost: '+get_time_cost(begin_time, end_time))
        if self.cfg.MODEL.TEST_SAVE:
            self.saver.save_test_result(pred_result)
        sse = SSE_error(true_ratings,predict_ratings)
        print("@ SSE:",sse)
        return sse

if __name__ == '__main__':
    from dataset import Dataset
    from config import cfg
    dataset = Dataset(cfg)
    train_set = dataset.get_trainset()
    cf = CF(cfg)
    cf.fit(train_set)
    re = cf.recommend(1)
    gt = np.argsort(-cf.score_matrix[1])[:50]
    print('re:',re) 
    print('gt:',gt)
    count = 0
    gt = set(gt)
    for i in re:
        if i in gt:
            count+=1
            print(i)
    print('count:',count)
    print(cf.predict(dataset.user_map_reverse[547],dataset.movie_map_reverse[1]))
    print(cf.predict(dataset.user_map_reverse[547],dataset.movie_map_reverse[6]))
    test_set = dataset.get_testset()
    print(cf.test(test_set))