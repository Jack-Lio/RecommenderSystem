# -*- coding = utf-8 -*-
"""
    Content based recommendation
    Calculate the similarity between the movie and the user by the content.

    the :mod `ContentBased` module provides the CB class
"""

from utls import *
from config import cfg

class CB:
    '''
    Content based recommendation.
    Top-K recommendation.
    '''

    def __init__(self, _cfg = cfg, _save_model=True):
        '''
        Initiate the Content based CF model.
        :param _cfg: the configuration of the model.
        :param n_rec_movie: the number of top-recommendation movies for each user.
        :param save_model: whether to save the model.
        '''
        self.n_rec_movie = _cfg.MODEL.N_MOVIE
        self._save_model = _save_model
        self.cfg = _cfg

        self.trainset = None
        self.testset = None
        self.user_map = None
        self.movie_map = None
        self.score_matrix = None
        self.movie_type_feature = None
        self.movie_popular = None
        self.movie_count = None
        self.movie_sim = None
        self.saver = None

    def fit(self,trainset):
        '''
        caculate the tf-idf of each movies. 
        fit the trainset by calculater the movie content similarity matrix.
        :param trainset: the trainset, which is a movie tag dictionary.
        '''
        self.saver  = ModelManager(self.cfg)
        self.trainset = trainset
    
        if not self.cfg.MODEL.TRAINING:
            self.movie_type_feature = self.saver.load_model('movie_type_feature')
            self.score_matrix = self.saver.load_model('score_matrix')
            self.movie_popular = self.saver.load_model('movie_popular')
            self.movie_sim = self.saver.load_model('movie_sim')
            self.movie_count = self.saver.load_model('movie_count')
            self.trainset = self.saver.load_model('trainset')
            print('load model successfully')
        else:
            print('model not found, training a new model')
            self.score_matrix, self.movie_popular,self.movie_sim,self.movie_count,self.user_map,self.movie_map,self.movie_type_feature = calculate_movie_similarity(self.trainset,self.cfg.MODEL.PRE_CALCULATE_SIMILARITY)
            if self._save_model:
                self.save_model()

    def save_model(self):
        """
            Save the model.
        """
        self.saver.save_config(self.cfg)
        self.saver.save_model(self.score_matrix,'movie_type_feature')
        self.saver.save_model(self.score_matrix,'score_matrix')
        self.saver.save_model(self.movie_popular,'movie_popular')
        self.saver.save_model(self.movie_sim,'movie_sim')
        self.saver.save_model(self.movie_count,'movie_count')
        self.saver.save_model(self.trainset,'trainset')
        print('save model successfully')

    def recommend(self,user):
        '''
        recommend the top-n movies for the user.

        :parm user: the user id.
        :return: the top-n movies for the user.
        '''
        n = self.n_rec_movie
        movie_sim_mat = self.movie_sim
        movie_count = self.movie_count
        score_matrix = self.score_matrix
        movie_type_feature = self.movie_type_feature

        if movie_sim_mat is None:
            raise Exception('The movie_sim_mat is None')

        movie_scores = np.zeros(movie_count)
        watched_movies_id = score_matrix[user].nonzero()[0]

        for movie_id in range(movie_count):
            watched_movies_scores = score_matrix[user][watched_movies_id]
            similarity_to_watched_movies = None
            
            if self.cfg.MODEL.PRE_CALCULATE_SIMILARITY:
                similarity_to_watched_movies = movie_sim_mat[movie_id,watched_movies_id]
            else:
                similarity_to_watched_movies = np.zeros(len(watched_movies_id))
                for i in range(len(watched_movies_id)):
                    similarity_to_watched_movies[i] = cosine_similarity(movie_type_feature[movie_id],movie_type_feature[watched_movies_id[i]])

            # make all  values non-negative, the negative value was set to 0 
            similarity_to_watched_movies = similarity_to_watched_movies.clip(min=0.0)
            # calculate the sum of the similarity of the watched movies
            all_sim = similarity_to_watched_movies.sum()
            if all_sim == 0:
                similarity_to_watched_movies = np.ones(len(watched_movies_id))/len(watched_movies_id)
            else:
                similarity_to_watched_movies = similarity_to_watched_movies/all_sim
            movie_scores[movie_id] =np.sum(watched_movies_scores * similarity_to_watched_movies)

        # filter the movies that the user has watched
        # movie_scores[watched_movies_id] = 0

        # sort the movie_scores in descending order
        # and get the top-n movies
        movie_ranks = movie_scores.argsort()[::-1]

        return movie_ranks[:n] 

    def predict(self,user,movie):
        ''''
        predict the score of the user for the movie.
        :param user: the user id.
        :param movie: the movie id.
        :return: the score of the user for the movie.
        '''
        n = self.n_rec_movie
        movie_sim_mat = self.movie_sim
        movie_count = self.movie_count
        score_matrix = self.score_matrix
        movie_type_feature = self.movie_type_feature

        if movie_sim_mat is None:
            raise Exception('The movie_sim_mat is None')

        movie_scores = np.zeros(movie_count)
        watched_movies_id = score_matrix[user].nonzero()[0]

        for movie_id in range(movie_count):
            watched_movies_scores = score_matrix[user][watched_movies_id]
            similarity_to_watched_movies = None
            
            if self.cfg.MODEL.PRE_CALCULATE_SIMILARITY:
                similarity_to_watched_movies = movie_sim_mat[movie_id,watched_movies_id]
            else:
                similarity_to_watched_movies = np.zeros(len(watched_movies_id))
                for i in range(len(watched_movies_id)):
                    similarity_to_watched_movies[i] = cosine_similarity(movie_type_feature[movie_id],movie_type_feature[watched_movies_id[i]])

            # make all  values non-negative, the negative value was set to 0 
            similarity_to_watched_movies = similarity_to_watched_movies.clip(min=0.0)
            # calculate the sum of the similarity of the watched movies
            all_sim = similarity_to_watched_movies.sum()
            if all_sim == 0:
                similarity_to_watched_movies = np.ones(len(watched_movies_id))/len(watched_movies_id)
            else:
                similarity_to_watched_movies = similarity_to_watched_movies/all_sim
            movie_scores[movie_id] =(watched_movies_scores * similarity_to_watched_movies).sum()

        return movie_scores[movie]


    def test(self,testset):
        """
        Test the model.
        :param testset: the testset, a tuple of (trainset,user_map,movie_map,type_map)
        :return: the precision and recall of the model.
        """
        self.testset = testset
        if self.movie_sim is None:
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
    cf = CB(cfg)
    cf.fit(train_set)
    re = cf.recommend(dataset.user_map_reverse[547])
    gt = np.argsort(-cf.score_matrix[1])[:50]
    print('re:',re) 
    print('gt:',gt)
    count = 0
    gt = set(gt)
    for i in re:
        if i in gt:
            count += 1
    print('count:',count)
    print(cf.predict(dataset.user_map_reverse[547],dataset.movie_map_reverse[1]))
    print(cf.predict(dataset.user_map_reverse[547],dataset.movie_map_reverse[6]))
    test_set = dataset.get_testset()
    print(cf.test(test_set))