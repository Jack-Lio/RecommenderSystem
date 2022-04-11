# -*- coding = utf-8 -*-
"""
    the :mod `dataset` module provides the dataset class
    and other subclasses which are used for managing datasets
"""
import pandas as pd
import numpy as np

class Dataset:
    """base class for loading datasets
    
    Note that you should never instantiate the class :class: `Dataset` class directly,
    and just use the below available methods for loading datasets.
    """

    def __init__(self,cfg):
        self._cfg = cfg
        
        self.movies = pd.read_csv(self._cfg.DATASET.MOVIE_SET)
        self.ratings_train = pd.read_csv(self._cfg.DATASET.TRAIN_SET)
        self.ratings_test = pd.read_csv(self._cfg.DATASET.TEST_SET)

        self.user_list_train = self.ratings_train['userId'].drop_duplicates().values.tolist()
        self.user_list_test = self.ratings_test['userId'].drop_duplicates().values.tolist()
        self.user_list = self.user_list_train + self.user_list_test
        self.user_list = list(set(self.user_list))

        self.movie_list = self.movies['movieId'].drop_duplicates().values.tolist()
        self.genre_list = self.movies['genres'].values.tolist()
        self.movie_type_list = self.get_movie_type_list(self.genre_list)

        self.user_map_train, self.user_map_reverse_train = self.get_list_index_map(self.user_list_train)
        self.user_map_test, self.user_map_reverse_test = self.get_list_index_map(self.user_list_test)
        self.type_map, self.type_map_reverse = self.get_list_index_map(self.movie_type_list)
        self.user_map, self.user_map_reverse = self.get_list_index_map(self.user_list)
        self.movie_map, self.movie_map_reverse = self.get_list_index_map(self.movie_list)

        self.movie_type_features = self.get_movie_type_features(self.movies)
    
    def get_movie_type_features(self,movies):
        """
        get the movie type features, tf-idf matrix
        """
        movie_type_features = np.zeros((len(self.movie_list),len(self.movie_type_list)))
        for row in self.movies.itertuples(index=True,name="Pandas"):
            movie_id = self.movie_map[getattr(row,'movieId')]
            movie_types = getattr(row,'genres').split('|')
            for movie_type in movie_types:
                if movie_type != '(no genres listed)':
                    movie_type_index = self.type_map[movie_type]
                    movie_type_features[movie_id,movie_type_index] = 1
        return movie_type_features # tfidf matri

    def get_list_index_map(self,list):
        """
        get the index map of a list
        """
        index_map = {}
        index_map_reverse = {}
        for i,item in enumerate(list):
            index_map[item] = i
            index_map_reverse[i] = item
        return index_map, index_map_reverse


    def get_movie_type_list(self,genres_list):
        """
        get the movie type list
        """
        movie_type_list = []
        for item in genres_list:
            movie_types = item.split('|')
            for movie_type in movie_types:
                if movie_type not in movie_type_list and movie_type != '(no genres listed)':
                    movie_type_list.append(movie_type)
        return movie_type_list

    def get_trainset(self):
        """
        get the trainset
        @return: (trainset,user_map,movie_map,type_map)
        """
        return (self.ratings_train,self.user_map,self.movie_map,self.movie_type_features)

    def get_testset(self):
        """
        get the testset
        @return: (testset,user_map,movie_map,type_map)
        """
        return (self.ratings_test,self.user_map,self.movie_map,self.movie_type_features)



if __name__ == '__main__':
    from config import cfg
    dataset = Dataset(cfg)
    # print(dataset.user_list)
    # print(dataset.movie_list)
    # print(dataset.movie_type_list)
    print(dataset.type_map)
    print(dataset.type_map_reverse)
    # print(dataset.user_map)
    # print(dataset.user_map_reverse)
    # print(dataset.movie_map)
    # print(dataset.movie_map_reverse)

    # genres type list 
    print(dataset.type_map.keys())
    # ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'Documentary', 'IMAX', 'War', 'Musical', 'Western', 'Film-Noir']
