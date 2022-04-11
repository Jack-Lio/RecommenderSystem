import os
import pickle
import shutil
import numpy as np
from tqdm import tqdm
import time

from yaml import YAMLObject

class ModelManager:
    '''
    Model manager is designed to load and save all models
    No matter what dataset name.
    '''
    path_name = './checkpoints/'
    @classmethod
    def __init__(cls, cfg):
        if not cfg.MODEL.TRAINING and cfg.PATH.MODEL_PATH is not None:
            cls.path_name = cfg.PATH.MODEL_PATH
        elif cfg.MODEL.TRAINING and cfg.MODEL.MODEL_NAME:
            cls.path_name += cfg.MODEL.MODEL_NAME+"-"+ time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()) +'/'
            cfg.PATH.MODEL_PATH = cls.path_name
        else:
            raise Exception('Model path initialization error, please check your config.py')
    def save_model(self, model, model_name):
        '''
        Save model to model/ dir
        :param model: model to be saved
        :param model_name: model name
        :return: None
        '''
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists( self.path_name):
            os.makedirs(self.path_name)
        pickle.dump(model,open(self.path_name+model_name,'wb'))

    def save_config(self,cfg):
        '''
        Save config to model/ dir as yaml file
        :param cfg: config
        :return: None
        '''
        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)
        cfg.PATH.CONFIG_PATH = self.path_name+'config.yaml'
        with open(self.path_name+'config.yaml','w') as f:
            f.write(cfg.dump())

    def load_model(self, model_name):
        '''
        load model from model/ dir
        :param model_name: model name
        :return: model
        '''

        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(self.path_name+model_name):
            raise Exception('Model not found %s'%(self.path_name+model_name))
        return pickle.load(open(self.path_name+model_name,'rb'))

    def save_test_result(self,test_result):
        '''
        Save test result to model/ dir
        :param test_result: test result, as txt file
        :return: None
        '''
        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)
        with open(self.path_name+'test_result.txt','w') as f:
            for item in test_result:
                f.write(str(item)+'\n')

    @staticmethod
    def clean_workspace():
        '''
        clean model/ dir
        :return: None
        '''
        if os.path.exists('checkpoints'):
            shutil.rmtree('checkpoints')

def get_time_cost(begin_time, end_time):
    '''
    get the time cost
    :param begin_time: the start time
    :param end_time: the end time
    :return: the time cost
    '''
    time_cost = end_time - begin_time
    return "%d day %d hour %d minute %.2f second"%(time_cost // 86400, time_cost % 86400 // 3600, time_cost % 3600 // 60, time_cost % 60)


def k_neighbors(sim_vector, k):
    '''
    input the  similarity matrix, the index of the user, and the k
    return the k nearest neighbor of the user
    :param sim_vector: the similarity matrix
    :param k: the k
    :return: the k nearest neighbor of the user and the similarity between the user and the neighbor
    '''
    # get the similarity matrix
    sim_vector = sim_vector
    # get the k
    k = k
    # get the k nearest neighbor of the user
    neighbor = np.argsort(sim_vector)[-k-1:-1]
    neighbor_sim = np.sort(sim_vector)[-k-1:-1] # do not include the user itself
    return neighbor, neighbor_sim

def get_score_matrix(train_rating,user_map,movie_map):
    '''
        get the score  matrix 
        @param: train_rating, the train rating 
        @param: user_map, the user map
        @param: movie_map, the movie map
        @return: score_matrix, the movie popularity, the movie count
    '''
    print("<<<< begin to conduct the score matrix")
    score_matrix = np.zeros((len(user_map.keys()),len(movie_map.keys())))
    movie_popular = np.zeros(len(movie_map.keys()))
    movie_count = len(movie_map.keys())
    tqdm_process = tqdm(total=train_rating.shape[0])
    for row in train_rating.itertuples(index=True,name="Pandas"):
        user = user_map[getattr(row,'userId')]
        movie = movie_map[getattr(row,'movieId')]
        rate = getattr(row,'rating')
        score_matrix[user][movie] = rate
        movie_popular[movie] += 1
        tqdm_process.update(1)
    tqdm_process.close()
    print(">>>> end to conduct the score matrix")
    print("@ score matrix shape:",score_matrix.shape)
    print('movie_popular shape:',movie_popular.shape)
    print('movie_count:',movie_count)
    return score_matrix, movie_popular, movie_count

def calculate_movie_similarity(train_set,pre_sim_calcul = False):
    '''
    calculate the tfidf of the movies
    :param train_set: the train set, a tuple of (trainset,user_map,movie_map,movie_type_features)
    :return: score_matrix, movie_popular, movie_sim, movie_count
    '''
    # get the train set
    train_rating, user_map, movie_map, movie_type_features = train_set
    
    score_matrix, movie_popular, movie_count = get_score_matrix(train_rating,user_map,movie_map)
    
    movie_sim= np.zeros((movie_count, movie_count))

    if pre_sim_calcul:
        print("<<<< begin to conduct the movie similarity matrix")
        begin_time = time.time()  # record the start time
        for i in tqdm(range(movie_count)):
            movie_sim[i][i] = 1
            for j in range(i+1,movie_count):
                movie_sim[i][j] = cosine_similarity(movie_type_features[i],movie_type_features[j])
                movie_sim[j][i] = movie_sim[i][j]
        end_time = time.time()  # record the end time
        
        print(">>>> end to conduct the movie similarity matrix")
        print("@ time cost: %s"%get_time_cost(begin_time,end_time))
    else:
        print("post calculate  the similarity during prediction!")
    return score_matrix, movie_popular, movie_sim, movie_count,user_map,movie_map,movie_type_features

def cosine_similarity(list1,list2):
    '''
    calculate the cosine_similarity of list1 and list2

    :param list1: the first list
    :param list2: the second list
    :return: the cosine_similarity
    '''
    # get the number of common items
    assert(len(list1) == len(list2))
    n = len(list1) 
    assert(n > 0)
    # calculate the sum of the two lists
    sum1 = sum(list1*list2)
    # calculate the square of the two lists
    den = np.sqrt(sum(list1**2)) * np.sqrt(sum(list2**2))
    # calculate the cosine similarity
    if den == 0:
        return 0
    else:
        return sum1/den

def calculate_user_sim_matrix(train_set,pre_sim_calcul = True):
    '''
    calculate the similarity matrix between users
    :param train_set: the train set, a tuple of (trainset,user_map,movie_map,movie_type_features)
        """
    :return: the score_matrix, the similarity matrix, movie_popular, movie_count  
    '''
    # conduct the score matrix 
    print("<<<<<< begin to caculate the similarity matrix, the movie popularity and the movie count")   
   
    train_rating, user_map, movie_map, movie_type_features  = train_set
    
    score_matrix, movie_popular, movie_count = get_score_matrix(train_rating,user_map,movie_map)
    
    # get the similarity matrix between users
    user_sim_matrix = np.zeros((score_matrix.shape[0],score_matrix.shape[0])) 
    if pre_sim_calcul:
        user_sim_matrix = get_user_sim_matrix(score_matrix)
    else:
        print("post calculate  the similarity during prediction!")
    print(">>>> end to caculate the similarity matrix.")
    print('user_sim_matrix shape:',user_sim_matrix.shape)
    return score_matrix,user_sim_matrix, movie_popular, movie_count,user_map,movie_map,movie_type_features 


def get_user_sim_matrix(input_matrix):
    '''
    get the similarity matrix between users with pearson similarity
    :param input_matrix: the input matrix with shape (n_users, n_items)
    :return: the similarity matrix
    '''
    # get the shape of the input matrix
    begin_time = time.time()  # record the start time
    print("<<<< begin to get the similarity matrix")
    input_matrix = np.array(input_matrix) # convert to numpy array
    print('input score matrix shape:',input_matrix.shape)
    # get the number of users
    n_users = input_matrix.shape[0]
    # calculate the similarity matrix between users with person similarity
    user_sim_matrix = np.zeros((n_users, n_users))
    print('user_sim_matrix shape:',user_sim_matrix.shape)
    
    for i in tqdm(range(n_users)):
        user_sim_matrix[i][i] = 1
        for j in range(i+1,n_users):
            user_sim_matrix[i][j] = pearson_sim(input_matrix[i],input_matrix[j])
            user_sim_matrix[j][i] = user_sim_matrix[i][j]
    print(">>>> end to get the similarity matrix")
    end_time = time.time()  # record the end time
    print('@ time cost: '+get_time_cost(begin_time, end_time))
    return user_sim_matrix

def pearson_sim(list1,list2):
    '''
    calculate the pearson similarity between two lists
    :param list1: the first list
    :param list2: the second list
    :return: the pearson similarity
    '''
    # get the number of common items
    assert len(list1) == len(list2)
    n = len(list1) 
    assert n > 0
    # calculate the sum of the two lists
    avg1 = sum(list1)/n
    avg2 = sum(list2)/n 
    norm1 = list1 - avg1
    norm2 = list2 - avg2
    # calculate the sum of the two lists
    sum1 = sum(norm1*norm2)
    # calculate the square of the two lists
    den = np.sqrt(sum(norm1**2)) * np.sqrt(sum(norm2**2))
    # calculate the pearson similarity
    if den == 0:
        return 0.0
    else:
        return sum1/den

def SSE_error(prediction,real_rating):
    '''
    calculate the SSE error
    :param prediction: the prediction of the user
    :param real_rating: the real rating of the user
    :return: the SSE error
    '''
    # get the prediction and the real rating
    prediction = np.array(prediction)
    real_rating = np.array(real_rating)
    # calculate the SSE error
    SSE = sum((prediction - real_rating)**2)
    return SSE

if __name__ == '__main__':
    # test the similarity matrix
    from dataset import Dataset
    from config import cfg
    dataset = Dataset(cfg)
    train_set = dataset.get_trainset()
    
    a = pearson_sim(np.array([1,2,3,4,5]),np.array([1,2,3,4,5]))
    b = pearson_sim(np.array([1,2,3,4,5]),np.array([5,4,3,2,1]))
    print(a,b)
    score_matrix,user_sim_matrix, movie_popular, movie_count,user_map,movie_map = calculate_user_sim_matrix(train_set,pre_sim_calcul = False)
    pickle.dump(user_sim_matrix,open('user_map.pkl','wb'))
    pickle.dump(movie_map,open('movie_map.pkl','wb'))
    # print(user_sim_matrix, movie_popular, movie_count)
    
    # pickle.dump(train_set, open('./checkpoints\CF-2022_04_11__11_32_40/trainset.pkl', 'wb'))
    # pickle.dump(score_matrix,open('checkpoints\CF-2022_04_11__11_32_40\score_matrix.pkl','wb'))

    # test the model_manager
    # model_manager = ModelManager(cfg)
    # model_manager.clean_workspace()
    # model_manager.save_model(user_sim_matrix, 'user_sim_matrix')
    # model_manager.save_model(movie_popular, 'movie_popular')
    # model_manager.save_model(movie_count, 'movie_count')
    # d = model_manager.load_model('score_matrix')
    # a = model_manager.load_model('user_sim_mat')
    # b = model_manager.load_model('movie_popular')
    # c = model_manager.load_model('movie_count')
    # print(a[0:3],b,c,d[0:3])

    # test the time cost
    # begin_time = time.time()  # record the start time
    # time.sleep(3)
    # end_time = time.time()  # record the end time

    # # print the time cost
    # print('@ time cost:',get_time_cost(begin_time,end_time))