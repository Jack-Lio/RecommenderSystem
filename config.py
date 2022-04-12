from yacs.config import CfgNode as CN

_C = CN()

# dataset config
_C.DATASET = CN()
_C.DATASET.TRAIN_SET = "./datasets/train_set.csv"
_C.DATASET.TEST_SET = "./datasets/test_set.csv"
_C.DATASET.MOVIE_SET = "./datasets/movies.csv"


# path config
_C.PATH = CN()
# _C.PATH.MODEL_PATH = "checkpoints/CF-2022_04_11__11_32_40/"
_C.PATH.MODEL_PATH = "checkpoints\CB-2022_04_11__19_21_20/"
_C.PATH.CONFIG_PATH = ""

# model config
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = "CF"  
_C.MODEL.TRAINING = True
_C.MODEL.PRE_CALCULATE_SIMILARITY = True
_C.MODEL.TEST_SAVE = True # save the test result or not 
_C.MODEL.K = 35           # the number of most similar users to consider for each user.
_C.MODEL.N_MOVIE = 50     # the number of top-recommendation movies for each user.

# model config as global config
cfg = _C