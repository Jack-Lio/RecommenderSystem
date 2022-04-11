# -*- coding = utf-8 -*-
'''
    the main function of the recommendation system.
    the :mod `main` module provides the main function.
'''


from utls import *
from config import cfg
from ContentBased import CB
from CF import CF
from dataset import Dataset
import argparse

# accept the args and merge them with the config.
def main(cfg):    
    # create the model.
    if cfg.MODEL.MODEL_NAME == "CB":
        model = CB(cfg)
    elif cfg.MODEL.MODEL_NAME == "CF":
        model = CF(cfg)
    else:
        raise ValueError("The model name is not supported.")
    
    # load the trainset.
    dataset = Dataset(cfg)
    trainset = dataset.get_trainset()
    testset = dataset.get_testset()

    # fit the model.
    model.fit(trainset)

    # print the config.
    print('='*50)
    print(cfg)
    print('='*50)
    # test the model.
    model.test(testset)

    model.save_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recommendation system for movie training and testing")
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    main(cfg)