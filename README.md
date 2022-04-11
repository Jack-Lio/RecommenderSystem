# RecommenderSystem
A recommenderSystem implemented for MovieLens Datasets

## file structure:
```
- checkpoints       # checkpoint files
- datasets          # contains the datasets
    - train_set.csv 
    - test_set.csv
    - movies.csv
    - ...
- refs 
main.py             # the main function 
CF.py               # Collaborative Filtering 
ContentBased.py     # Content Based Filtering
dataset.py          # the dataset class
utls.py             # the utils class used in the project
config.py           # the config class used in the project
```

## train & test
train the Collaborative Filtering model
```
python main.py  MODEL.TRAINING  True MODEL.PRE_CALCULATE_SIMILARITY True  MODEL.MODEL_NAME "CF" MODEL.TEST_SAVE True
```

train the Content Based Filtering model
```
python main.py  MODEL.TRAINING  True MODEL.PRE_CALCULATE_SIMILARITY True  MODEL.MODEL_NAME "CB" MODEL.TEST_SAVE True
```
if just want to test the model, set MODEL.TRAINING False.

test the collaborative filtering model
```
python main.py --config "checkpoints/CF-2022_04_11__22_45_50/config.yaml" MODEL.TRAINING False
```

test the content based filtering model
```
python main.py --config "checkpoints/CB-2022_04_11__22_13_11/config.yaml" MODEL.TRAINING False
```

## evaluation
train time for the two models: 

- CF: 
    - train: 15 m 35 s 
    - test: 0 m  16 s
    - test result: @ SSE: 87.1135991894249
- CB:
    - train: 11 m 20 s 
    - test: 41 s
    - test result: @ SSE: 67.45523445640697
