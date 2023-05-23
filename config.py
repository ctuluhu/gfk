import os


SEED = 0

LEARNING_PARAMS = {
        'objective': 'binary:logistic',
        'n_estimators': 2000,
        'max_depth': 6,
        'learning_rate': 0.5,
        'min_child_weight': 4,
        'scale_pos_weight': 0.08,
        'seed': SEED,
}

TEST_DATASET_PATH = os.path.join('data', 'validation.csv')
TRAIN_DATASET_PATH = os.path.join('data', 'training.csv')
