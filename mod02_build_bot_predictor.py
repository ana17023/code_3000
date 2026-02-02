# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        #Max depth is the main one to change, but change all of the numebbrs around to get better or worse models
        max_depth=8,
        subsample=1,
        min_samples_leaf=1,
        random_state=seed
    )
    model.fit(X, y)
    return model
