import pandas as pd
from sklearn.datasets import fetch_openml

def load_data():
    # Load Adult dataset
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    return df