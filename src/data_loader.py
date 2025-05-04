import pandas as pd


def load_data(filepath="data/Reviews.csv"):
    """Load review data from Excel dynamically."""
    df = pd.read_csv(filepath)
    return df
