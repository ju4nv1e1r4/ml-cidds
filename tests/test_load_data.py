import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_load_dataset(load):
    df = load.load_dataset()
    assert isinstance(df, pd.DataFrame)

def test_split_dataset(load):
    df = load.load_dataset()
    X_train, X_test, y_train, y_test = load.split_dataset(df)

    print(f"Tipo de X_train: {type(X_train)}")
    print(f"Tipo de X_test: {type(X_test)}")
    print(f"Tipo de y_train: {type(y_train)}")
    print(f"Tipo de y_test: {type(y_test)}")

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)