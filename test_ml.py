import pytest
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_model_type():
    """
    # This test checks that the model used in train_model is the anticipated type
    """
    # Create fake data for testing
    X_train = np.random.rand(115, 50)
    y_train = np.random.randint(2, size=115)

    # Use train_model on the fake data
    model = train_model(X_train, y_train)

    # Check if the model is of the correct type
    assert isinstance(model, LogisticRegression)


def test_predictions():
    """
    # This test checks that the inference function performs as expected
    """
    # Create fake data for testing
    X_train = np.random.rand(115, 50)
    y_train = np.random.randint(2, size=115)

    # Use train_model on the fake data
    model = train_model(X_train, y_train)

    # Make predictions
    preds = inference(model, X_train)

    # Check that predictions are as expected
    assert set(preds) <= {0, 1}


def test_metrics():
    """
    # This test checks that compute_model_metrics returns values of expected type
    """
    # Create fake data for testing
    # Create fake data for testing
    X_train = np.random.rand(115, 50)
    y_train = np.random.randint(2, size=115)

    # Use train_model on the fake data
    model = train_model(X_train, y_train)

    # Make predictions
    preds = inference(model, X_train)

    # Compute metrics
    p, r, fb = compute_model_metrics(y_train, preds)

    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(fb, float) 

