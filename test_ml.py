import pytest
# TODO: add necessary import
from model.py import train_model, inference, compute_model_metrics
from sklearn.datasets import make_classification

# TODO: implement the first test. Change the function name and input as needed
def model_type_test():
    """
    # This test checks that the model used in train_model is the anticipated type
    """
    # Generate random data for classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=2, random_state=33)

    # Use train_model on the data
    model = train_model(X_dummy, y_dummy)

    # Check if the model is of the correct type
    assert isinstance(model, LogisticRegression)


# TODO: implement the second test. Change the function name and input as needed
def predictions_test():
    """
    # This test checks that the inference function performs as expected
    """
    # Create data
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=2, random_state=33)

    # Train the model
    model = train_model(X_dummy, y_dummy)

    # Make predictions
    preds = inference(model, X_dummy)

    # Check that predictions are as expected
    assert set(preds) <= {0, 1}


# TODO: implement the third test. Change the function name and input as needed
def test_metrics():
    """
    # This test checks that compute_model_metrics returns expected values
    """
    # Create data
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=2, random_state=33)

    # Train the model
    model = train_model(X_dummy, y_dummy)

    # Make predictions
    preds = inference(model, X_dummy)

    # Compute metrics
    p, r, fb = compute_model_metrics(y_dummy, preds)

    assert p > 0.5 
    assert r > 0.5
    assert fb > 0.5 

