"""Robust ML pipeline with logging and error handling.

A single Iris classification pipeline (load -> validate -> preprocess ->
split -> train -> evaluate -> predict) where every stage is both logged
and guarded by error handling. Successful steps are logged at INFO level;
failures are caught and logged at ERROR level so the pipeline fails
gracefully instead of crashing.
"""

#Step 1: Configure logging
import logging

# Log INFO and above to both a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

#Step 2: Imports
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#Step 3: Load the dataset
def load_data():
    """Load the Iris dataset as a pandas DataFrame."""
    try:
        logger.info("Loading dataset...")
        iris = load_iris(as_frame=True)
        logger.info("Dataset loaded successfully (%d rows).", len(iris.frame))
        return iris
    except Exception as e:
        logger.error("Error while loading dataset: %s", e)
        raise


#Step 4: Validate the input data
def validate_data(data):
    """Check that the input is a non-empty DataFrame with no missing values."""
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Dataset is empty.")
        if data.isnull().values.any():
            raise ValueError("Missing values detected in the dataset.")
        logger.info("Data validation successful.")
        return True
    except ValueError as e:
        logger.error("Data validation error: %s", e)
        return False


#Step 5: Preprocess and split
def preprocess_and_split(iris):
    """Fill any missing values and split into train/test sets."""
    try:
        logger.info("Starting data preprocessing...")
        X = iris.data.fillna(0)
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


#Step 6: Train the model
def train_model(X_train, y_train):
    """Train a decision tree classifier with error handling."""
    try:
        logger.info("Starting model training...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except ValueError as e:
        logger.error("Model training error: %s", e)
        return None


#Step 7: Evaluate and predict
def evaluate_and_predict(model, X_test, y_test):
    """Score the model and log a small sample of predictions."""
    try:
        logger.info("Starting model predictions...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info("Test accuracy: %.2f", accuracy)
        # In production, limit the amount of data logged.
        logger.info("Prediction sample: %s", list(predictions[:5]))
        return predictions
    except Exception as e:
        logger.error("Error during predictions: %s", e)
        return None


#Step 8: Run the full pipeline
def run_pipeline():
    """Execute the full pipeline end to end."""
    iris = load_data()

    if not validate_data(iris.frame):
        logger.error("Aborting pipeline: data validation failed.")
        return

    X_train, X_test, y_train, y_test = preprocess_and_split(iris)

    model = train_model(X_train, y_train)
    if model is None:
        logger.error("Aborting pipeline: model training failed.")
        return

    evaluate_and_predict(model, X_test, y_test)
    logger.info("Pipeline completed successfully.")


#Step 9: Demonstrate error handling and logging on bad input
def demonstrate_failure():
    """Introduce a missing value to show validation catching and logging it."""
    logger.info("Running failure demonstration on corrupted data...")
    iris = load_iris(as_frame=True)
    df_with_missing = iris.frame.copy()
    df_with_missing.iloc[0, 0] = None
    validate_data(df_with_missing)  # logs an ERROR instead of crashing


if __name__ == "__main__":
    run_pipeline()
    demonstrate_failure()
