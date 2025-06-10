from trainers.base_trainer import BaseTrainer
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback
import mlflow

# Custom Keras callback to log metrics to MLflow
class MLflowMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mlflow.log_metrics(logs, step=epoch)

class DeepLearningTrainer(BaseTrainer):
    """
    Trainer for a simple deep learning model (feed-forward network).
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        """Trains a simple Keras sequential model."""
        params = params or {}
        epochs = params.get("epochs", 5)
        batch_size = params.get("batch_size", 32)
        
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1) # Output layer for regression, change for classification
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[MLflowMetricsCallback()],
            verbose=0 # Set to 1 or 2 to see progress
        )
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the Keras model."""
        loss = model.evaluate(X_test, y_test, verbose=0)
        return {"test_loss": loss}

