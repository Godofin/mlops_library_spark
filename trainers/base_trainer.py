from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers using PySpark.
    The sole responsibility of a trainer is to train a model.
    """
    @abstractmethod
    def train(self, df: DataFrame, params: dict = None) -> PipelineModel:
        """
        Trains a model using a Spark DataFrame.

        Args:
            df (DataFrame): The input Spark DataFrame, expected to contain 'features' and 'label' columns.
            params (dict, optional): Hyperparameters for the model.

        Returns:
            PipelineModel: The trained model, typically as a Spark ML PipelineModel.
        """
        pass
