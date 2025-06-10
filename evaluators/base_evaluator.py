from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators using PySpark.
    """
    @abstractmethod
    def evaluate(self, model: PipelineModel, df: DataFrame) -> dict:
        """
        Evaluates the trained model on a Spark DataFrame.

        Args:
            model (PipelineModel): The trained Spark ML PipelineModel.
            df (DataFrame): The Spark DataFrame for evaluation.

        Returns:
            dict: A dictionary of performance metrics.
        """
        pass
