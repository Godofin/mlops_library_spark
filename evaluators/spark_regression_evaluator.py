from evaluators.base_evaluator import BaseEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

class SparkRegressionEvaluator(BaseEvaluator):
    """
    Evaluator for regression models using PySpark ML.
    """
    def evaluate(self, model: PipelineModel, df: DataFrame) -> dict:
        """Evaluates the regression model."""
        predictions = model.transform(df)
        
        evaluator = RegressionEvaluator(
            predictionCol="prediction", 
            labelCol="label"
        )
        
        # Calculate multiple metrics
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        mse = evaluator.setMetricName("mse").evaluate(predictions)
        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        mae = evaluator.setMetricName("mae").evaluate(predictions)
        
        metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
        return metrics
