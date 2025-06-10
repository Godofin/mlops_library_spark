from evaluators.base_evaluator import BaseEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import ClusteringEvaluator

class SparkClusteringEvaluator(BaseEvaluator):
    """
    Evaluator for clustering models using PySpark ML.
    """
    def evaluate(self, model: PipelineModel, df: DataFrame) -> dict:
        """Evaluates the clustering model."""
        predictions = model.transform(df)
        
        # The clustering evaluator in Spark needs the features column used for training
        # which is 'features_vec' as defined in the trainer.
        evaluator = ClusteringEvaluator(
            featuresCol="features_vec", 
            predictionCol="prediction"
        )
        
        silhouette = evaluator.setMetricName("silhouette").evaluate(predictions)
        
        metrics = {
            "silhouette": silhouette
        }
        return metrics
