from evaluators.base_evaluator import BaseEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

class SparkClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification models using PySpark ML.
    """
    def evaluate(self, model: PipelineModel, df: DataFrame) -> dict:
        """Evaluates the classification model."""
        predictions = model.transform(df)

        # Evaluator for AUC
        binary_evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction", 
            labelCol="label", 
            metricName="areaUnderROC"
        )
        
        # Evaluator for Accuracy, F1, etc.
        multiclass_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction", 
            labelCol="label",
            metricName="f1" 
        )
        
        auc = binary_evaluator.evaluate(predictions)
        f1_score = multiclass_evaluator.evaluate(predictions)
        
        # Change metric to accuracy
        multiclass_evaluator.setMetricName("accuracy")
        accuracy = multiclass_evaluator.evaluate(predictions)
        
        metrics = {
            "areaUnderROC": auc,
            "f1": f1_score,
            "accuracy": accuracy
        }
        return metrics
