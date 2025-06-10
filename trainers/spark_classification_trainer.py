from trainers.base_trainer import BaseTrainer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SparkClassificationTrainer(BaseTrainer):
    """
    Trainer for classification models using PySpark ML.
    """
    def train(self, df: DataFrame, params: dict = None) -> PipelineModel:
        """
        Trains a Logistic Regression model using PySpark.

        Args:
            df (DataFrame): Input DataFrame. Must contain a 'label' column and feature columns.
            params (dict, optional): Hyperparameters for the LogisticRegression model.

        Returns:
            PipelineModel: The fitted pipeline model.
        """
        params = params or {}
        
        # PySpark ML requires features to be in a single vector column.
        # We assume the input DataFrame has separate feature columns and we assemble them.
        feature_cols = [col for col in df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
        
        # Set up the model
        lr = LogisticRegression(featuresCol="features_vec", labelCol="label")
        
        # Map and set parameters
        # This is where you map your config params to Spark's param setters
        if 'regParam' in params:
            lr.setRegParam(params['regParam'])
        if 'elasticNetParam' in params:
            lr.setElasticNetParam(params['elasticNetParam'])
        if 'maxIter' in params:
            lr.setMaxIter(params['maxIter'])

        logger.info(f"Configured LogisticRegression with params: {lr.extractParamMap()}")

        # Create and fit the pipeline
        pipeline = Pipeline(stages=[assembler, lr])
        model = pipeline.fit(df)
        
        return model
