from trainers.base_trainer import BaseTrainer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SparkRegressionTrainer(BaseTrainer):
    """
    Trainer for regression models using PySpark ML.
    """
    def train(self, df: DataFrame, params: dict = None) -> PipelineModel:
        """
        Trains a Linear Regression model using PySpark.

        Args:
            df (DataFrame): Input DataFrame with a 'label' column and feature columns.
            params (dict, optional): Hyperparameters for the LinearRegression model.

        Returns:
            PipelineModel: The fitted pipeline model.
        """
        params = params or {}
        
        feature_cols = [col for col in df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
        
        lr = LinearRegression(featuresCol="features_vec", labelCol="label")

        if 'regParam' in params:
            lr.setRegParam(params['regParam'])
        if 'elasticNetParam' in params:
            lr.setElasticNetParam(params['elasticNetParam'])
        if 'maxIter' in params:
            lr.setMaxIter(params['maxIter'])
            
        logger.info(f"Configured LinearRegression with params: {lr.extractParamMap()}")
        
        pipeline = Pipeline(stages=[assembler, lr])
        model = pipeline.fit(df)
        
        return model
