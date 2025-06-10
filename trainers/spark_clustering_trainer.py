from trainers.base_trainer import BaseTrainer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SparkClusteringTrainer(BaseTrainer):
    """
    Trainer for clustering models using PySpark ML.
    """
    def train(self, df: DataFrame, params: dict = None) -> PipelineModel:
        """
        Trains a K-Means model using PySpark.

        Args:
            df (DataFrame): Input DataFrame with feature columns. A 'label' column is not needed.
            params (dict, optional): Hyperparameters for the KMeans model.

        Returns:
            PipelineModel: The fitted pipeline model.
        """
        params = params or {}
        
        # For clustering, we usually don't have a label.
        feature_cols = [col for col in df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
        
        kmeans = KMeans(featuresCol="features_vec")
        
        if 'k' in params:
            kmeans.setK(params['k'])
        if 'seed' in params:
            kmeans.setSeed(params['seed'])
        if 'initSteps' in params:
            kmeans.setInitSteps(params['initSteps'])

        logger.info(f"Configured KMeans with params: {kmeans.extractParamMap()}")
            
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(df)
        
        return model
