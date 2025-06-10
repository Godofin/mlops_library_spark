from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row

from sklearn.datasets import make_classification, make_regression, make_blobs

# Componentes da biblioteca MLOps
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from mlops_lib.experiment_tracking import ExperimentTracker
from mlops_lib.model_management import ModelManager
from mlops_lib.pipeline_integration import PipelineIntegrator

# Trainers e Evaluators para Spark
from trainers.spark_classification_trainer import SparkClassificationTrainer
from evaluators.spark_classification_evaluator import SparkClassificationEvaluator
from trainers.spark_regression_trainer import SparkRegressionTrainer
from evaluators.spark_regression_evaluator import SparkRegressionEvaluator
from trainers.spark_clustering_trainer import SparkClusteringTrainer
from evaluators.spark_clustering_evaluator import SparkClusteringEvaluator


def get_spark_session():
    """Inicializa e retorna uma SparkSession."""
    return SparkSession.builder \
        .appName("MLOps Library with PySpark Demo") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()


def main():
    """
    Função principal para demonstrar o uso da biblioteca MLOps com PySpark.
    """
    logger = setup_logger("main_script")
    logger.info("\n#########################################################")
    logger.info("\n#####     INICIANDO DEMO MLOPS COM PYSPARK      #####")
    logger.info("\n#########################################################")

    # 0. Iniciar Spark Session
    spark = get_spark_session()
    logger.info("SparkSession iniciada com sucesso.")

    # 1. Carregar Configurações
    config_loader = ConfigLoader(config_path="config/config.yaml")
    config = config_loader.load()
    mlflow_config = config['mlflow_config']
    params_config = config['model_params']

    # 2. Inicializar os componentes da biblioteca MLOps
    tracker = ExperimentTracker(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=mlflow_config['experiment_name']
    )
    manager = ModelManager(tracking_uri=mlflow_config['tracking_uri'])
    pipeline_integrator = PipelineIntegrator(tracker, manager)

    # --- Executar pipeline de classificação ---
    logger.info("\n--- Iniciando Pipeline de Classificação ---")
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=5, n_classes=2, random_state=42)
    # PySpark ML espera colunas 'label' e 'features'
    class_data = [Row(label=float(l), features=Vectors.dense(f)) for f, l in zip(X_class, y_class)]
    class_df = spark.createDataFrame(class_data)
    train_class_df, test_class_df = class_df.randomSplit([0.8, 0.2], seed=42)
    
    pipeline_integrator.run_training_pipeline(
        trainer=SparkClassificationTrainer(),
        evaluator=SparkClassificationEvaluator(),
        train_df=train_class_df,
        test_df=test_class_df,
        model_name="CreditRiskClassifierSpark",
        params=params_config['classification'],
        model_flavor="spark", # Usar o flavor do Spark
        run_name="spark_logistic_regression_run",
        register_threshold=0.85,
        threshold_metric="areaUnderROC" # Métrica do PySpark
    )

    # --- Executar pipeline de regressão ---
    logger.info("\n--- Iniciando Pipeline de Regressão ---")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=25, random_state=42)
    reg_data = [Row(label=float(l), features=Vectors.dense(f)) for f, l in zip(X_reg, y_reg)]
    reg_df = spark.createDataFrame(reg_data)
    train_reg_df, test_reg_df = reg_df.randomSplit([0.8, 0.2], seed=42)

    pipeline_integrator.run_training_pipeline(
        trainer=SparkRegressionTrainer(),
        evaluator=SparkRegressionEvaluator(),
        train_df=train_reg_df,
        test_df=test_reg_df,
        model_name="HousePriceRegressorSpark",
        params=params_config['regression'],
        model_flavor="spark",
        run_name="spark_linear_regression_run",
        register_threshold=0.70,
        threshold_metric="r2" # Métrica do PySpark
    )

    # --- Executar pipeline de clusterização ---
    logger.info("\n--- Iniciando Pipeline de Clusterização ---")
    X_clust, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)
    # Para clusterização, não precisamos da coluna 'label' no treino
    clust_data = [Row(features=Vectors.dense(f)) for f in X_clust]
    clust_df = spark.createDataFrame(clust_data)
    train_clust_df, test_clust_df = clust_df.randomSplit([0.8, 0.2], seed=42)

    pipeline_integrator.run_training_pipeline(
        trainer=SparkClusteringTrainer(),
        evaluator=SparkClusteringEvaluator(),
        train_df=train_clust_df,
        test_df=test_clust_df, # O mesmo df pode ser usado para teste
        model_name="CustomerSegmentationSpark",
        params=params_config['clustering'],
        model_flavor="spark",
        run_name="spark_kmeans_clustering_run",
        register_threshold=0.5,
        threshold_metric="silhouette" # Métrica do PySpark
    )
    
    logger.info("\n#########################################################")
    logger.info("\n#####            DEMONSTRAÇÃO FINALIZADA            #####")
    logger.info("\n#########################################################")

    # 9. Parar a sessão Spark
    spark.stop()

if __name__ == "__main__":
    main()
