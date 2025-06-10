from mlops_lib.experiment_tracking import ExperimentTracker
from mlops_lib.model_management import ModelManager
from utils.logger import setup_logger
from pyspark.sql import DataFrame

logger = setup_logger(__name__)

class PipelineIntegrator:
    """
    Orchestrates the MLOps pipeline using PySpark, from training to registration.
    """
    def __init__(self, tracker: ExperimentTracker, manager: ModelManager):
        """
        Initializes the PipelineIntegrator.

        Args:
            tracker (ExperimentTracker): An instance of ExperimentTracker.
            manager (ModelManager): An instance of ModelManager.
        """
        self.tracker = tracker
        self.manager = manager
        logger.info("\n✅ PipelineIntegrator (PySpark version) initialized successfully.")

    def run_training_pipeline(
        self,
        trainer,
        evaluator,
        train_df: DataFrame,
        test_df: DataFrame,
        model_name: str,
        params: dict,
        model_flavor: str,
        run_name: str,
        register_threshold: float = None,
        threshold_metric: str = None
    ):
        """
        Executes a complete training and registration pipeline for a model using PySpark.
        
        Args:
            trainer: An instance of a Spark-based trainer.
            evaluator: An instance of a Spark-based evaluator.
            train_df (DataFrame): The Spark DataFrame for training.
            test_df (DataFrame): The Spark DataFrame for testing.
            model_name (str): The name for the model in the registry.
            params (dict): Hyperparameters for the model.
            model_flavor (str): The MLflow model flavor (should be 'spark').
            run_name (str): The name for this MLflow run.
            register_threshold (float, optional): The performance threshold to register the model.
            threshold_metric (str, optional): The metric to check against the threshold.
        """
        logger.info(f"\n--------------------------------------------------")
        logger.info(f"▶️  Starting PySpark pipeline for model: {model_name}")
        logger.info(f"--------------------------------------------------")
        
        with self.tracker.start_run(run_name=run_name) as run:
            # 1. Train the model
            logger.info("🧠  Training the model with PySpark...")
            model = trainer.train(train_df, params)
            logger.info("👍  Training completed.")
            
            # 2. Evaluate the model
            logger.info("📊  Evaluating the model...")
            metrics = evaluator.evaluate(model, test_df)
            logger.info(f"📈  Evaluation metrics: {metrics}")
            
            # 3. Log everything to MLflow
            logger.info("📝  Logging parameters and metrics to MLflow...")
            self.tracker.log_params(params)
            self.tracker.log_metrics(metrics)
            
            # 4. Log the model artifact
            # For Spark ML models, the model itself is a pipeline with stages (assembler, model)
            model_path = f"model-{model_flavor}"
            self.tracker.log_model(model, model_flavor, model_path)
            
            # 5. Decide if the model should be registered
            should_register = True
            if register_threshold is not None and threshold_metric is not None:
                metric_value = metrics.get(threshold_metric)
                if metric_value is None or metric_value < register_threshold:
                    should_register = False
                    logger.warning(
                        f"⚠️  Model did not meet the performance threshold. "
                        f"Metric '{threshold_metric}' was {metric_value:.4f}, "
                        f"but the threshold was {register_threshold}. Model will not be registered."
                    )

            if should_register:
                logger.info("🏆  Excellent performance! Registering model in Model Registry...")
                model_version = self.manager.register_model(
                    run_id=run.info.run_id,
                    model_path=model_path,
                    model_name=model_name
                )
                
                self.manager.transition_model_stage(
                    model_name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
            
            logger.info(f"🏁 Pipeline for model {model_name} finished.")
