import mlflow
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperimentTracker:
    """
    Manages experiment tracking with MLflow.
    """
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the ExperimentTracker.

        Args:
            tracking_uri (str): The URI for the MLflow tracking server.
            experiment_name (str): The name of the experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        logger.info(f"MLflow experiment set to: {self.experiment_name}")

    def start_run(self, run_name: str = None):
        """
        Starts a new MLflow run.

        Args:
            run_name (str, optional): An optional name for the run.

        Returns:
            mlflow.ActiveRun: The active run object.
        """
        logger.info(f"Starting MLflow run: {run_name or 'Untitled'}")
        return mlflow.start_run(run_name=run_name)

    def log_param(self, key: str, value):
        """Logs a single parameter."""
        mlflow.log_param(key, value)
        logger.debug(f"Logged param: {key}={value}")

    def log_params(self, params: dict):
        """Logs multiple parameters from a dictionary."""
        mlflow.log_params(params)
        logger.debug(f"Logged params: {params}")

    def log_metric(self, key: str, value: float):
        """Logs a single metric."""
        mlflow.log_metric(key, value)
        logger.debug(f"Logged metric: {key}={value}")

    def log_metrics(self, metrics: dict):
        """Logs multiple metrics from a dictionary."""
        mlflow.log_metrics(metrics)
        logger.debug(f"Logged metrics: {metrics}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Logs a local file or directory as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact from {local_path} to {artifact_path or ''}")
    
    def log_model(self, model, flavor: str, artifact_path: str):
        """
        Logs a model using a specific MLflow model flavor.

        Args:
            model: The model object to log.
            flavor (str): The MLflow model flavor (e.g., 'sklearn', 'keras').
            artifact_path (str): The path within the run to save the model.
        """
        log_func = getattr(mlflow, flavor).log_model
        log_func(model, artifact_path)
        logger.info(f"Logged model as flavor '{flavor}' to path '{artifact_path}'")


    def end_run(self):
        """Ends the current MLflow run."""
        mlflow.end_run()
        logger.info("MLflow run finished.")
