import mlflow
from mlflow.tracking import MlflowClient
# CORREÇÃO: A anotação de tipo para ModelVersion mudou em versões recentes do MLflow
from mlflow.entities.model_registry import ModelVersion
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    """
    Gerencia o registro e o ciclo de vida do modelo no MLflow Model Registry.
    """
    def __init__(self, tracking_uri: str):
        """
        Inicializa o ModelManager.

        Args:
            tracking_uri (str): O URI para o servidor de rastreamento MLflow.
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info("Cliente MLflow inicializado para gerenciamento de modelos.")

    def register_model(self, run_id: str, model_path: str, model_name: str) -> ModelVersion:
        """
        Registra um modelo no MLflow Model Registry.

        Args:
            run_id (str): O ID da execução onde o modelo foi logado.
            model_path (str): O caminho do artefato do modelo dentro da execução.
            model_name (str): O nome sob o qual registrar o modelo.

        Returns:
            ModelVersion: O objeto da versão do modelo criado.
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        try:
            self.client.create_registered_model(model_name)
            logger.info(f"Criado novo modelo registrado: {model_name}")
        except mlflow.exceptions.RestException:
            logger.info(f"Modelo '{model_name}' ja existe no registro.")

        model_version = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Registrado modelo '{model_name}' versao {model_version.version} da execucao {run_id}.")
        return model_version

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transiciona uma versão de modelo para um novo estágio.

        Args:
            model_name (str): O nome do modelo registrado.
            version (str): A versão do modelo a ser transicionada.
            stage (str): O estágio de destino (ex: "Staging", "Production", "Archived").
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production")
        )
        logger.info(f"Modelo '{model_name}' versao {version} transicionado para o estagio '{stage}'.")

    def load_model(self, model_name: str, stage: str):
        """
        Carrega um modelo do registro com base em seu estágio.

        Args:
            model_name (str): O nome do modelo registrado.
            stage (str): O estágio do modelo a ser carregado.

        Returns:
            O objeto do modelo carregado.
        """
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Carregando modelo '{model_name}' do estagio '{stage}'...")
        # O 'flavor' deve ser conhecido, mlflow.pyfunc é uma interface genérica
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modelo carregado com sucesso.")
        return loaded_model
