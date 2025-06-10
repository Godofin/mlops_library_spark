import yaml
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ConfigLoader:
    """
    A class to load configurations from a YAML file.
    """
    def __init__(self, config_path: str):
        """
        Initializes the ConfigLoader.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = None

    def load(self) -> dict:
        """
        Loads the YAML configuration file.

        Returns:
            dict: A dictionary containing the configuration.
        
        Raises:
            FileNotFoundError: If the config file is not found.
            Exception: For other loading errors.
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise
