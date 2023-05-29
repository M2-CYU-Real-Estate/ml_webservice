
from pydantic import BaseSettings

class Properties(BaseSettings):
    """A configuration class gathering all variables in '.env...' files.

    This supports '.env' file and '.env.local' files.
    """
    api_version: str = "1.0"
    regression_model_path: str
    
    class Config:
        """An inner configuration class for defining how to retrieve
        the settings attributes
        """
        env_file = 'resources/.env'
        env_file_encofing = 'utf-8'
