
from datetime import datetime


class HealthService:
    """A useless class (over engineering) only for showing how to construct services 
    and use them 
    """
    
    def __init__(self, api_version: str) -> None:
        self.api_version = api_version
        
    def get_api_version(self) -> str:
        return self.api_version

    def get_current_time(self) -> datetime:
        return datetime.now()
