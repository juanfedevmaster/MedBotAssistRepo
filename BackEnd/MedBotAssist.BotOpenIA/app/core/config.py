from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MedBot Assistant API"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    ALLOWED_HOSTS: List[str] = ["*"]  # Configure according to your needs
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # SQL Server Database Configuration
    DB_SERVER: str = "medbotserver.database.windows.net"
    DB_DATABASE: str = "MedBotAssistDB"
    DB_USER: str = "medicaluser"
    DB_PASSWORD: str = "Admin123!"
    DB_DRIVER: str = "ODBC Driver 18 for SQL Server"  # Try version 18 first, fallback to 17
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "medbot_documents"
    CHROMA_DEMOGRAPHIC_COLLECTION: str = "demographic_patients_namespace"
    
    # Vector Search Configuration
    VECTOR_SEARCH_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
