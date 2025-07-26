from typing import List, Dict, Any, Optional
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.core.config import settings
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for handling database operations."""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        drivers_to_try = [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server", 
            "ODBC Driver 13 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server"
        ]
        
        for driver in drivers_to_try:
            try:
                # Create connection string
                connection_string = (
                    f"mssql+pyodbc://{settings.DB_USER}:{settings.DB_PASSWORD}@"
                    f"{settings.DB_SERVER}/{settings.DB_DATABASE}?"
                    f"driver={driver.replace(' ', '+')}&"
                    f"TrustServerCertificate=yes&"
                    f"Encrypt=yes"
                )
                
                self.engine = create_engine(
                    connection_string,
                    echo=False,  # Set to True for debugging SQL queries
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                logger.info(f"Database service initialized successfully with driver: {driver}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect with driver {driver}: {e}")
                continue
        
        # If all drivers failed, raise the last exception
        raise Exception("Could not connect to database with any available ODBC driver. Please ensure SQL Server ODBC drivers are installed.")
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        try:
            query = text("""
                SELECT 
                    FullName,
                    IdentificationNumber,
                    BirthDate,
                    Phone,
                    Email
                FROM Patients
                ORDER BY FullName
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                patients = []
                
                for row in result:
                    patient = {
                        "full_name": row.FullName,
                        "identification_number": row.IdentificationNumber,
                        "birth_date": row.BirthDate,
                        "phone": row.Phone,
                        "email": row.Email
                    }
                    patients.append(patient)
                
                logger.info(f"Retrieved {len(patients)} patients from database")
                return patients
                
        except Exception as e:
            logger.error(f"Error retrieving patients: {e}")
            raise
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict[str, Any]]:
        try:
            query = text("""
                SELECT 
                    FullName,
                    IdentificationNumber,
                    BirthDate,
                    Phone,
                    Email
                FROM Patients
                WHERE PatientId = :patient_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"patient_id": patient_id})
                row = result.first()
                
                if row:
                    patient = {
                        "full_name": row.FullName,
                        "identification_number": row.IdentificationNumber,
                        "birth_date": row.BirthDate,
                        "phone": row.Phone,
                        "email": row.Email
                    }
                    logger.info(f"Retrieved patient with ID {patient_id}")
                    return patient
                else:
                    logger.warning(f"No patient found with ID {patient_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving patient {patient_id}: {e}")
            raise
    
    def search_patients_by_name(self, name: str) -> List[Dict[str, Any]]:
        try:
            query = text("""
                SELECT 
                    FullName,
                    IdentificationNumber,
                    BirthDate,
                    Phone,
                    Email
                FROM Patients
                WHERE FullName LIKE :name_pattern
                ORDER BY FullName
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"name_pattern": f"%{name}%"})
                patients = []
                
                for row in result:
                    patient = {
                        "full_name": row.FullName,
                        "identification_number": row.IdentificationNumber,
                        "birth_date": row.BirthDate,
                        "phone": row.Phone,
                        "email": row.Email
                    }
                    patients.append(patient)
                
                logger.info(f"Found {len(patients)} patients matching '{name}'")
                return patients
                
        except Exception as e:
            logger.error(f"Error searching patients by name '{name}': {e}")
            raise
    
    def convert_patients_to_natural_language(self, patients: List[Dict[str, Any]]) -> List[str]:
        descriptions = []
        
        for patient in patients:
            try:
                # Calculate age if birth date is available
                age_text = ""
                if patient.get("birth_date"):
                    birth_date = patient["birth_date"]
                    if isinstance(birth_date, str):
                        birth_date = datetime.strptime(birth_date, "%Y-%m-%d").date()
                    
                    today = datetime.now().date()
                    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    age_text = f", {age} years old"
                
                # Format birth date
                birth_date_text = ""
                if patient.get("birth_date"):
                    if isinstance(patient["birth_date"], str):
                        birth_date_formatted = datetime.strptime(patient["birth_date"], "%Y-%m-%d").strftime("%B %d, %Y")
                    else:
                        birth_date_formatted = patient["birth_date"].strftime("%B %d, %Y")
                    birth_date_text = f", born on {birth_date_formatted}"
                
                # Build natural language description
                description = f"Patient {patient.get('full_name', 'Name not available')}"
                
                if patient.get("identification_number"):
                    description += f" with identification number {patient['identification_number']}"
                
                description += age_text + birth_date_text
                
                if patient.get("phone"):
                    description += f", contact phone {patient['phone']}"
                
                if patient.get("email"):
                    description += f", email address {patient['email']}"
                
                description += "."
                descriptions.append(description)
                
            except Exception as e:
                logger.error(f"Error converting patient to natural language: {e}")
                # Fallback description
                descriptions.append(f"Patient {patient.get('full_name', 'Information not available')}")
        
        return descriptions
    
    def get_patients_as_natural_language(self, limit: Optional[int] = None) -> List[str]:
        try:
            patients = self.get_all_patients()
            
            if limit:
                patients = patients[:limit]
            
            return self.convert_patients_to_natural_language(patients)
            
        except Exception as e:
            logger.error(f"Error getting patients as natural language: {e}")
            raise
    
    def check_database_health(self) -> Dict[str, str]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as patient_count FROM Patients"))
                patient_count = result.first().patient_count
                
                return {
                    "status": "healthy",
                    "connection": "connected",
                    "total_patients": str(patient_count)
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e)
            }
