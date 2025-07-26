from typing import List, Dict, Any, Optional
from langchain.tools import tool
from app.services.vectorization_service import VectorizationService
import logging

logger = logging.getLogger(__name__)

# Initialize the vectorization service globally
vectorization_service = VectorizationService()

@tool
def search_patients(query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Search for patients using natural language queries.
    
    Args:
        query: Natural language query to search for patients
        top_k: Maximum number of results to return (default: 5)
        similarity_threshold: Minimum similarity threshold (default: 0.7)
    
    Examples:
    - "patients with diabetes"
    - "male patients aged 45"
    - "patients with blood type O+"
    - "young female patients"
    """
    try:
        # Use the vectorization service to search
        results = vectorization_service.search_similar_patients(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        if not results:
            return f"No patients found matching the query: '{query}'"
        
        response = f"🔍 Found {len(results)} patients matching '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            patient = result.get('metadata', {})
            
            response += f"{i}. Patient ID: {patient.get('id', 'Unknown')}\n"
            response += f"   Score: {score:.3f}\n"
            response += f"   Description: {patient.get('description', 'No description available')}\n"
            
            if patient.get('demographics'):
                demo = patient['demographics']
                response += f"   Demographics: Age {demo.get('age', 'N/A')}, "
                response += f"Gender {demo.get('gender', 'N/A')}, "
                response += f"Blood Type {demo.get('blood_type', 'N/A')}\n"
            
            response += "\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching patients: {str(e)}")
        return f"Error searching patients: {str(e)}"

@tool
def get_patient_summary(include_demographics: bool = True) -> str:
    """
    Get a summary of all patients in the database.
    
    Args:
        include_demographics: Include demographic statistics (default: True)
    
    Returns:
        A summary including total number of patients, demographic statistics, and recent additions.
    """
    try:
        summary = vectorization_service.get_patient_data_summary()
        
        response = "📊 Patient Database Summary:\n\n"
        response += f"Total Patients: {summary.get('total_patients', 'Unknown')}\n"
        
        if summary.get('patients_with_email'):
            response += f"Patients with Email: {summary['patients_with_email']}\n"
        if summary.get('patients_with_phone'):
            response += f"Patients with Phone: {summary['patients_with_phone']}\n"
        
        if include_demographics and summary.get('sample_descriptions'):
            response += "\n📝 Sample Patient Descriptions:\n"
            for i, desc in enumerate(summary['sample_descriptions'][:3], 1):
                response += f"{i}. {desc}\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting patient summary: {str(e)}")
        return f"Error getting patient summary: {str(e)}"

@tool
def filter_demographics(age_range: str = None, gender: str = None, blood_type: str = None) -> str:
    """
    Filter patients by specific demographic criteria.
    
    Args:
        age_range: Age range like '20-30' or 'young' or 'elderly'
        gender: Gender filter: 'male', 'female', 'masculine', 'feminine'
        blood_type: Blood type like 'O+', 'A-', 'AB+', etc.
    
    Returns:
        Filtered list of patients matching the demographic criteria.
    """
    try:
        # Build query based on filters
        query_parts = []
        
        if age_range:
            if age_range.lower() in ['young', 'joven']:
                query_parts.append("paciente joven")
            elif age_range.lower() in ['elderly', 'mayor', 'anciano']:
                query_parts.append("paciente mayor")
            elif '-' in age_range:
                query_parts.append(f"paciente de {age_range} años")
            else:
                query_parts.append(f"paciente de {age_range}")
        
        if gender:
            if gender.lower() in ['male', 'masculine', 'masculino', 'hombre']:
                query_parts.append("masculino")
            elif gender.lower() in ['female', 'feminine', 'femenino', 'mujer']:
                query_parts.append("femenino")
        
        if blood_type:
            query_parts.append(f"tipo de sangre {blood_type}")
        
        if not query_parts:
            return "Please provide at least one demographic filter (age_range, gender, or blood_type)."
        
        # Combine query parts
        query = " ".join(query_parts)
        
        # Search using the combined query
        results = vectorization_service.search_similar_patients(
            query=query,
            top_k=10,
            similarity_threshold=0.5
        )
        
        if not results:
            filters_str = []
            if age_range:
                filters_str.append(f"Age: {age_range}")
            if gender:
                filters_str.append(f"Gender: {gender}")
            if blood_type:
                filters_str.append(f"Blood Type: {blood_type}")
            
            return f"No patients found matching the filters: {', '.join(filters_str)}"
        
        response = f"🎯 Filtered Results for Demographics:\n"
        if age_range:
            response += f"   Age Range: {age_range}\n"
        if gender:
            response += f"   Gender: {gender}\n"
        if blood_type:
            response += f"   Blood Type: {blood_type}\n"
        
        response += f"\nFound {len(results)} matching patients:\n\n"
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            patient = result.get('metadata', {})
            
            response += f"{i}. Patient ID: {patient.get('id', 'Unknown')}\n"
            response += f"   Relevance Score: {score:.3f}\n"
            response += f"   Description: {patient.get('description', 'No description available')}\n"
            
            if patient.get('demographics'):
                demo = patient['demographics']
                response += f"   Age: {demo.get('age', 'N/A')}, "
                response += f"Gender: {demo.get('gender', 'N/A')}, "
                response += f"Blood Type: {demo.get('blood_type', 'N/A')}\n"
            
            response += "\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error filtering by demographics: {str(e)}")
        return f"Error filtering by demographics: {str(e)}"

# List of all tools for easy access
ALL_TOOLS = [search_patients, get_patient_summary, filter_demographics]
