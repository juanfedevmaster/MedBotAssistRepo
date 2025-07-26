from fastapi import APIRouter, HTTPException, Depends, status
from app.models.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    ConversationHistoryResponse,
    ErrorResponse
)
from app.agents.medical_agent import MedicalQueryAgent
import time
from typing import Dict, Any, Optional
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Global agent instance (in production, use proper dependency injection)
medical_agent = None

def get_medical_agent() -> MedicalQueryAgent:
    """Dependency to get or create medical agent instance."""
    global medical_agent
    if medical_agent is None:
        medical_agent = MedicalQueryAgent()
    return medical_agent

@router.post(
    "/chat",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with the medical query agent",
    description="Send natural language queries to the medical agent for patient information",
    responses={
        200: {"description": "Successful agent response"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def chat_with_agent(
    request: AgentQueryRequest,
    agent: MedicalQueryAgent = Depends(get_medical_agent)
) -> AgentQueryResponse:
    """
    Chat with the medical query agent using natural language.
    
    The agent can help you with:
    - Searching for patients by description
    - Getting patient database summaries  
    - Filtering patients by demographics
    - Answering questions about patient data
    
    Examples:
    - "Show me all male patients aged 45"
    - "Find patients with type O+ blood"
    - "How many patients do we have?"
    - "Search for patients with diabetes"
    """
    try:
        start_time = time.time()
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        
        # Process query with agent
        result = await agent.query(
            message=request.message,
            conversation_id=conversation_id
        )
        
        # Get available tools
        available_tools = [tool["name"] for tool in agent.get_available_tools()]
        
        response = AgentQueryResponse(
            response=result["response"],
            conversation_id=conversation_id,
            agent_used_tools=result.get("agent_used_tools", False),
            available_tools=available_tools,
            status=result.get("status", "success")
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Agent query processed in {processing_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing agent query: {str(e)}"
        )

@router.get(
    "/tools",
    summary="Get available agent tools",
    description="List all tools available to the medical query agent"
)
async def get_agent_tools(
    agent: MedicalQueryAgent = Depends(get_medical_agent)
) -> Dict[str, Any]:
    """
    Get a list of all tools available to the medical agent.
    """
    try:
        tools = agent.get_available_tools()
        
        return {
            "tools": tools,
            "total_tools": len(tools),
            "agent_type": "medical_query_agent",
            "capabilities": [
                "Patient search and retrieval",
                "Database statistics and summaries", 
                "Demographic filtering",
                "Natural language query processing"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent tools: {str(e)}"
        )

@router.get(
    "/conversation/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve conversation history for a specific conversation ID"
)
async def get_conversation_history(
    conversation_id: str,
    agent: MedicalQueryAgent = Depends(get_medical_agent)
) -> ConversationHistoryResponse:
    """
    Get conversation history for a specific conversation.
    """
    try:
        history = agent.get_conversation_history(conversation_id)
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=history,
            total_messages=len(history)
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation history: {str(e)}"
        )

@router.delete(
    "/conversation/{conversation_id}",
    summary="Clear conversation history",
    description="Clear conversation history for a specific conversation ID"
)
async def clear_conversation_history(
    conversation_id: str,
    agent: MedicalQueryAgent = Depends(get_medical_agent)
) -> Dict[str, Any]:
    """
    Clear conversation history for a specific conversation.
    """
    try:
        agent.clear_conversation_history(conversation_id)
        
        return {
            "message": f"Conversation history cleared for conversation {conversation_id}",
            "conversation_id": conversation_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing conversation history: {str(e)}"
        )

@router.post(
    "/health",
    summary="Check agent health",
    description="Check if the medical query agent is working properly"
)
async def check_agent_health(
    agent: MedicalQueryAgent = Depends(get_medical_agent)
) -> Dict[str, Any]:
    """
    Check the health status of the medical query agent.
    """
    try:
        # Test basic agent functionality
        test_query = "How many patients are in the database?"
        result = await agent.query(test_query, "health_check")
        
        return {
            "status": "healthy" if result["status"] == "success" else "unhealthy",
            "agent_initialized": agent.agent_executor is not None,
            "llm_configured": agent.llm is not None,
            "tools_available": len(agent.get_available_tools()),
            "test_query_successful": result["status"] == "success",
            "message": "Medical query agent is operational"
        }
        
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        return {
            "status": "unhealthy",
            "agent_initialized": False,
            "llm_configured": False,
            "tools_available": 0,
            "test_query_successful": False,
            "error": str(e),
            "message": "Medical query agent has issues"
        }

@router.post(
    "/load-sample-data",
    status_code=status.HTTP_200_OK,
    summary="Load sample patient data for testing",
    description="Load sample patient data into the vector database for testing the agent"
)
async def load_sample_data() -> Dict[str, Any]:
    """
    Load sample patient data into the vector database for testing.
    """
    try:
        from app.services.vectorization_service import VectorizationService
        
        vectorization_service = VectorizationService()
        
        # Try to get real patient data from database first
        try:
            logger.info("Attempting to load real patient data from database...")
            real_patients = vectorization_service.db_service.get_patients_as_natural_language()
            
            if real_patients and len(real_patients) > 0:
                # Use real patient data from database
                logger.info(f"Found {len(real_patients)} real patients from database")
                patient_descriptions = real_patients[:6]  # Limit to 6 for consistency
                data_source = "SQL Server Database"
            else:
                raise Exception("No real patient data found in database")
                
        except Exception as db_error:
            logger.warning(f"Could not load real patient data: {db_error}")
            logger.info("Falling back to sample patient data...")
            
            # Fallback to sample patient data (6 patients only)
            patient_descriptions = [
                "Paciente masculino de 45 años con diabetes tipo 2, peso 85kg, altura 175cm, tipo de sangre O+, presión arterial 140/90",
                "Paciente femenino de 32 años con hipertensión, peso 68kg, altura 162cm, tipo de sangre A-, presión arterial 150/95",
                "Paciente masculino de 28 años sano, peso 75kg, altura 180cm, tipo de sangre B+, presión arterial 120/80",
                "Paciente femenino de 55 años con artritis reumatoide, peso 62kg, altura 158cm, tipo de sangre AB+, toma metotrexato",
                "Paciente masculino de 38 años con asma, peso 78kg, altura 172cm, tipo de sangre O-, usa inhalador",
                "Paciente femenino de 29 años embarazada, peso 65kg, altura 165cm, tipo de sangre A+, 20 semanas de gestación"
            ]
            data_source = "Sample Data (Database connection failed)"
        
        # Load data into vector database
        await vectorization_service._vectorize_all_patients(patient_descriptions)
        
        return {
            "status": "success",
            "message": f"Successfully loaded {len(patient_descriptions)} patients into vector database",
            "patients_loaded": len(patient_descriptions),
            "data_source": data_source,
            "collection_used": "demographic_patients_namespace"
        }
        
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading sample data: {str(e)}"
        )

@router.post(
    "/refresh-patient-data",
    status_code=status.HTTP_200_OK,
    summary="Refresh patient data from database",
    description="Manually refresh vectorized patient data from SQL Server database before starting a conversation"
)
async def refresh_patient_data() -> Dict[str, Any]:
    """
    Manually refresh patient data from database.
    Use this endpoint before starting a conversation to ensure you have the latest patient data.
    """
    try:
        from app.services.vectorization_service import VectorizationService
        
        vectorization_service = VectorizationService()
        
        logger.info("Manual refresh of patient data requested...")
        
        # Get real patient data from database
        try:
            real_patients = vectorization_service.db_service.get_patients_as_natural_language()
            
            if real_patients and len(real_patients) > 0:
                # Use real patient data from database
                logger.info(f"Found {len(real_patients)} patients in database")
                patient_descriptions = real_patients
                data_source = "SQL Server Database"
            else:
                return {
                    "status": "warning",
                    "message": "No patient data found in database",
                    "patients_loaded": 0,
                    "data_source": "Empty Database"
                }
                
        except Exception as db_error:
            logger.error(f"Could not load patient data: {db_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database connection failed: {str(db_error)}"
            )
        
        # Refresh data in vector database (this will detect changes automatically)
        await vectorization_service._ensure_patient_data_in_vector_db(patient_descriptions)
        
        return {
            "status": "success",
            "message": f"Successfully refreshed {len(patient_descriptions)} patients from database",
            "patients_loaded": len(patient_descriptions),
            "data_source": data_source,
            "collection_used": "demographic_patients_namespace",
            "refresh_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing patient data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error refreshing patient data: {str(e)}"
        )
