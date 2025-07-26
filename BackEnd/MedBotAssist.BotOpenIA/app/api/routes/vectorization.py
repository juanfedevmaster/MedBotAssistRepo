from fastapi import APIRouter, HTTPException, Depends, status
from app.models.schemas import (
    VectorizationRequest,
    VectorizationResponse,
    VectorDocument,
    ErrorResponse,
    HealthResponse
)
from app.services.vectorization_service import VectorizationService
from app.core.config import settings
import time
from typing import Dict, Any

router = APIRouter()

# Dependency to get vectorization service
def get_vectorization_service() -> VectorizationService:
    return VectorizationService()

@router.post(
    "/search",
    response_model=VectorizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Search similar patient data using vector similarity",
    description="Vectorize a query and search for similar patient descriptions from the database",
    responses={
        200: {"description": "Successful vectorization and search"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def vectorize_and_search(
    request: VectorizationRequest,
    vectorization_service: VectorizationService = Depends(get_vectorization_service)
) -> VectorizationResponse:
    """
    Vectorize a query and search for similar patient data.
    
    This endpoint:
    1. Queries the SQL Server database for patient information
    2. Converts patient data to natural language descriptions
    3. Stores the descriptions in ChromaDB as vectors
    4. Takes a text query and converts it to vector embeddings using OpenAI
    5. Searches for similar patient descriptions in ChromaDB
    6. Returns the most relevant patient information with similarity scores
    """
    try:
        start_time = time.time()
        
        # Perform vectorization and search with patient data
        result = await vectorization_service.vectorize_and_search(
            query=request.query,
            top_k=request.top_k or 5,
            similarity_threshold=request.similarity_threshold or 0.7,
            collection_name=request.collection_name
        )
        
        # Format response
        response = VectorizationResponse(
            query=request.query,
            embedding_model=result["embedding_model"],
            documents=[
                VectorDocument(
                    id=doc["id"],
                    content=doc["content"],
                    similarity_score=doc["similarity_score"],
                    metadata=doc["metadata"] if request.include_metadata else None
                )
                for doc in result["documents"]
            ],
            total_documents=result["total_documents"],
            search_time_ms=result["search_time_ms"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during vectorization: {str(e)}"
        )

@router.post(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check vectorization service health",
    description="Check the health status of vectorization components including database connectivity"
)
async def check_vectorization_health(
    vectorization_service: VectorizationService = Depends(get_vectorization_service)
) -> HealthResponse:
    """
    Check the health of vectorization components.
    
    Validates:
    - Vector database connection
    - OpenAI API connectivity
    - SQL Server database connection
    - Service status
    """
    try:
        health_status = vectorization_service.check_health()
        
        return HealthResponse(
            status=health_status.get("vectorization_service", "unknown"),
            vector_db_status=health_status.get("chromadb_connection", "unknown"),
            openai_status=health_status.get("openai_connection", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get(
    "/collections",
    summary="List available collections",
    description="Get list of available vector collections"
)
async def list_collections(
    vectorization_service: VectorizationService = Depends(get_vectorization_service)
) -> Dict[str, Any]:
    """
    List all available vector collections.
    """
    try:
        collections = vectorization_service.list_collections()
        
        return {
            "collections": collections,
            "total_collections": len(collections)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )

@router.get(
    "/patients/summary",
    summary="Get patient data summary",
    description="Get summary of patient data from the SQL Server database"
)
async def get_patient_summary(
    vectorization_service: VectorizationService = Depends(get_vectorization_service)
) -> Dict[str, Any]:
    """
    Get a summary of patient data from the database.
    
    Returns:
    - Total number of patients
    - Statistics about contact information
    - Sample patient descriptions in natural language
    """
    try:
        summary = await vectorization_service.get_patient_data_summary()
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting patient summary: {str(e)}"
        )
