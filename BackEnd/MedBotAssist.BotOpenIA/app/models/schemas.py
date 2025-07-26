from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VectorizationRequest(BaseModel):
    
    query: str = Field(
        ...,
        description="Text query to be vectorized",
        min_length=1,
        max_length=10000,
        example="¿Cuáles son los síntomas de la diabetes?"
    )
    
    collection_name: Optional[str] = Field(
        default=None,
        description="Name of the collection to search in",
        example="medical_documents"
    )
    
    top_k: Optional[int] = Field(
        default=5,
        description="Number of top similar documents to return",
        ge=1,
        le=50,
        example=5
    )
    
    similarity_threshold: Optional[float] = Field(
        default=0.7,
        description="Minimum similarity threshold for results",
        ge=0.0,
        le=1.0,
        example=0.7
    )
    
    include_metadata: Optional[bool] = Field(
        default=True,
        description="Whether to include document metadata in response"
    )

class VectorDocument(BaseModel):
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    similarity_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")

class VectorizationResponse(BaseModel):
    query: str = Field(..., description="Original query")
    embedding_model: str = Field(..., description="Embedding model used")
    documents: List[VectorDocument] = Field(..., description="Similar documents found")
    total_documents: int = Field(..., description="Total number of documents in collection")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    vector_db_status: str = Field(..., description="Vector database status")
    openai_status: str = Field(..., description="OpenAI connection status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")

# Agent Schemas
class AgentQueryRequest(BaseModel):
    message: str = Field(
        ...,
        description="Natural language message to the medical agent",
        min_length=1,
        max_length=2000,
        example="Show me all male patients aged 45"
    )
    
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to maintain context",
        example="conv_123456"
    )

class AgentQueryResponse(BaseModel):
    response: str = Field(..., description="Agent's response to the query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    agent_used_tools: bool = Field(..., description="Whether the agent used tools to answer")
    available_tools: List[str] = Field(default=[], description="List of available tools")
    status: str = Field(..., description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class ConversationHistoryResponse(BaseModel):
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total number of messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
