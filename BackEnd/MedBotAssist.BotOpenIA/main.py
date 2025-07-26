from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import vectorization
from app.api.routes import agent
from app.core.config import settings
import uvicorn

# Create FastAPI instance
app = FastAPI(
    title="MedBot Assistant API",
    description="API para asistente médico con capacidades de vectorización y agentes IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    vectorization.router,
    prefix="/api/v1/vectorization",
    tags=["vectorization"]
)

# Include agent routes
app.include_router(
    agent.router,
    prefix="/api/v1/agent",
    tags=["agent"]
)

@app.get("/")
async def root():
    return {
        "message": "MedBot Assistant API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MedBot Assistant API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
