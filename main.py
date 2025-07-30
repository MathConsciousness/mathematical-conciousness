"""FastAPI application for Mathematical Consciousness Framework

Main application entry point with API server setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.endpoints import router


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Mathematical Consciousness Framework",
        description="Advanced scientific computing framework for mathematical consciousness",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Mathematical Consciousness Framework API",
            "version": "0.1.0",
            "status": "operational",
            "endpoints": {
                "docs": "/docs",
                "api": "/api/v1",
                "system_status": "/api/v1/system/status",
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "mathematical-consciousness"}
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )