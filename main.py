from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from inspection_controller import InspectionController, create_tables

# Lifespan manager
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    await create_tables()
    yield
    # Shutdown (if needed)

# Initialize FastAPI app
app = FastAPI(
    title="Cable Inspection API",
    description="AI-powered cable inspection API with Vault & Conduit, Installation, and Deck endpoints",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize controller
controller = InspectionController()

@app.get("/")
async def root():
    return {"message": "Welcome to the Inspection API  go to /docs for more information"}

@app.post("/sealing")
async def inspect_sealing(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of sealing installation")
):
    """Analyze sealing installations for compliance with standards."""
    return await controller.inspect_sealing(prompt, image)



@app.post("/vault-flooding")
async def inspect_vault_flooding(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of vault flooding prevention installation")
):
    """Analyze vault flooding prevention measures for compliance with standards."""
    return await controller.inspect_vault_flooding(prompt, image)




@app.post("/duct-bend")
async def inspect_duct_bend(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of duct bend installation")
):
    """Analyze duct bend installations for compliance with standards."""
    return await controller.inspect_duct_bend(prompt, image)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)