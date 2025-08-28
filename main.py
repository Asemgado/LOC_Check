from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from inspection_controller import InspectionController, create_tables, create_conversation, get_conversation, get_message, ConversationResponse, ConversationWithMessages, MessageResponse

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


@app.post("/create-conversation", response_model=ConversationResponse)
async def create_new_conversation(
    user_id: str = Form(..., description="User ID"),
    endpoint_name: str = Form(...,
                              description="Endpoint name (sealing, vault-flooding, duct-bend)")
):
    """Create a new conversation for a user and endpoint"""
    return await create_conversation(user_id, endpoint_name)


@app.get("/conversation/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation_details(conversation_id: str):
    """Get conversation details with all messages by conversation ID"""
    return await get_conversation(conversation_id)


@app.get("/message/{message_id}", response_model=MessageResponse)
async def get_message_details(message_id: str):
    """Get message details by message ID"""
    return await get_message(message_id)


@app.post("/sealing")
async def inspect_sealing(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of sealing installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze sealing installations for compliance with standards."""
    return await controller.inspect_sealing(prompt, image, conversation_id)


@app.post("/vault-flooding")
async def inspect_vault_flooding(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of vault flooding prevention installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze vault flooding prevention measures for compliance with standards."""
    return await controller.inspect_vault_flooding(prompt, image, conversation_id)


@app.post("/duct-bend")
async def inspect_duct_bend(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="image of duct bend installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze duct bend installations for compliance with standards."""
    return await controller.inspect_duct_bend(prompt, image, conversation_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
