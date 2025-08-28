from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
import uvicorn
from controller import InspectionController, create_conversation, get_conversation, get_message, get_user_logs, get_user_conversations, get_all_logs, get_validation_ledger, submit_appeal
from models import create_tables, ConversationResponse, ConversationWithMessages, MessageResponse, UserLogResponse, ValidationLedgerResponse, AppealSubmissionResponse

# Lifespan manager
import uvicorn
from controller import InspectionController, create_conversation, get_conversation, get_message
from models import create_tables, ConversationResponse, ConversationWithMessages, MessageResponse

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
    description="AI-powered cable inspection API for analyzing cable installations, including sealing, vault flooding, and duct bends.",
    version="1.0.0",
    contact={
        "name": "Support Team",
        "url": "https://example.com/support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
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


# Add tags to group endpoints logically
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Inspection API. Visit /docs for more information."}


@app.post("/create-conversation", response_model=ConversationResponse, tags=["Conversations"])
async def create_new_conversation(
    user_id: str = Form(..., description="User ID"),
    endpoint_name: str = Form(...,
                              description="Endpoint name (sealing, vault-flooding, duct-bend)")
):
    """Create a new conversation for a user and endpoint"""
    return await create_conversation(user_id, endpoint_name)


@app.get("/conversation/{conversation_id}", response_model=ConversationWithMessages, tags=["Conversations"])
async def get_conversation_details(conversation_id: str):
    """Get conversation details with all messages by conversation ID"""
    return await get_conversation(conversation_id)


@app.get("/user/{user_id}/conversations", response_model=List[ConversationResponse], tags=["Conversations"])
async def get_user_conversations_endpoint(user_id: str):
    """Get all conversations for a specific user"""
    return await get_user_conversations(user_id)


@app.get("/message/{message_id}", response_model=MessageResponse, tags=["Messages"])
async def get_message_details(message_id: str):
    """Get message details by message ID"""
    return await get_message(message_id)


@app.get("/logs/{user_id}", response_model=List[UserLogResponse], tags=["Logs"])
async def get_user_logs_endpoint(user_id: str, limit: int = 100):
    """Get user activity logs by user ID"""
    return await get_user_logs(user_id, limit)


@app.get("/logs", response_model=List[UserLogResponse], tags=["Logs"])
async def get_all_logs_endpoint(limit: int = 100, offset: int = 0):
    """Get all system logs with pagination"""
    return await get_all_logs(limit, offset)


@app.get("/validation-ledger/{conversation_id}", response_model=ValidationLedgerResponse, tags=["Validation"])
async def get_validation_ledger_endpoint(conversation_id: str):
    """Get validation ledger - all photos in a conversation with their status, confidence, and date"""
    return await get_validation_ledger(conversation_id)


@app.post("/appeal", response_model=AppealSubmissionResponse, tags=["Appeals"])
async def submit_appeal_endpoint(
    message_id: str = Form(..., description="Message ID to appeal"),
    user_id: str = Form(..., description="User ID submitting the appeal"),
    appeal_reason: str = Form(..., description="Reason for the appeal"),
    supporting_image: Optional[UploadFile] = File(
        None, description="Optional supporting image for the appeal")
):
    """Submit an appeal for a specific message with optional supporting image"""
    return await submit_appeal(message_id, user_id, appeal_reason, supporting_image)


@app.post("/sealing", tags=["Inspections"])
async def inspect_sealing(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="Image of sealing installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze sealing installations for compliance with standards."""
    return await controller.inspect_sealing(prompt, image, conversation_id)


@app.post("/vault-flooding", tags=["Inspections"])
async def inspect_vault_flooding(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="Image of vault flooding prevention installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze vault flooding prevention measures for compliance with standards."""
    return await controller.inspect_vault_flooding(prompt, image, conversation_id)


@app.post("/duct-bend", tags=["Inspections"])
async def inspect_duct_bend(
    prompt: Optional[str] = Form(None, description="Analysis prompt"),
    image: Optional[UploadFile] = File(
        None, description="Image of duct bend installation"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID")
):
    """Analyze duct bend installations for compliance with standards."""
    return await controller.inspect_duct_bend(prompt, image, conversation_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
