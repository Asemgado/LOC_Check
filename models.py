import os
import uuid
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, DateTime, Float, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Firebase setup
firebase_config = {
    "apiKey": os.getenv("VITE_FIREBASE_apiKey", "").strip('"'),
    "authDomain": os.getenv("VITE_FIREBASE_authDomain", "").strip('"'),
    "projectId": os.getenv("VITE_FIREBASE_projectId", "").strip('"'),
    "storageBucket": os.getenv("VITE_FIREBASE_storageBucket", "").strip('"'),
    "messagingSenderId": os.getenv("VITE_FIREBASE_messagingSenderId", "").strip('"'),
    "appId": os.getenv("VITE_FIREBASE_appId", "").strip('"'),
    "measurementId": os.getenv("VITE_FIREBASE_measurementId", "").strip('"')
}

firebase_bucket_name = firebase_config["storageBucket"]

# Database setup
DATABASE_URL = os.getenv("DATABASE_CONNECTION_STRING")
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://")

engine = create_async_engine(DATABASE_URL or '')
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

# Database models


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    endpoint_name = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Messages(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey(
        'conversations.id'), nullable=True)
    endpoint = Column(String(50), nullable=False)
    user_prompt = Column(Text)
    model_response = Column(Text, nullable=False)
    verdict = Column(String(20))
    confidence_score = Column(Float)
    analysis = Column(Text)
    issues = Column(Text)  # Store as JSON string
    recommendations = Column(Text)  # Store as JSON string
    image_url = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserLogs(Base):
    __tablename__ = "user_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    # Action type (e.g., "create_conversation", "inspect_sealing", etc.)
    action = Column(String(100), nullable=False)
    endpoint = Column(String(50), nullable=True)  # API endpoint called
    conversation_id = Column(UUID(as_uuid=True), ForeignKey(
        'conversations.id'), nullable=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey(
        'messages.id'), nullable=True)
    # Store request parameters as JSON
    request_data = Column(Text, nullable=True)
    response_data = Column(Text, nullable=True)  # Store response data as JSON
    status = Column(String(20), nullable=False)  # SUCCESS, ERROR, FAILED
    # Error details if status is ERROR/FAILED
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # User's IP address
    user_agent = Column(Text, nullable=True)  # User's browser/client info
    # Request processing time in milliseconds
    duration_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Create tables function
async def create_tables():
    async with engine.begin() as conn:
        # Drop all tables with CASCADE to handle foreign key constraints
        await conn.execute(text("DROP TABLE IF EXISTS inspection_logs CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS user_logs CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


# Pydantic models for API responses
class ResponseStructure(BaseModel):
    message: str
    verdict: str
    analysis: str
    issues: List[str]
    recommendations: List[str]
    confidence_score: float


class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    endpoint_name: str
    created_at: str


class MessageResponse(BaseModel):
    message_id: str
    endpoint: str
    user_prompt: str
    model_response: str
    verdict: str
    confidence_score: float
    analysis: str
    issues: List[str]
    recommendations: List[str]
    image_url: str
    created_at: str


class ConversationWithMessages(BaseModel):
    conversation_id: str
    user_id: str
    endpoint_name: str
    created_at: str
    messages: List[str]  # Array of message IDs only


class UserLogResponse(BaseModel):
    log_id: str
    user_id: str
    action: str
    endpoint: str
    conversation_id: Optional[str]
    message_id: Optional[str]
    request_data: Optional[str]
    response_data: Optional[str]
    status: str
    error_message: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    duration_ms: Optional[float]
    created_at: str
