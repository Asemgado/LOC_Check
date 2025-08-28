from google import genai
from google.genai import types
import tempfile
import os
import uuid
import requests
import base64
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, DateTime, Float, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException

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

# Database model for storing prompts and responses


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

# Create tables function


async def create_tables():
    async with engine.begin() as conn:
        # Drop all tables with CASCADE to handle foreign key constraints
        await conn.execute(text("DROP TABLE IF EXISTS inspection_logs CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

# Initialize Gemini client globally
client = genai.Client()

# PDF path (will be uploaded when needed)
pdf_path = os.path.join(os.getcwd(), 'knowledge_base.pdf')
pdf_knowledge_base = None

# Response models


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


# System prompts for different endpoints
SEALING_PROMPT = """
You are AI validation friendly chatbot for infrastructure build QA make your answer friendly and always take the knowledge base pdf as an reference for your answer, and you check Sealing: 
ensuring that joints, closures, or chambers are sealed correctly to prevent water/dust ingress.

Sealing Validations:

Inputs: 
Field photos/video of closures, tags referencing location, technician report.

Checks:
Visual confirmation of sealing materials (heat shrink, gaskets, foam seals, etc.).
Detect gaps, incomplete sealing, or exposed cables.
Flag absence of mandatory sealant or improper closure positioning.
Outputs: Pass/Fail + annotated image highlighting seal issues.


Format your response as:

**MESSAGE:** [Response message here]
"""

VAULT_FLOODING_PROMPT = """
You are AI validation friendly chatbot for infrastructure build QA make your answer friendly and always take the knowledge base pdf as an reference for your answer, and you check Vault Bed / Flooding Detection: 

validating whether handholes/vaults are properly prepared (e.g., gravel base, drainage) and not at risk of standing water or flooding.


Vault Bed & Flood Detection:

Inputs:
Field images/video inside vault/handhole, environmental sensor data (if available), site survey metadata.

Checks:
Detect standing water, dampness, or improper drainage.
Confirm proper bedding material (e.g., gravel, sand layer, concrete base).
Identify debris/obstructions that could block drainage.
Outputs: Risk score (low/med/high) + recommendation (e.g., "Re-bed vault with gravel", "Add pump-out/drainage").

Format your response as:

**MESSAGE:** [Response message here]
"""

DUCT_BEND_PROMPT = """
You are AI validation friendly chatbot for infrastructure build QA make your answer friendly and always take the knowledge base pdf as an reference for your answer, and you check Duct Bend Installations.

Duct Bend Compliance:

Inputs: 
Photo/video of duct runs, LiDAR/AR scan (if available), blueprint reference.

Checks:
Estimate bend angles and radius from visual data.
Compare against compliance spec (e.g., ≥ 10× duct diameter or 56B bend radius).
Detect sharp bends, kinks, or crushing of conduit.
Outputs: Compliant / Non-Compliant + annotated visualization showing where bend radius fails

Format your response as:

**MESSAGE:** [Response message here]
"""

MoreStructure = '''
**VERDICT:** [APPROVED/NOT APPROVED]
**ANALYSIS:** [Brief technical analysis - max 3 sentences]
**ISSUES:** [List key issues only - max 3 points]
**RECOMMENDATIONS:** [Specific actionable recommendations - max 2 points]
**CONFIDENCE:** [Score from 0.0 to 1.0]
'''


def parse_analysis_response(response_text: str) -> ResponseStructure:
    """Parse Gemini response into structured format"""
    try:
        lines = response_text.split('\n')
        message = ""
        verdict = "UNKNOWN"
        analysis = ""
        issues = []
        recommendations = []
        confidence = 0.5

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('**MESSAGE:'):
                current_section = 'message'
                message = line.replace(
                    '**MESSAGE:', '').replace('**', '').strip()
            elif line.startswith('**VERDICT:'):
                verdict = line.replace(
                    '**VERDICT:', '').replace('**', '').strip()
            elif line.startswith('**ANALYSIS:'):
                current_section = 'analysis'
                analysis = line.replace(
                    '**ANALYSIS:', '').replace('**', '').strip()
            elif line.startswith('**ISSUES:'):
                current_section = 'issues'
            elif line.startswith('**RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line.startswith('**CONFIDENCE:'):
                try:
                    conf_text = line.replace(
                        '**CONFIDENCE:', '').replace('**', '').strip()
                    confidence = float(conf_text)
                except:
                    confidence = 0.5
                current_section = None
            elif line and current_section == 'message' and not line.startswith('**'):
                message += " " + line
            elif line and current_section == 'analysis' and not line.startswith('**'):
                analysis += " " + line
            elif line and current_section == 'issues' and not line.startswith('**'):
                if line.startswith('-') or line.startswith('•'):
                    issues.append(line[1:].strip())
                else:
                    issues.append(line)
            elif line and current_section == 'recommendations' and not line.startswith('**'):
                if line.startswith('-') or line.startswith('•'):
                    recommendations.append(line[1:].strip())
                else:
                    recommendations.append(line)

        return ResponseStructure(
            message=message.strip() if message else "Analysis completed",
            verdict=verdict,
            analysis=analysis.strip(),
            issues=[issue for issue in issues if issue.strip()],
            recommendations=[rec for rec in recommendations if rec.strip()],
            confidence_score=max(0.0, min(1.0, confidence))
        )
    except Exception as e:
        return ResponseStructure(
            message="Error occurred during analysis",
            verdict="ERROR",
            analysis=f"Failed to parse response: {str(e)}",
            issues=["System error during parsing"],
            recommendations=["Please try again or contact support"],
            confidence_score=0.0
        )


async def upload_image_to_firebase(image_content: bytes, filename: str) -> str:
    """Upload image to Firebase Storage and return the download URL"""
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_filename = f"inspection_images/{timestamp}_{filename}"

        # Firebase Storage REST API endpoint
        project_id = firebase_config["projectId"]
        api_key = firebase_config["apiKey"]

        # Upload URL
        upload_url = f"https://firebasestorage.googleapis.com/v0/b/{firebase_bucket_name}/o?name={unique_filename}"

        # Upload the file
        headers = {
            'Content-Type': 'application/octet-stream'
        }

        response = requests.post(
            upload_url,
            data=image_content,
            headers=headers,
            params={'key': api_key}
        )

        if response.status_code == 200:
            # Get the download URL
            download_url = f"https://firebasestorage.googleapis.com/v0/b/{firebase_bucket_name}/o/{unique_filename}?alt=media"
            print(f"Image uploaded successfully: {unique_filename}")
            return download_url
        else:
            print(
                f"Upload failed with status {response.status_code}: {response.text}")
            return ""

    except Exception as e:
        print(f"Error uploading image to Firebase: {str(e)}")
        return ""


async def create_conversation(user_id: str, endpoint_name: str) -> ConversationResponse:
    """Create a new conversation and return the conversation details"""
    try:
        async with AsyncSessionLocal() as db:
            conversation = Conversation(
                user_id=user_id,
                endpoint_name=endpoint_name
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)

            return ConversationResponse(
                conversation_id=str(conversation.id),
                user_id=conversation.user_id,
                endpoint_name=conversation.endpoint_name,
                created_at=conversation.created_at.isoformat()
            )
    except Exception as e:
        print(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create conversation: {str(e)}")


async def get_conversation(conversation_id: str) -> ConversationWithMessages:
    """Get conversation details with all messages by conversation ID"""
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select

            # Query the conversation by ID
            stmt = select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id))
            result = await db.execute(stmt)
            conversation = result.scalar_one_or_none()

            if not conversation:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

            # Query all messages/inspection logs for this conversation
            messages_stmt = select(Messages).where(
                Messages.conversation_id == uuid.UUID(conversation_id)
            ).order_by(Messages.created_at)
            messages_result = await db.execute(messages_stmt)
            messages = messages_result.scalars().all()

            # Convert messages to message IDs only
            message_ids = []
            for msg in messages:
                message_ids.append(str(msg.id))

            return ConversationWithMessages(
                conversation_id=str(conversation.id),
                user_id=conversation.user_id,
                endpoint_name=conversation.endpoint_name,
                created_at=conversation.created_at.isoformat(),
                messages=message_ids
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")


async def get_message(message_id: str) -> MessageResponse:
    """Get message details by message ID"""
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select

            # Query the message by ID
            stmt = select(Messages).where(
                Messages.id == uuid.UUID(message_id))
            result = await db.execute(stmt)
            message = result.scalar_one_or_none()

            if not message:
                raise HTTPException(
                    status_code=404, detail="Message not found")

            # Parse issues and recommendations from JSON strings
            try:
                issues = json.loads(message.issues) if message.issues else []
            except:
                issues = []

            try:
                recommendations = json.loads(
                    message.recommendations) if message.recommendations else []
            except:
                recommendations = []

            return MessageResponse(
                message_id=str(message.id),
                endpoint=message.endpoint,
                user_prompt=message.user_prompt or "",
                model_response=message.model_response,
                verdict=message.verdict or "",
                confidence_score=message.confidence_score or 0.0,
                analysis=message.analysis or "",
                issues=issues,
                recommendations=recommendations,
                image_url=message.image_url or "",
                created_at=message.created_at.isoformat()
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving message: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve message: {str(e)}")


async def save_to_database(endpoint: str, user_prompt: str, response: ResponseStructure, image_url: str = "", conversation_id: Optional[str] = None):
    """Save user prompt and model response to database"""
    try:
        async with AsyncSessionLocal() as db:
            log_entry = Messages(
                endpoint=endpoint,
                conversation_id=uuid.UUID(
                    conversation_id) if conversation_id else None,
                user_prompt=user_prompt,
                model_response=response.message,
                verdict=response.verdict,
                confidence_score=response.confidence_score,
                analysis=response.analysis,
                issues=json.dumps(response.issues),
                recommendations=json.dumps(response.recommendations),
                image_url=image_url if image_url else None
            )
            db.add(log_entry)
            await db.commit()
    except Exception as e:
        print(f"Error saving to database: {str(e)}")


async def image_response(image_file: UploadFile, instructions: str, prompt: str) -> ResponseStructure:
    """Analyze image using Gemini with the specified prompt"""
    try:
        # Upload PDF knowledge base if not already uploaded
        global pdf_knowledge_base
        if not pdf_knowledge_base:
            try:
                pdf_knowledge_base = client.files.upload(file=pdf_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to upload PDF knowledge base: {str(e)}")

        # Check if the uploaded file is an image
        if not image_file.content_type or not image_file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, detail="File must be an image")
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        if prompt is None:
            prompt = 'analyze this photo'

        try:
            # Upload image to Gemini
            uploaded_image = client.files.upload(file=temp_path)
            # Generate content with prompt, PDF knowledge base, and image
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, pdf_knowledge_base, uploaded_image],
                config=types.GenerateContentConfig(
                    system_instruction=instructions + MoreStructure
                )
            )

            # Parse and return response
            return parse_analysis_response(response.text if response.text else "No response")

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        return ResponseStructure(
            message="Error occurred during Image analysis",
            verdict="ERROR",
            analysis=f"Analysis failed: {str(e)}",
            issues=["System error during analysis"],
            recommendations=["Please try again or contact support"],
            confidence_score=0.0
        )


async def text_response(prompt: str, instructions: str) -> ResponseStructure:
    try:
        # Upload PDF knowledge base if not already uploaded
        global pdf_knowledge_base
        if not pdf_knowledge_base:
            try:
                pdf_knowledge_base = client.files.upload(file=pdf_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to upload PDF knowledge base: {str(e)}")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, pdf_knowledge_base],
            config=types.GenerateContentConfig(
                system_instruction=instructions
            )
        )

        return parse_analysis_response(response.text if response.text else "No response")

    except Exception as e:
        return ResponseStructure(
            message="Error occurred during text analysis",
            verdict="ERROR",
            analysis=f"Analysis failed: {str(e)}",
            issues=["System error during text analysis"],
            recommendations=["Please try again or contact support"],
            confidence_score=0.0
        )


class InspectionController:
    """Main controller class for handling inspection operations"""

    async def inspect_sealing(self, prompt: Optional[str], image: Optional[UploadFile], conversation_id: Optional[str] = None) -> ResponseStructure:
        """Analyze sealing installations for compliance with standards."""
        if not image:
            response = await text_response(prompt, SEALING_PROMPT)
            await save_to_database("sealing", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, SEALING_PROMPT, prompt)
        await save_to_database("sealing", prompt or "", response, image_url, conversation_id)
        return response

    async def inspect_vault_flooding(self, prompt: Optional[str], image: Optional[UploadFile], conversation_id: Optional[str] = None) -> ResponseStructure:
        """Analyze vault flooding prevention measures for compliance with standards."""
        if not image:
            response = await text_response(prompt, VAULT_FLOODING_PROMPT)
            await save_to_database("vault-flooding", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, VAULT_FLOODING_PROMPT, prompt)
        await save_to_database("vault-flooding", prompt or "", response, image_url, conversation_id)
        return response

    async def inspect_duct_bend(self, prompt: Optional[str], image: Optional[UploadFile], conversation_id: Optional[str] = None) -> ResponseStructure:
        """Analyze duct bend installations for compliance with standards."""
        if not image:
            response = await text_response(prompt, DUCT_BEND_PROMPT)
            await save_to_database("duct-bend", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, DUCT_BEND_PROMPT, prompt)
        await save_to_database("duct-bend", prompt or "", response, image_url, conversation_id)
        return response
