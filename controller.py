from google import genai
from google.genai import types
import tempfile
import os
import uuid
import requests
import base64
import json
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from datetime import datetime, timezone
from sqlalchemy import select

# Import from models
from models import (
    AsyncSessionLocal, Conversation, Messages, UserLogs, Appeals, firebase_config, firebase_bucket_name,
    ResponseStructure, ConversationResponse, MessageResponse, ConversationWithMessages, UserLogResponse,
    ValidationLedgerResponse, ValidationLedgerItem, AppealResponse, AppealSubmissionResponse
)

# Initialize Gemini client globally
client = genai.Client()

# PDF path (will be uploaded when needed)
pdf_path = os.path.join(os.getcwd(), 'knowledge_base.pdf')
pdf_knowledge_base = None

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


async def log_user_action(
    user_id: str,
    action: str,
    endpoint: Optional[str] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    request_data: Optional[dict] = None,
    response_data: Optional[dict] = None,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_ms: Optional[float] = None
):
    """Log user action in the system"""
    try:
        async with AsyncSessionLocal() as db:
            log_entry = UserLogs(
                user_id=user_id,
                action=action,
                endpoint=endpoint,
                conversation_id=uuid.UUID(
                    conversation_id) if conversation_id else None,
                message_id=uuid.UUID(message_id) if message_id else None,
                request_data=json.dumps(
                    request_data) if request_data else None,
                response_data=json.dumps(
                    response_data) if response_data else None,
                status=status,
                error_message=error_message,
                ip_address=ip_address,
                user_agent=user_agent,
                duration_ms=duration_ms
            )
            db.add(log_entry)
            await db.commit()
    except Exception as e:
        print(f"Error logging user action: {str(e)}")


async def get_user_logs(user_id: str, limit: int = 100) -> List[UserLogResponse]:
    """Get user logs by user ID"""
    try:
        async with AsyncSessionLocal() as db:
            # Query user logs by user ID
            stmt = select(UserLogs).where(
                UserLogs.user_id == user_id
            ).order_by(UserLogs.created_at.desc()).limit(limit)
            result = await db.execute(stmt)
            logs = result.scalars().all()

            # Convert logs to response format
            log_responses = []
            for log in logs:
                log_responses.append(UserLogResponse(
                    log_id=str(log.id),
                    user_id=log.user_id,
                    action=log.action,
                    endpoint=log.endpoint or "",
                    conversation_id=str(
                        log.conversation_id) if log.conversation_id else None,
                    message_id=str(log.message_id) if log.message_id else None,
                    request_data=log.request_data,
                    response_data=log.response_data,
                    status=log.status,
                    error_message=log.error_message,
                    ip_address=log.ip_address,
                    user_agent=log.user_agent,
                    duration_ms=log.duration_ms,
                    created_at=log.created_at.isoformat()
                ))

            return log_responses
    except Exception as e:
        print(f"Error retrieving user logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve user logs: {str(e)}")


async def get_all_logs(limit: int = 100, offset: int = 0) -> List[UserLogResponse]:
    """Get all logs from the system with pagination"""
    try:
        async with AsyncSessionLocal() as db:
            # Query all logs with pagination
            stmt = select(UserLogs).order_by(
                UserLogs.created_at.desc()
            ).limit(limit).offset(offset)
            result = await db.execute(stmt)
            logs = result.scalars().all()

            # Convert logs to response format
            log_responses = []
            for log in logs:
                log_responses.append(UserLogResponse(
                    log_id=str(log.id),
                    user_id=log.user_id,
                    action=log.action,
                    endpoint=log.endpoint or "",
                    conversation_id=str(
                        log.conversation_id) if log.conversation_id else None,
                    message_id=str(log.message_id) if log.message_id else None,
                    request_data=log.request_data,
                    response_data=log.response_data,
                    status=log.status,
                    error_message=log.error_message,
                    ip_address=log.ip_address,
                    user_agent=log.user_agent,
                    duration_ms=log.duration_ms,
                    created_at=log.created_at.isoformat()
                ))

            return log_responses
    except Exception as e:
        print(f"Error retrieving all logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve all logs: {str(e)}")


async def get_validation_ledger(conversation_id: str):
    """Get all photos in a conversation with their status, confidence, and date"""
    try:
        async with AsyncSessionLocal() as db:
            # First verify the conversation exists
            conversation_stmt = select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id)
            )
            conversation_result = await db.execute(conversation_stmt)
            conversation = conversation_result.scalar_one_or_none()

            if not conversation:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

            # Query all messages with photos in this conversation
            messages_stmt = select(Messages).where(
                Messages.conversation_id == uuid.UUID(conversation_id),
                Messages.image_url.isnot(None),
                Messages.image_url != ""
            ).order_by(Messages.created_at)

            messages_result = await db.execute(messages_stmt)
            messages = messages_result.scalars().all()

            # Convert to response format
            photos = []
            for msg in messages:
                photos.append({
                    "message_id": str(msg.id),
                    "image_url": msg.image_url,
                    "status": msg.verdict or "Unknown",
                    "confidence_score": msg.confidence_score or 0.0,
                    "endpoint": msg.endpoint,
                    "created_at": msg.created_at.isoformat()
                })

            return {
                "conversation_id": conversation_id,
                "total_photos": len(photos),
                "photos": photos
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving validation ledger: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve validation ledger: {str(e)}")


async def submit_appeal(message_id: str, user_id: str, appeal_reason: str, supporting_image: Optional[UploadFile] = None) -> AppealSubmissionResponse:
    """Submit an appeal for a specific message with optional supporting image"""
    try:
        async with AsyncSessionLocal() as db:
            # First verify the message exists and belongs to the user
            message_stmt = select(Messages).where(
                Messages.id == uuid.UUID(message_id))
            message_result = await db.execute(message_stmt)
            message = message_result.scalar_one_or_none()

            if not message:
                raise HTTPException(
                    status_code=404, detail="Message not found")

            # Get conversation to verify user ownership
            conversation_stmt = select(Conversation).where(
                Conversation.id == message.conversation_id)
            conversation_result = await db.execute(conversation_stmt)
            conversation = conversation_result.scalar_one_or_none()

            if not conversation or conversation.user_id != user_id:
                raise HTTPException(
                    status_code=403, detail="You can only appeal your own messages")

            supporting_image_url = None

            # Handle supporting image upload if provided
            if supporting_image and supporting_image.filename:
                try:
                    import firebase_admin
                    from firebase_admin import credentials, storage

                    # Initialize Firebase if not already done
                    if not firebase_admin._apps:
                        # Use service account key or default credentials
                        firebase_admin.initialize_app()

                    # Upload image to Firebase Storage
                    bucket = storage.bucket(firebase_bucket_name)
                    blob_name = f"appeals/{uuid.uuid4()}_{supporting_image.filename}"
                    blob = bucket.blob(blob_name)

                    # Read and upload file content
                    file_content = await supporting_image.read()
                    blob.upload_from_string(
                        file_content, content_type=supporting_image.content_type)

                    # Make the blob public and get URL
                    blob.make_public()
                    supporting_image_url = blob.public_url

                except Exception as upload_error:
                    print(
                        f"Error uploading supporting image: {str(upload_error)}")
                    # Continue without the image rather than failing the entire appeal
                    supporting_image_url = None

            # Create the appeal
            appeal = Appeals(
                message_id=uuid.UUID(message_id),
                user_id=user_id,
                appeal_reason=appeal_reason,
                supporting_image_url=supporting_image_url,
                status="pending"
            )

            db.add(appeal)
            await db.commit()
            await db.refresh(appeal)

            return AppealSubmissionResponse(
                appeal_id=str(appeal.id),
                message="Appeal submitted successfully",
                status="pending"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error submitting appeal: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit appeal: {str(e)}")


async def create_conversation(user_id: str, endpoint_name: str) -> ConversationResponse:
    """Create a new conversation and return the conversation details"""
    start_time = datetime.now(timezone.utc)
    try:
        async with AsyncSessionLocal() as db:
            conversation = Conversation(
                user_id=user_id,
                endpoint_name=endpoint_name
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)

            response = ConversationResponse(
                conversation_id=str(conversation.id),
                user_id=conversation.user_id,
                endpoint_name=conversation.endpoint_name,
                created_at=conversation.created_at.isoformat()
            )

            # Log the action
            duration = (datetime.now(timezone.utc) -
                        start_time).total_seconds() * 1000
            await log_user_action(
                user_id=user_id,
                action="create_conversation",
                endpoint="/create-conversation",
                conversation_id=str(conversation.id),
                request_data={"user_id": user_id,
                              "endpoint_name": endpoint_name},
                response_data={"conversation_id": str(conversation.id)},
                status="SUCCESS",
                duration_ms=duration
            )

            return response
    except Exception as e:
        # Log the error
        duration = (datetime.now(timezone.utc) -
                    start_time).total_seconds() * 1000
        await log_user_action(
            user_id=user_id,
            action="create_conversation",
            endpoint="/create-conversation",
            request_data={"user_id": user_id, "endpoint_name": endpoint_name},
            status="ERROR",
            error_message=str(e),
            duration_ms=duration
        )
        print(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create conversation: {str(e)}")


async def get_conversation(conversation_id: str) -> ConversationWithMessages:
    """Get conversation details with all messages by conversation ID"""
    try:
        async with AsyncSessionLocal() as db:
            # Query the conversation by ID
            stmt = select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id))
            result = await db.execute(stmt)
            conversation = result.scalar_one_or_none()

            if not conversation:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

            # Query all messages for this conversation
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


async def get_user_conversations(user_id: str) -> List[ConversationResponse]:
    """Get all conversations for a specific user"""
    try:
        async with AsyncSessionLocal() as db:
            # Query all conversations for the user
            stmt = select(Conversation).where(
                Conversation.user_id == user_id
            ).order_by(Conversation.created_at.desc())
            result = await db.execute(stmt)
            conversations = result.scalars().all()

            # Convert to response format
            conversation_responses = []
            for conv in conversations:
                conversation_responses.append(ConversationResponse(
                    conversation_id=str(conv.id),
                    user_id=conv.user_id,
                    endpoint_name=conv.endpoint_name,
                    created_at=conv.created_at.isoformat()
                ))

            return conversation_responses
    except Exception as e:
        print(f"Error retrieving user conversations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve user conversations: {str(e)}")


async def get_message(message_id: str) -> MessageResponse:
    """Get message details by message ID"""
    try:
        async with AsyncSessionLocal() as db:
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
            response = await text_response(prompt or "", SEALING_PROMPT)
            await save_to_database("sealing", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, SEALING_PROMPT, prompt or "")
        await save_to_database("sealing", prompt or "", response, image_url, conversation_id)
        return response

    async def inspect_vault_flooding(self, prompt: Optional[str], image: Optional[UploadFile], conversation_id: Optional[str] = None) -> ResponseStructure:
        """Analyze vault flooding prevention measures for compliance with standards."""
        if not image:
            response = await text_response(prompt or "", VAULT_FLOODING_PROMPT)
            await save_to_database("vault-flooding", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, VAULT_FLOODING_PROMPT, prompt or "")
        await save_to_database("vault-flooding", prompt or "", response, image_url, conversation_id)
        return response

    async def inspect_duct_bend(self, prompt: Optional[str], image: Optional[UploadFile], conversation_id: Optional[str] = None) -> ResponseStructure:
        """Analyze duct bend installations for compliance with standards."""
        if not image:
            response = await text_response(prompt or "", DUCT_BEND_PROMPT)
            await save_to_database("duct-bend", prompt or "", response, "", conversation_id)
            return response

        # Upload image to Firebase
        image_content = await image.read()
        image_url = await upload_image_to_firebase(image_content, image.filename or "image.jpg")

        # Reset file pointer for analysis
        await image.seek(0)
        response = await image_response(image, DUCT_BEND_PROMPT, prompt or "")
        await save_to_database("duct-bend", prompt or "", response, image_url, conversation_id)
        return response
