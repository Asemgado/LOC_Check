from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import tempfile
import os
import uvicorn
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Cable Inspection API",
    description="AI-powered cable inspection API with Vault & Conduit, Installation, and Deck endpoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client globally
client = genai.Client()

# Upload PDF knowledge base globally

pdf_path = os.path.join(os.getcwd(), 'knowledge_base.pdf')
pdf_knowledge_base = client.files.upload(file=pdf_path)

# Response model


class InspectionResponse(BaseModel):
    verdict: str
    analysis: str
    issues: List[str]
    recommendations: List[str]
    confidence_score: float


# System prompts for different endpoints
SEALING_PROMPT = """
You are a sealing inspection expert. Analyze the uploaded image for proper sealing installations and compliance against the PDF knowledge base.

Provide SHORT and SPECIFIC analysis. Keep responses concise and technical.

Format your response as:
**VERDICT: [APPROVED/NOT APPROVED]**
**ANALYSIS:** [Brief technical analysis - max 3 sentences]
**ISSUES:** [List key issues only - max 3 points]
**RECOMMENDATIONS:** [Specific actionable recommendations - max 2 points]
**CONFIDENCE:** [Score from 0.0 to 1.0]
"""

VAULT_FLOODING_PROMPT = """
You are a vault flooding prevention specialist. Analyze the uploaded image for vault flooding risks and protection measures compliance against the PDF knowledge base.

Provide SHORT and SPECIFIC analysis. Keep responses concise and technical.

Format your response as:
**VERDICT: [APPROVED/NOT APPROVED]**
**ANALYSIS:** [Brief technical analysis - max 3 sentences]
**ISSUES:** [List key issues only - max 3 points]
**RECOMMENDATIONS:** [Specific actionable recommendations - max 2 points]
**CONFIDENCE:** [Score from 0.0 to 1.0]
"""

DUCT_BEND_PROMPT = """
You are a duct bend installation specialist. Analyze the uploaded image for proper duct bend installations and compliance against the PDF knowledge base.

Provide SHORT and SPECIFIC analysis. Keep responses concise and technical.

Format your response as:
**VERDICT: [APPROVED/NOT APPROVED]**
**ANALYSIS:** [Brief technical analysis - max 3 sentences]
**ISSUES:** [List key issues only - max 3 points]
**RECOMMENDATIONS:** [Specific actionable recommendations - max 2 points]
**CONFIDENCE:** [Score from 0.0 to 1.0]
"""


def parse_analysis_response(response_text: str) -> InspectionResponse:
    """Parse Gemini response into structured format"""
    try:
        lines = response_text.split('\n')
        verdict = "UNKNOWN"
        analysis = ""
        issues = []
        recommendations = []
        confidence = 0.5

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('**VERDICT:'):
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

        return InspectionResponse(
            verdict=verdict,
            analysis=analysis.strip(),
            issues=[issue for issue in issues if issue.strip()],
            recommendations=[rec for rec in recommendations if rec.strip()],
            confidence_score=max(0.0, min(1.0, confidence))
        )
    except Exception as e:
        return InspectionResponse(
            verdict="ERROR",
            analysis=f"Failed to parse response: {str(e)}",
            issues=[],
            recommendations=[],
            confidence_score=0.0
        )


async def generate_response(image_file: UploadFile, prompt: str, instructions: str) -> InspectionResponse:
    """Analyze image using Gemini with the specified prompt"""
    try:
        if not pdf_knowledge_base:
            raise HTTPException(
                status_code=500, detail="PDF knowledge base not loaded")

        # if user did not upload an image
        if not image_file:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, pdf_knowledge_base],
                config=types.GenerateContentConfig(
                    system_instruction=instructions
                )
            )
            return response.txt

        if not image_file.content_type or not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Upload image to Gemini
            uploaded_image = client.files.upload(file=temp_path)

            # Generate content with prompt, PDF knowledge base, and image
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, pdf_knowledge_base, uploaded_image],
                config=types.GenerateContentConfig(
                    system_instruction=instructions
                )
            )

            # Parse and return response
            return parse_analysis_response(response.text if response.text else "No response")

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        return InspectionResponse(
            verdict="ERROR",
            analysis=f"Analysis failed: {str(e)}",
            issues=["System error during analysis"],
            recommendations=["Please try again or contact support"],
            confidence_score=0.0
        )


@app.get("/")
async def root():
    return {"message": "Welcome to the Inspection API  go to /docs for more information"}


@app.post("/sealing", response_model=InspectionResponse)
async def inspect_sealing(prompt: str, image: UploadFile = File(..., description="Image of sealing installation")):
    """Analyze sealing installations for compliance with standards."""

    return await generate_response(image, SEALING_PROMPT, prompt)


@app.post("/vault-flooding", response_model=InspectionResponse)
async def inspect_vault_flooding(prompt: str, image: UploadFile = File(..., description="Image of vault flooding prevention installation")):
    """Analyze vault flooding prevention measures for compliance with standards."""

    return await generate_response(image, VAULT_FLOODING_PROMPT, prompt)


@app.post("/duct-bend", response_model=InspectionResponse)
async def inspect_duct_bend(prompt: str, image: UploadFile = File(..., description="Image of duct bend installation")):
    """Analyze duct bend installations for compliance with standards."""

    return await generate_response(image, DUCT_BEND_PROMPT, prompt)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
