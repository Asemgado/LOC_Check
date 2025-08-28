# Cable Inspection App

A Streamlit application that uses Google Gemini AI to analyze cable and slack images against approval criteria from uploaded PDF documents.

---

## Table of Contents

- [Features](#features)

- [Setup Instructions](#setup-instructions)

- [How to Use](#how-to-use)

- [Requirements](#requirements)

- [File Structure](#file-structure)

- [API Examples](#api-examples)

- [Contribution Guidelines](#contribution-guidelines)

---

## Features

- 🔌 Upload and analyze cable/slack images

- 📄 PDF knowledge base integration for approval criteria

- 🤖 AI-powered analysis using Google Gemini

- ✅ Clear approval/rejection verdicts with detailed explanations

- 📊 Multiple image batch processing

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in the application

### 3. Run the Application

```bash
streamlit run app.py
```

---

## How to Use

1. **Configure API Key**: Enter your Google Gemini API key in the sidebar

2. **Upload Knowledge Base**: Upload a PDF file containing cable approval criteria

3. **Upload Images**: Select cable or slack images you want to analyze

4. **Analyze**: Click the "Analyze Images" button to get AI-powered inspection results

---

## Requirements

- Python 3.8+

- Streamlit

- Google Generative AI SDK

- PyPDF2

- Pillow (PIL)

---

## File Structure

```
LOC_Check/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## API Examples

### Create a Conversation

```bash
curl -X POST \
  -F "user_id=12345" \
  -F "endpoint_name=sealing" \
  http://localhost:8000/create-conversation
```

### Get Conversation Details

```bash
curl -X GET http://localhost:8000/conversation/{conversation_id}
```

### Analyze Sealing Installation

```bash
curl -X POST \
  -F "prompt=Check sealing compliance" \
  -F "image=@path/to/image.jpg" \
  http://localhost:8000/sealing
```

---

## Contribution Guidelines

We welcome contributions! Please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Commit your changes with clear messages.

4. Submit a pull request.

---
