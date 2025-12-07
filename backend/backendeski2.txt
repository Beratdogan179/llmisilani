import os
import io
import time
import json
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Pydantic Models ---
class MatchResult(BaseModel):
    job_title: str
    general_score: float
    skill_match: float
    experience_match: float
    report_summary: str

# --- FastAPI Setup ---
app = FastAPI(title="LLM CV Matching API")

origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM and Vector Store Setup ---
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) 
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    MOCK_JOB_ADS = [
        "İlan Başlığı: Kıdemli Python Geliştiricisi\nGereksinimler: Minimum 5 yıl tecrübe, FastAPI, PostgreSQL, AWS Cloud bilgisi. Takım liderliği deneyimi tercih sebebidir.",
        "İlan Başlığı: Veri Analisti\nGereksinimler: İstatistik, SQL ve Python Pandas bilgisi. R dili ve Power BI tecrübesi zorunludur. Deneyim: 3 yıl.",
        "İlan Başlığı: Mobil Uygulama Geliştiricisi\nGereksinimler: Flutter veya React Native tecrübesi. Backend bilgisi önemsizdir. UI/UX bilgisi beklenir.",
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = text_splitter.create_documents(MOCK_JOB_ADS)
    
    vector_store = FAISS.from_documents(all_chunks, embedding_model)
    retriever = vector_store.as_retriever()
    
except Exception as e:
    print(f"ERROR: LLM setup failed: {e}")
    llm = None
    retriever = None

# IMPROVED PROMPT - More specific instructions
prompt_template = """You are an AI analyst matching a candidate's CV with job ads.

Analyze the CV against EACH job ad below and return results in STRICT JSON format.

CV CONTENT:
---
{cv_content}
---

JOB ADS:
---
{context}
---

CRITICAL: Return ONLY a JSON array with this EXACT structure (no extra text, no markdown):
[
  {{
    "job_title": "exact job title from ad",
    "general_score": 0.85,
    "skill_match": 0.80,
    "experience_match": 0.90,
    "report_summary": "Brief summary of match quality, strengths and gaps"
  }}
]

Rules:
- Scores must be between 0.0 and 1.0
- Include one object per job ad
- All fields are required
- Return valid JSON only
"""

PROMPT = ChatPromptTemplate.from_template(prompt_template)

# --- Helper Functions ---

def parse_document(file: UploadFile):
    """Read uploaded file and return text content."""
    with NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        file_content = file.file.read()
        temp_file.write(file_content)
        temp_path = temp_file.name

    text_content = ""
    try:
        file_extension = temp_path.split('.')[-1].lower()

        if file_extension == 'pdf':
            loader = PyPDFLoader(temp_path)
        elif file_extension in ['txt']:
            loader = TextLoader(temp_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file format. Use PDF or TXT.")
        
        documents = loader.load()
        text_content = "\n\n".join(doc.page_content for doc in documents)

    except Exception as e:
        print(f"ERROR: File reading failed: {e}")
        text_content = "File reading error. Please check file format."
    finally:
        os.remove(temp_path)
    
    return text_content

def format_docs(docs):
    """Format retrieved documents as single text."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def parse_llm_json_response(response_text: str):
    """Parse LLM response and clean JSON."""
    # Remove markdown code blocks if present
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw response: {response_text[:500]}")
        raise

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "LLM CV Matching API is running"}

@app.post("/api/match_cv", response_model=list[MatchResult])
async def match_cv(file: UploadFile = File(...)):
    if not llm or not retriever:
        raise HTTPException(
            status_code=503, 
            detail="LLM service not ready. Check API key."
        )
        
    start_time = time.time()
    
    # 1. Parse CV to text
    cv_text = parse_document(file)
    if "error" in cv_text.lower():
        raise HTTPException(status_code=400, detail=cv_text)
        
    # 2. Retrieve relevant jobs
    relevant_jobs = retriever.invoke(cv_text)
    
    # 3. Build RAG chain with String output
    rag_chain = (
        {
            "context": lambda x: format_docs(x['relevant_jobs']), 
            "cv_content": lambda x: x['cv_content']
        }
        | PROMPT
        | llm
        | StrOutputParser()  # Get string output first
    )
    
    try:
        # Get LLM response as string
        llm_response = rag_chain.invoke({
            "relevant_jobs": relevant_jobs, 
            "cv_content": cv_text
        })
        
        print(f"LLM Raw Response: {llm_response[:300]}")
        
        # Parse JSON manually
        results = parse_llm_json_response(llm_response)
        
        # Ensure it's a list
        if not isinstance(results, list):
            results = [results]
        
        # Validate with Pydantic
        validated_results = []
        for res in results:
            try:
                validated_results.append(MatchResult(**res))
            except Exception as validation_error:
                print(f"Validation error for result: {res}")
                print(f"Error: {validation_error}")
                # Add default values if missing
                validated_results.append(MatchResult(
                    job_title=res.get('job_title', 'Unknown Position'),
                    general_score=res.get('general_score', 0.0),
                    skill_match=res.get('skill_match', 0.0),
                    experience_match=res.get('experience_match', 0.0),
                    report_summary=res.get('report_summary', 'Analysis incomplete')
                ))
        
    except Exception as e:
        print(f"LLM Output Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get valid JSON from LLM: {str(e)}"
        )

    end_time = time.time()
    print(f"INFO: Matching completed in {end_time - start_time:.2f} seconds")
    
    return validated_results