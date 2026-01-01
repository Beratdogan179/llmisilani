import os
import json
import shutil
import re
from typing import List
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from database import engine, Base, SessionLocal, get_db
import models

from dotenv import load_dotenv
load_dotenv()

# --- AYARLAR ---
JOB_DATA_FILE = "parsed_jobs_FINAL.json"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Modeller ---
class MatchResult(BaseModel):
    job_title: str
    general_score: float
    skill_match: float
    experience_match: float
    report_summary: str

# Bu yeni model sadece CV kontrolü için
class CVValidationResult(BaseModel):
    is_cv: bool = Field(description="Is this document a CV/Resume? True or False")
    reason: str = Field(description="Why is it or why is it not a CV?")

app = FastAPI(title="HR TalentScout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None

# --- Veri Yükleme ---
def load_and_index_jobs():
    global vector_store
    docs = []
    
    # Öncelikli olarak Veritabanından (Supabase) yüklemeyi dene
    try:
        db = SessionLocal()
        db_jobs = db.query(models.JobPosting).all()
        if db_jobs:
            for job in db_jobs:
                content = f"Title: {job.title}\nDesc: {job.description}\nQuals: {job.requirements}"
                # Meta veriyi JSON formatına yakın tut
                meta = {
                    "job_title": job.title,
                    "description": job.description,
                    "qualifications_raw": job.requirements,
                    "company": job.company,
                    "location": job.location
                }
                docs.append(Document(page_content=content, metadata=meta))
            print(f"✅ Supabase'den {len(docs)} ilan yüklendi.")
    except Exception as e:
        print(f"⚠️ DB Yükleme Hatası (JSON'a geçiliyor): {e}")
    finally:
        db.close()

    # Eğer DB boşsa veya hata alındıysa JSON'dan yükle
    if not docs and os.path.exists(JOB_DATA_FILE):
        try:
            with open(JOB_DATA_FILE, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            for job in jobs_data:
                content = f"Title: {job.get('job_title', '')}\nDesc: {job.get('description', '')}\nQuals: {job.get('qualifications_raw', '')}"
                docs.append(Document(page_content=content, metadata=job))
            print(f"✅ JSON'dan {len(docs)} ilan yüklendi.")
        except Exception as e:
            print(f"❌ JSON Hata: {e}")

    if docs:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(docs, embeddings)
        print("✅ Vektör Deposu Hazır.")
    else:
        print("⚠️ Hiç ilan bulunamadı.")

@app.on_event("startup")
async def startup_event():
    load_and_index_jobs()

# --- 1. AŞAMA: CV DOĞRULAMA PROMPT'U (BEKÇİ) ---
validation_template = """
You are a Document Classifier. Determine if the text below is a **Personal CV / Resume** or something else.

TEXT CONTENT:
---
{text_sample}
---

RULES:
1. A CV **MUST** describe a PERSON (Education, Experience, Skills, Contact Info).
2. It is **NOT A CV** if it is:
   - A list of criteria or questions (e.g., "Kriter No", "Alt Soru").
   - A project plan, an article, a form, or a book excerpt.
   - A job advertisement itself.
   
OUTPUT JSON:
{{
    "is_cv": boolean,
    "reason": "Short explanation in Turkish"
}}
"""

# --- 2. AŞAMA: EŞLEŞTİRME PROMPT'U ---
matching_template = """
You are an HR AI. Match the CV to the Job.

JOB: {job_title} - {description}
CV: {cv_content}

INSTRUCTIONS:
- Score match from 0.0 to 1.0.
- Summary in Turkish.

OUTPUT JSON:
{{
    "job_title": "{job_title}",
    "general_score": 0.0,
    "skill_match": 0.0,
    "experience_match": 0.0,
    "report_summary": "..."
}}
"""

def extract_text_from_upload(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()
    with NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    text = ""
    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                text += page.page_content + "\n"
        elif ext in ["docx", "doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
        elif ext == "txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            text = docs[0].page_content
    except:
        pass
    finally:
        os.remove(tmp_path)
    
    return re.sub(r'\s+', ' ', text).strip()

@app.post("/api/match_cv", response_model=List[MatchResult])
async def match_cv(file: UploadFile = File(...)):
    
    # 1. Metni Oku
    cv_text = extract_text_from_upload(file)
    
    # --- DEBUG ---
    print(f"\n📄 Metin Uzunluğu: {len(cv_text)}")
    print(f"📄 Başlangıç: {cv_text[:100]}...\n")

    if len(cv_text) < 20:
        return [MatchResult(
            job_title="Okuma Hatası",
            general_score=0.0, skill_match=0.0, experience_match=0.0,
            report_summary="UYARI: Dosya boş veya okunamadı."
        )]

    # ---------------------------------------------------------
    # ADIM 1: BU BİR CV Mİ? (BEKÇİ KONTROLÜ)
    # ---------------------------------------------------------
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    
    validator_parser = JsonOutputParser(pydantic_object=CVValidationResult)
    validator_prompt = ChatPromptTemplate.from_template(validation_template)
    
    try:
        validator_chain = validator_prompt | llm | validator_parser
        # Sadece ilk 1000 karakter yeterli karar vermek için
        validation_res = validator_chain.invoke({
            "text_sample": cv_text[:1000], 
            "format_instructions": validator_parser.get_format_instructions()
        })
        
        print(f"🕵️ BEKÇİ KARARI: {validation_res}")

        # Eğer CV Değilse, hemen çık!
        if not validation_res['is_cv']:
            return [MatchResult(
                job_title="Geçersiz Belge",
                general_score=0.0,
                skill_match=0.0,
                experience_match=0.0,
                report_summary=f"UYARI: {validation_res['reason']} (Bu bir CV olarak algılanmadı, iş eşleşmesi yapılmadı.)"
            )]

    except Exception as e:
        print(f"Validasyon Hatası: {e}")
        # Hata olursa devam etmeyi dene (Safe fail)

    # ---------------------------------------------------------
    # ADIM 2: EŞLEŞTİRME (Sadece CV ise buraya gelir)
    # ---------------------------------------------------------
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Sistem hazır değil.")
    
    relevant_docs = vector_store.similarity_search(cv_text, k=3)
    
    results = []
    matcher_parser = JsonOutputParser(pydantic_object=MatchResult)
    matcher_prompt = ChatPromptTemplate.from_template(matching_template)
    
    for doc in relevant_docs:
        job_meta = doc.metadata
        try:
            chain = matcher_prompt | llm | matcher_parser
            res = chain.invoke({
                "job_title": job_meta.get('job_title', 'Bilinmiyor'),
                "description": job_meta.get('description', ''),
                "cv_content": cv_text[:3000],
                "format_instructions": matcher_parser.get_format_instructions()
            })
            results.append(MatchResult(**res))
        except:
            continue

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)