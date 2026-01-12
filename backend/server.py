import os
import json
import shutil
import re
import asyncio
from typing import List
from tempfile import NamedTemporaryFile

import uvicorn
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

# Eşik değerleri - yüksek kalite odaklı
SIMILARITY_THRESHOLD = 0.25 # Vektör benzerlik eşiği
MIN_GENERAL_SCORE = 0.65    # Minimum genel skor (%65 uyum altı gösterilmez)
TOP_K_RESULTS = 12          # Vektör aramasından kaç aday getir
MAX_RETURN_RESULTS = 8      # Kullanıcıya max kaç sonuç göster

# --- Modeller ---
class MatchResult(BaseModel):
    job_title: str
    general_score: float
    skill_match: float
    experience_match: float
    report_summary: str

class CVValidationResult(BaseModel):
    is_cv: bool = Field(description="Is this document a CV/Resume? True or False")
    reason: str = Field(description="Why is it or why is it not a CV?")

app = FastAPI(title="HR TalentScout API - Optimized Matching")

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
    
    # 1. Supabase/DB Denemesi
    try:
        db = SessionLocal()
        db_jobs = db.query(models.JobPosting).all()
        if db_jobs:
            for job in db_jobs:
                content = f"Title: {job.title}\nDesc: {job.description}\nQuals: {job.requirements}"
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

    # 2. JSON Fallback
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

# --- PROMPTLAR (GELİŞTİRİLMİŞ) ---
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
matching_template = """
You are an expert Senior Technical Recruiter and HR AI.
Your task is to evaluate the relevance of a candidate's CV for a specific Job Posting.

JOB POSTING:
Title: {job_title}
Description: {description}

CANDIDATE CV:
{cv_content}

### EVALUATION GUIDELINES (READ CAREFULLY):

1. **PRIORITIZE SKILLS OVER BACKGROUND:**
   - If a candidate has the required technical skills (tools, languages, frameworks), rate them HIGH, even if their university degree is unrelated (e.g., a Philosophy grad who knows Python & React is a GOOD match for a Developer role).
   - **Internships, Bootcamps, and Projects COUNT as valid experience.** Do not treat them as zero experience.

2. **SCORING RUBRIC (0.0 to 1.0):**
   - **0.85 - 1.00 (Excellent):** Perfect match. Has all critical skills, relevant experience, and fits the seniority level.
   - **0.65 - 0.84 (Good/Strong):** Strong candidate. Has core skills but might lack some "nice-to-have" features or comes from a different background but proved their skills.
   - **0.40 - 0.64 (Potential/Junior):** Junior candidate, career switcher, or missing some specific tools but has a solid foundation. Worth an interview for junior/mid roles.
   - **0.00 - 0.39 (Mismatch):** Completely irrelevant (e.g., a Driver applying for a Surgeon role) or lacks fundamental required skills.

3. **ANALYSIS RULES:**
   - **Career Switchers:** If a candidate is switching fields (e.g., Sales -> QA Tester), look for *transferable skills* and *certifications*. If they exist, do NOT give a low score.
   - **Overqualified:** Do not penalize overqualified candidates unless the job strictly forbids it.

### OUTPUT FORMAT:
Return a valid JSON object with the following fields:
- **job_title**: The title from the job posting.
- **general_score**: A float between 0.0 and 1.0 representing the overall fit.
- **skill_match**: A float between 0.0 and 1.0 representing technical/hard skill alignment.
- **experience_match**: A float between 0.0 and 1.0 representing experience relevance.
- **report_summary**: A concise professional summary in **TURKISH** (2-3 sentences). Explain WHY they are a match or mismatch. Highlight key skills found or missing.

JSON OUTPUT:
"""
# matching_template = """
# You are an expert HR AI matching specialist. Evaluate how well this CV matches the job posting with BALANCED scoring that considers career alignment while staying fair.

# JOB POSTING:
# Title: {job_title}
# Description: {description}

# CANDIDATE CV:
# {cv_content}

# EVALUATION CRITERIA:

# 0. **CAREER FIELD ALIGNMENT CHECK - IMPORTANT BUT FLEXIBLE:**
#    - Assess if candidate's education/career background relates to the job's field
#    - Consider career transitions and transferable skills
#    - Examples of STRONG MISMATCHES (still scoreable but with awareness):
#      * Computer Engineer CV → Pure HR/Recruitment (no tech) 🔶
#      * Pure Marketing → Pure Backend Development 🔶
#      * Pure Accounting → Creative Design 🔶
#    - Examples of ACCEPTABLE or GOOD MATCHES:
#      * Software Engineer → Any Tech Role ✅
#      * Business/Any background → Tech-related Business roles ✅
#      * Engineering → Technical Product/Project Management ✅
#      * Any background → Entry-level roles (learning opportunity) ✅
#    - **Career field mismatch reduces score but doesn't eliminate candidates**

# 1. **SKILL MATCH (0.0-1.0)** - MOST IMPORTANT FACTOR:
#    - Technical skill alignment (tools, technologies, methods)
#    - Related/transferable skills valued
#    - **BONUS**: If candidate has 65%+ required skills, ADD +0.14
#    - Adjacent/similar skills get good credit
#    - 0.90+ = All critical skills + expertise
#    - 0.75-0.89 = Most required skills present
#    - 0.60-0.74 = Solid foundation with learnable gaps
#    - 0.45-0.59 = Relevant foundation exists
#    - <0.45 = Significant skill gaps

# 2. **EXPERIENCE MATCH (0.0-1.0)** - IMPORTANT FACTOR:
#    - Years of relevant experience - quality matters
#    - Role/seniority alignment - allow growth
#    - Industry relevance - adjacent fields count
#    - **BONUS**: If experience appropriate (±2 years), ADD +0.10
#    - **IMPORTANT**: Internships and practical training count as valuable experience
#    - 0.85+ = Perfect: same role, relevant industry, right level
#    - 0.70-0.84 = Strong: similar role/industry
#    - 0.50-0.69 = Good: relevant experience, growth potential
#    - 0.30-0.49 = Moderate: internships, bootcamp projects, career transition
#    - 0.20-0.35 = Entry-level: fresh grad with internships/strong projects
#    - <0.20 = Minimal experience (but skills can compensate)

# 3. **EDUCATION/CAREER BACKGROUND (0.0-1.0)** - CONTEXTUAL FACTOR:
#    - How well does educational background align?
#    - Career transitions are valued, not penalized heavily
#    - 0.85+ = Perfect alignment (CS → Software Dev)
#    - 0.70-0.84 = Strong alignment (Engineering → Tech roles)
#    - 0.55-0.69 = Moderate alignment (Business → Tech Business)
#    - 0.40-0.54 = Weak alignment but transferable (Any → Entry roles)
#    - 0.25-0.39 = Different field (considere if skills compensate)
#    - <0.25 = Very different (but still scoreable)

# 4. **GENERAL SCORE (0.0-1.0)** - WEIGHTED CALCULATION:
#    - **BALANCED FORMULA**: 
#      Base = (skill_match × 0.50) + (experience_match × 0.32) + (career_background × 0.12) + (certs/projects × 0.06)
   
#    - **RATIONALE**: 
#      * Skills (50%) = Primary factor - can they do the work?
#      * Experience (32%) = Very important - includes internships and projects
#      * Career Background (12%) = Context, not dealbreaker
#      * Other (6%) = Supporting achievements
   
#    - **GENEROUS BONUSES** (max +0.30 total):
#      * +0.15: Strong skill match (65%+ skills)
#      * +0.10: Experience level appropriate (including internships)
#      * +0.06: Good education/career alignment
#      * +0.06: Portfolio/GitHub with relevant work or strong internship projects
#      * +0.04: Recent learning/certifications in field
   
#    - **LIGHT PENALTIES** (only for clear issues):
#      * -0.08: Missing most critical must-have skills
#      * -0.06: Experience significantly misaligned (3+ years off)
#      * -0.04: Career background quite different AND no compensating factors
   
#    - **IMPORTANT**: Career background difference alone shouldn't eliminate candidates
#    - If skills are strong (0.70+), career background matters less
#    - Cap maximum at 0.95
#    - **MINIMUM THRESHOLD**: Score <0.65 = not qualified

# 5. **REPORT SUMMARY**:
#    - Write in Turkish, 2-3 sentences
#    - Lead with STRENGTHS - skills, experience, achievements
#    - For career transitions: Acknowledge background but emphasize relevant skills
#    - Examples:
#      * Good match: "Python ve Django konusunda güçlü, 2 yıl backend deneyimi var..."
#      * Career shift: "İşletme geçmişli ancak Python, SQL ve veri analizi becerilerine sahip..."
#      * Tech to business: "Yazılım geliştirme deneyimi teknik ürün yönetimi için değerli..."
#    - Be encouraging while honest

# REALISTIC CAREER BACKGROUND SCORING:

# **PERFECT ALIGNMENT (0.85-1.0):**
# - CS/Software Eng → Software Developer
# - Business/Marketing → Marketing Manager
# - Design → UX/UI Designer

# **STRONG ALIGNMENT (0.70-0.84):**
# - Any Engineering → Tech roles
# - Any Business degree → Business/Sales roles
# - Related technical fields → Similar tech roles

# **MODERATE ALIGNMENT (0.55-0.69):**
# - Business → Tech Product/Project roles
# - Engineering → Product Management
# - Any background → Entry-level in new field

# **WEAK BUT ACCEPTABLE (0.40-0.54):**
# - Different background + relevant bootcamp/courses
# - Career transition with proven projects
# - Different field but strong transferable skills

# **DIFFERENT FIELD (0.25-0.39):**
# - Significantly different but some skills transfer
# - Can still score 0.65+ if skills and experience compensate

# REALISTIC SCORING EXAMPLES:

# ✅ **CS Grad → Junior Dev (Perfect Match):**
# ```
# Skills: 0.75, Experience: 0.35, Career: 0.90
# Base = (0.75×0.50) + (0.35×0.32) + (0.90×0.12) + (0.60×0.06) = 0.595
# Bonuses: +0.15 +0.06 = +0.21
# Final: 0.805 ✅
# ```

# ✅ **CS Grad with Internship → Junior Dev:**
# ```
# Skills: 0.72, Experience: 0.40 (internship valued!), Career: 0.90
# Base = (0.72×0.50) + (0.40×0.32) + (0.90×0.12) + (0.65×0.06) = 0.615
# Bonuses: +0.15 +0.10 +0.06 = +0.31 (capped at +0.30)
# Final: 0.915 ✅ (Excellent!)
# ```

# ✅ **Business Grad + Bootcamp + Internship → Junior Dev:**
# ```
# Skills: 0.70 (bootcamp + projects), Experience: 0.35 (bootcamp internship), Career: 0.48
# Base = (0.70×0.50) + (0.35×0.32) + (0.48×0.12) + (0.70×0.06) = 0.519
# Bonuses: +0.15 +0.06 +0.04 = +0.25
# Final: 0.769 ✅ (Strong match despite career change!)
# ```

# ✅ **Business Grad + Bootcamp (no internship) → Junior Dev:**
# ```
# Skills: 0.68, Experience: 0.28 (projects only), Career: 0.45
# Base = (0.68×0.50) + (0.28×0.32) + (0.45×0.12) + (0.70×0.06) = 0.481
# Bonuses: +0.15 +0.06 +0.04 = +0.25
# Final: 0.731 ✅ (Still passes!)
# Penalty: -0.04 (if any) = 0.691 ✅ (Passes comfortably)
# ```

# ✅ **Engineer → Tech Product Manager:**
# ```
# Skills: 0.72, Experience: 0.58, Career: 0.68
# Base = (0.72×0.50) + (0.58×0.32) + (0.68×0.12) + (0.55×0.06) = 0.630
# Bonuses: +0.15 +0.06 = +0.21
# Final: 0.840 ✅
# ```

# ✅ **Mid-level Dev → Mid-level Dev (Perfect):**
# ```
# Skills: 0.85, Experience: 0.78, Career: 0.90
# Base = (0.85×0.50) + (0.78×0.32) + (0.90×0.12) + (0.70×0.06) = 0.765
# Bonuses: +0.15 +0.10 +0.06 +0.06 = +0.37 (capped at +0.30)
# Final: 0.95 ✅ (At cap - perfect match!)
# ```

# 🔶 **Fresh Grad (good skills, no internship) → Entry Role:**
# ```
# Skills: 0.68, Experience: 0.22 (projects only), Career: 0.85
# Base = (0.68×0.50) + (0.22×0.32) + (0.85×0.12) + (0.60×0.06) = 0.508
# Bonuses: +0.15 +0.04 = +0.19
# Final: 0.698 ✅ (Passes!)
# ```

# 🔶 **Computer Engineer → HR Role (Strong Mismatch):**
# ```
# Skills: 0.48 (some soft skills), Experience: 0.20, Career: 0.32
# Base = (0.48×0.50) + (0.20×0.32) + (0.32×0.12) + (0.40×0.06) = 0.346
# Penalty: -0.04
# Final: 0.306 ❌ (Fails naturally - correct outcome)
# ```

# 🔶 **Marketing → Software Dev (Weak Match Unless Compensated):**
# ```
# WITHOUT compensation:
# Skills: 0.42, Experience: 0.18, Career: 0.30
# Base = 0.324, Penalty: -0.04 = 0.284 ❌

# WITH bootcamp + strong projects + internship:
# Skills: 0.68, Experience: 0.38 (internship!), Career: 0.42
# Base = (0.68×0.50) + (0.38×0.32) + (0.42×0.12) + (0.75×0.06) = 0.536
# Bonuses: +0.15 +0.06 +0.04 = +0.25
# Final: 0.786 ✅ (Great! Skills + effort compensated)
# ```

# KEY PRINCIPLES:
# - **Skills can overcome career background differences**
# - **Internships are highly valued** - treat as real experience
# - Career transitions with proof (bootcamp, projects, internships) are rewarded
# - Background provides context but doesn't eliminate candidates
# - Strong skills (0.68+) + any experience (0.28+) can reach 0.65+
# - Don't penalize career changes - people grow and learn
# - Entry-level roles should be accessible to career changers with effort
# - Technical backgrounds valuable for technical business roles
# - Be encouraging about transferable skills and learning journey
# - **Fresh graduates with internships should score well (0.75-0.85)**
# - Only apply penalties when there's NO compensating factor at all

# OUTPUT VALID JSON:
# {{
#     "job_title": "{job_title}",
#     "general_score": 0.0,
#     "skill_match": 0.0,
#     "experience_match": 0.0,
#     "report_summary": "..."
# }}
# """

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
            for page in pages: text += page.page_content + "\n"
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
        if os.path.exists(tmp_path): os.remove(tmp_path)
    
    return re.sub(r'\s+', ' ', text).strip()

async def process_single_job(doc, cv_text, llm, parser, prompt_template):
    """
    Tek bir iş ilanını asenkron olarak analiz eder.
    """
    job_meta = doc.metadata
    try:
        chain = prompt_template | llm | parser
        
        res = await chain.ainvoke({
            "job_title": job_meta.get('job_title', 'Bilinmiyor'),
            "description": job_meta.get('description', ''),
            "cv_content": cv_text[:4000],  # Daha fazla context
            "format_instructions": parser.get_format_instructions()
        })
        return MatchResult(**res)
    except Exception as e:
        print(f"⚠️ İlan analiz hatası: {e}")
        return None

@app.post("/api/match_cv", response_model=List[MatchResult])
async def match_cv(file: UploadFile = File(...)):
    
    # 1. Metni Oku
    cv_text = extract_text_from_upload(file)
    
    if len(cv_text) < 20:
        return [MatchResult(
            job_title="Okuma Hatası",
            general_score=0.0, skill_match=0.0, experience_match=0.0,
            report_summary="UYARI: Dosya boş veya okunamadı."
        )]

    # ---------------------------------------------------------
    # ADIM 1: BU BİR CV Mİ? (BEKÇİ)
    # ---------------------------------------------------------
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    
    validator_parser = JsonOutputParser(pydantic_object=CVValidationResult)
    validator_prompt = ChatPromptTemplate.from_template(validation_template)
    
    try:
        validator_chain = validator_prompt | llm | validator_parser
        validation_res = await validator_chain.ainvoke({
            "text_sample": cv_text[:1500], 
            "format_instructions": validator_parser.get_format_instructions()
        })
        
        if not validation_res['is_cv']:
            return [MatchResult(
                job_title="Geçersiz Belge",
                general_score=0.0,
                skill_match=0.0,
                experience_match=0.0,
                report_summary=f"UYARI: {validation_res['reason']}"
            )]

    except Exception as e:
        print(f"Validasyon Hatası: {e}")

    # ---------------------------------------------------------
    # ADIM 2: AKILLI EŞLEŞTİRME (PARALEL + FİLTRELİ)
    # ---------------------------------------------------------
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Sistem hazır değil.")
    
    # Vektör benzerliğine göre en yakın ilanları getir
    relevant_docs_with_scores = vector_store.similarity_search_with_score(
        cv_text, 
        k=TOP_K_RESULTS
    )
    
    # İlk filtreleme: Vektör benzerliği eşiğini geçenleri al
    filtered_docs = [
        doc for doc, score in relevant_docs_with_scores 
        if score >= SIMILARITY_THRESHOLD
    ]
    
    if not filtered_docs:
        return [MatchResult(
            job_title="Eşleşme Bulunamadı",
            general_score=0.0,
            skill_match=0.0,
            experience_match=0.0,
            report_summary="Sistemimizdeki ilanlarla yeterli benzerlik bulunamadı. Lütfen farklı bir CV deneyin veya profil bilgilerinizi güncelleyin."
        )]
    
    matcher_parser = JsonOutputParser(pydantic_object=MatchResult)
    matcher_prompt = ChatPromptTemplate.from_template(matching_template)
    
    # Paralel işleme görevleri oluştur
    tasks = [
        process_single_job(doc, cv_text, llm, matcher_parser, matcher_prompt)
        for doc in filtered_docs
    ]
    
    # Tüm analizleri eşzamanlı başlat
    results_raw = await asyncio.gather(*tasks)
    
    # Hatalı sonuçları temizle
    valid_results = [res for res in results_raw if res is not None]
    
    # İkinci filtreleme: Minimum skor eşiğini geçenleri al
    qualified_results = [
        res for res in valid_results 
        if res.general_score >= MIN_GENERAL_SCORE
    ]
    
    # Skorlara göre sırala (en yüksek önce)
    qualified_results.sort(key=lambda x: x.general_score, reverse=True)
    
    # En iyi sonuçları döndür
    final_results = qualified_results[:MAX_RETURN_RESULTS]
    
    # Hiç kaliteli eşleşme yoksa bilgilendirici mesaj
    if not final_results:
        return [MatchResult(
            job_title="Uygun Pozisyon Bulunamadı",
            general_score=0.0,
            skill_match=0.0,
            experience_match=0.0,
            report_summary=f"CV'niz analiz edildi ancak sistemdeki pozisyonlarla %65'in üzerinde eşleşme sağlanamadı. İncelenen {len(valid_results)} pozisyon uyum eşiğinin altında kaldı. Beceri ve deneyimlerinizi güçlendirerek daha uygun pozisyonlara başvurabilirsiniz."
        )]
    
    return final_results

@app.get("/api/health")
async def health_check():
    """Sistem durumu kontrolü"""
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None,
        "config": {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "min_score": MIN_GENERAL_SCORE,
            "min_score_percentage": f"%{int(MIN_GENERAL_SCORE*100)}",
            "max_results": MAX_RETURN_RESULTS
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)