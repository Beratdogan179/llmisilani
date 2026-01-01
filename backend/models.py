from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from database import Base

class JobPosting(Base):
    """İş İlanları Tablosu"""
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)          # İlan Başlığı
    company = Column(String, index=True)        # Şirket Adı
    description = Column(Text)                  # Detaylı İş Tanımı (LLM bunu okuyacak)
    requirements = Column(Text)                 # Gereksinimler
    location = Column(String)                   # Lokasyon
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Embedding vektörünü de veritabanında saklayabiliriz (pgvector eklentisi ile)
    # Ancak şimdilik FAISS'i ayrı tutmak daha basit olacaktır.