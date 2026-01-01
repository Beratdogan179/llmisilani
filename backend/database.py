import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# --- Veritabanı Bağlantı Bilgileri ---
# Supabase veya yerel PostgreSQL bağlantısı için .env dosyasından SUPABASE_DB_URL okur.
# Örn: postgresql://postgres:[password]@[host]:[port]/postgres
SQLALCHEMY_DATABASE_URL = os.getenv("SUPABASE_DB_URL", "postgresql://postgres:123@localhost/llm_jobs_db")

# Veritabanı motorunu oluştur
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Oturum (Session) oluşturucu
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base sınıfı (Modeller için)
Base = declarative_base()

# Veritabanı oturumu alma fonksiyonu (Dependency Injection için)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()