import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Veritabanı Bağlantı Bilgileri ---
# Şifrenizi ve veritabanı adınızı buraya göre güncelleyin.
# Varsayılan: kullanıcı='postgres', şifre='kendi_sifreniz', host='localhost', port='5432', db='llm_jobs_db'
# NOT: Güvenlik için şifreyi environment variable'dan almak daha iyidir ama şimdilik doğrudan yazabiliriz.
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:123@localhost/llm_jobs_db"

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