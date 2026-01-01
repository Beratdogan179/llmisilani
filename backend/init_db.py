from database import engine, Base
import models

def init_db():
    print("Veritabanı tabloları oluşturuluyor...")
    Base.metadata.create_all(bind=engine)
    print("Tablolar başarıyla oluşturuldu!")

if __name__ == "__main__":
    init_db()