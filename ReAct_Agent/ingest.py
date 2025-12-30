import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- AYARLAR ---
PDF_PATH = "data/is_kanunu.pdf"
DB_PATH = "faiss_index"  # Klasor adi degisti
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def veritabani_olustur():
    print("🔄 PDF yukleniyor...")
    if not os.path.exists(PDF_PATH):
        print(f"❌ HATA: '{PDF_PATH}' dosyasi bulunamadi!")
        return

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    # Metni parcala
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Metin {len(chunks)} parcaya bolundu.")

    # Embedding modelini yukle
    print("🧠 Embedding modeli hazirlaniyor...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # FAISS veritabanini olustur ve kaydet
    print("💾 FAISS veritabani olusturuluyor...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    
    print(f"✅ BAŞARILI! Veritabani '{DB_PATH}' klasorune kaydedildi.")

if __name__ == "__main__":
    veritabani_olustur()