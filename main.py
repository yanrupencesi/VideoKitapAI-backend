from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from openai import OpenAI

# ------------------ FastAPI uygulaması ------------------ #

app = FastAPI(title="VideoKitapAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # geliştirme için açık
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ OpenAI client ------------------ #

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ------------------ Veri Modelleri ------------------ #

class Book(BaseModel):
    id: int
    title: str
    description: str

class AIAskRequest(BaseModel):
    book_id: int
    question: str

class AIAskResponse(BaseModel):
    answer: str

class ProfileRequest(BaseModel):
    goal: str
    challenge: str
    time_per_day: Optional[str] = None

class ProfileRecommendationResponse(BaseModel):
    book_id: int
    book_title: str
    questions: List[str]

# ------------------ Basit Kitap Veritabanı ------------------ #

BOOKS: List[Book] = [
    Book(id=1, title="Atomik Alışkanlıklar",
         description="Küçük alışkanlık değişimlerinin büyük sonuçlara dönüşmesini anlatır."),
    Book(id=2, title="Savaş Sanatı",
         description="Strateji ve taktik üzerine zamansız prensipler."),
    Book(id=3, title="Akış (Flow)",
         description="Akış hali ve derin odaklanmayı anlatır."),
    Book(id=4, title="Zengin Baba Yoksul Baba",
         description="Finansal özgürlük ve para yönetimi üzerine önemli bakış açıları."),
    Book(id=5, title="Düşün ve Zengin Ol",
         description="Başarı, hedef, motivasyon ve zenginliği zihinsel prensiplerle açıklayan klasik eser."),
    Book(id=6, title="Alışkanlıkların Gücü",
         description="Alışkanlık döngüsünü ve davranış değişimini açıklar."),
    Book(id=7, title="İknanın Psikolojisi",
         description="Etkileme, ikna ve davranış bilimlerinin temel prensipleri."),
    Book(id=8, title="Pür Dikkat (Deep Work)",
         description="Derin çalışma ve odaklanmayı geri kazanma üzerine pratik rehber."),
]

def get_book(book_id: int) -> Optional[Book]:
    for b in BOOKS:
        if b.id == book_id:
            return b
    return None

# ------------------ Profil Analizi: Kitap Önerisi ------------------ #

def recommend_book_and_questions(goal: str, challenge: str):
    goal = (goal or "").lower()
    challenge = (challenge or "").lower()
    text = f"{goal} {challenge}"

    # --- ZENGİN BABA YOKSUL BABA ---
    if any(k in text for k in ["para", "finans", "borç", "gelir", "yatırım"]):
        return {
            "book_id": 4,
            "book_title": "Zengin Baba Yoksul Baba",
            "questions": [
                "Bu kitap, para ve zenginlik hakkında bakış açımı nasıl değiştirebilir?",
                "Finansal özgürlük için bugün başlayabileceğim en basit adımlar neler?",
                "Gelirimi artırmak için hangi prensipleri uygulayabilirim?"
            ],
        }

    # --- DÜŞÜN VE ZENGİN OL ---
    if any(k in text for k in [
        "zengin olmak", "finansal özgürlük", "para kazanmak",
        "finansal hedef", "başarı", "kariyer", "motivasyon"
    ]):
        return {
            "book_id": 5,
            "book_title": "Düşün ve Zengin Ol",
            "questions": [
                "Finansal hedefime ulaşabilmem için bu kitaptan hangi adımlarla başlamalıyım?",
                "Düşünce gücünü kullanarak motivasyonumu nasıl artırabilirim?",
                "Başarı için bugün uygulayabileceğim 3 adımı söyleyebilir misin?"
            ],
        }

    # --- ALIŞKANLIKLARIN GÜCÜ ---
    if "alışkanlık" in text and any(k in text for k in ["iş", "kurumsal", "ofis"]):
        return {
            "book_id": 6,
            "book_title": "Alışkanlıkların Gücü",
            "questions": [
                "Alışkanlık döngüsünü hayatıma nasıl uygulayabilirim?",
                "Kötü alışkanlıklarımı nasıl değiştirebilirim?",
                "Bugün uygulayabileceğim bir alışkanlık deneyi önerir misin?"
            ],
        }

    # --- ATOMİK ALIŞKANLIKLAR ---
    if any(k in text for k in ["alışkanlık", "disiplin", "ertelemek"]):
        return {
            "book_id": 1,
            "book_title": "Atomik Alışkanlıklar",
            "questions": [
                "Hangi küçük alışkanlıklarla başlamalıyım?",
                "Erteleme ile nasıl başa çıkabilirim?",
                "Bugün uygulayabileceğim 3 mikro alışkanlık önerir misin?"
            ],
        }

    # --- PÜR DİKKAT ---
    if any(k in text for k in ["odak", "verimlilik", "deep work", "dikkat", "sosyal medya"]):
        return {
            "book_id": 8,
            "book_title": "Pür Dikkat (Deep Work)",
            "questions": [
                "Dikkatimi toparlamak için nereden başlamalıyım?",
                "Odaklanmayı öldüren alışkanlıkları nasıl azaltabilirim?",
                "Bugün uygulayabileceğim bir derin çalışma planı verebilir misin?"
            ],
        }

    # --- AKIŞ (FLOW) ---
    if any(k in text for k in ["akış", "flow", "tutku", "yaratıcılık"]):
        return {
            "book_id": 3,
            "book_title": "Akış (Flow)",
            "questions": [
                "Akış haline nasıl girebilirim?",
                "Odaklanma sorunu yaşadığımda ne yapmalıyım?",
                "Akış için bugün yapabileceğim 2 küçük egzersiz ne?"
            ],
        }

    # --- İKNANIN PSİKOLOJİSİ ---
    if any(k in text for k in ["ikna", "satış", "pazarlama", "etkilemek"]):
        return {
            "book_id": 7,
            "book_title": "İknanın Psikolojisi",
            "questions": [
                "İkna prensiplerini günlük hayatta nasıl kullanabilirim?",
                "Müşteri iletişiminde bana nasıl avantaj sağlar?",
                "Etik bir şekilde daha etkileyici olmak için neler yapmalıyım?"
            ],
        }

    # --- SAVAŞ SANATI ---
    if any(k in text for k in ["strateji", "taktik", "rekabet", "liderlik"]):
        return {
            "book_id": 2,
            "book_title": "Savaş Sanatı",
            "questions": [
                "Stratejik düşünmeye nereden başlamalıyım?",
                "Mevcut zorluklara hangi stratejiler daha uygun olur?",
                "Bugün uygulayabileceğim 2-3 taktik verebilir misin?"
            ],
        }

    # --- Hiçbiri uymuyorsa: Atomik Alışkanlıklar ---
    return {
        "book_id": 1,
        "book_title": "Atomik Alışkanlıklar",
        "questions": [
            "Hedefime ulaşmak için hangi temel adımlarla başlamalıyım?",
            "Bu kitap bana nasıl yardımcı olabilir?",
            "Bugün uygulayabileceğim küçük adımlar nelerdir?"
        ],
    }


# ------------------ Endpoint'ler ------------------ #

@app.get("/")
def root():
    return {"status": "ok", "message": "VideoKitapAI backend çalışıyor."}

@app.get("/books", response_model=List[Book])
def list_books():
    return BOOKS

@app.post("/profile/recommend", response_model=ProfileRecommendationResponse)
def profile_recommend(req: ProfileRequest):
    result = recommend_book_and_questions(req.goal, req.challenge)
    return ProfileRecommendationResponse(**result)

@app.post("/ai/ask", response_model=AIAskResponse)
def ai_ask(req: AIAskRequest):
    book = get_book(req.book_id)

    if not OPENAI_API_KEY or client is None:
        # Fallback
        fallback = (
            "Sunucuda OPENAI_API_KEY tanımlı olmadığı için gerçek yapay zeka cevabı "
            "üretemiyorum.\n\n"
        )
        if book:
            fallback += f"Kitap: {book.title}\n\nÖzet: {book.description}\n\n"
        return AIAskResponse(answer=fallback)

    if not book:
        return AIAskResponse(answer="Bu kitap ID'si bulunamadı.")

    system_msg = (
        "Sen VideoKitapAI adlı bir uygulamada çalışan bir asistansın. "
        "Görevin: kitabı oyunlaştırılmış şekilde açıklamak.\n"
        "• 2–3 cümle özet\n"
        "• Ana prensipler maddeleri\n"
        "• Mini görevler (ölçülebilir, uygulanabilir)\n"
        "• Sade ve motive edici ton\n"
    )

    user_msg = f"""
Seçilen kitap: {book.title}
Kitap açıklaması: {book.description}

Kullanıcının sorusu:
{req.question}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        answer_text = completion.choices[0].message.content.strip()
    except Exception as e:
        answer_text = f"Hata: {e}"

    # DEBUG: Kitap adını cevabın başına ekle
    debug_prefix = f"[Seçilen kitap: {book.title}] "

    return AIAskResponse(answer=debug_prefix + answer_text)
