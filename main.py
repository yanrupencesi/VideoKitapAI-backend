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
    Book(
        id=1,
        title="Atomik Alışkanlıklar",
        description=(
            "Küçük, tekrarlanan alışkanlıkların zamanla kimliğini ve hayatını "
            "nasıl değiştirdiğini anlatan bir kişisel gelişim kitabı."
        ),
    ),
    Book(
        id=2,
        title="Savaş Sanatı",
        description=(
            "Strateji, taktik ve rakibi anlamaya dayalı; hem savaş hem iş hem de günlük hayata "
            "uygulanabilen zamansız prensipler içerir."
        ),
    ),
    Book(
        id=3,
        title="Akış (Flow)",
        description=(
            "İnsanın yaptığı işe tamamen daldığı, zaman algısını kaybettiği 'akış hali'ni "
            "ve bunu hayatına daha çok nasıl taşıyabileceğini anlatır."
        ),
    ),
    Book(
        id=4,
        title="Zengin Baba Yoksul Baba",
        description=(
            "Paraya bakış açını değiştiren, finansal özgürlük için zenginlerin nasıl düşündüğünü "
            "ve hareket ettiğini anlatan bir finansal zihinset kitabı."
        ),
    ),
]

def get_book(book_id: int) -> Optional[Book]:
    for b in BOOKS:
        if b.id == book_id:
            return b
    return None

# ------------------ Profilden kitap + soru önerisi ------------------ #

def recommend_book_and_questions(goal: str, challenge: str):
    """
    Kullanıcının hedef ve zorluk bilgilerine göre
    önerilen kitabı ve kitabın kişiye özel sorularını döner.
    """
    goal = (goal or "").lower()
    challenge = (challenge or "").lower()

    # --- ZENGİN BABA YOKSUL BABA ---
    if ("zengin" in goal or "para" in goal or "finans" in goal or
        "zengin" in challenge or "para" in challenge or "borç" in challenge):
        return {
            "book_id": 4,
            "book_title": "Zengin Baba Yoksul Baba",
            "questions": [
                "Finansal hedefime ulaşmak için nereden başlamalıyım?",
                "Şu an yaşadığım finansal zorlukları bu kitaba göre nasıl aşabilirim?",
                "Bugün uygulayabileceğim 2-3 küçük finansal alışkanlık önerir misin?"
            ]
        }

    # --- ATOMİK ALIŞKANLIKLAR ---
    if ("disiplin" in goal or "alışkanlık" in goal or "düzen" in goal or
        "erte" in challenge or "motivasyon" in challenge):
        return {
            "book_id": 1,
            "book_title": "Atomik Alışkanlıklar",
            "questions": [
                "Hedefime ulaşmak için hangi küçük alışkanlıklarla başlayabilirim?",
                "Erteleme sorunumu bu kitabı kullanarak nasıl çözebilirim?",
                "Bugün uygulayabileceğim 3 küçük alışkanlık önerir misin?"
            ]
        }

    # --- AKIŞ (FLOW) ---
    if ("odak" in goal or "akış" in goal or "flow" in goal or
        "dikkat" in challenge or "konsant" in challenge):
        return {
            "book_id": 3,
            "book_title": "Akış (Flow)",
            "questions": [
                "Daha fazla akış yaşayabilmek için nereden başlamalıyım?",
                "Odaklanma sorunuma göre ne önerirsin?",
                "Bugün uygulayabileceğim 2-3 odak egzersizi verebilir misin?"
            ]
        }

    # --- SAVAŞ SANATI ---
    if ("strateji" in goal or "rekabet" in goal or "liderlik" in goal or
        "rekabet" in challenge or "analiz" in challenge or "yarış" in challenge):
        return {
            "book_id": 2,
            "book_title": "Savaş Sanatı",
            "questions": [
                "Stratejik düşünme becerimi geliştirmek için nereden başlamalıyım?",
                "Şu an yaşadığım zorluklara hangi stratejiler daha uygun olur?",
                "Bugün uygulayabileceğim 2-3 basit strateji örneği verebilir misin?"
            ]
        }

    # --- HİÇBİRİ UYMAZSA GENEL KİŞİSEL GELİŞİM ---
    return {
        "book_id": 1,
        "book_title": "Atomik Alışkanlıklar",
        "questions": [
            "Hedefime ulaşmak için hangi temel adımlarla başlamalıyım?",
            "Bu kitap bana nasıl yardımcı olabilir?",
            "Bugün uygulayabileceğim 2-3 küçük adım önerir misin?"
        ]
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
    """
    Kullanıcının profil bilgilerine göre:
    - Önerilen kitabı
    - Kitaba göre 3 soru önerisini döner.
    """
    result = recommend_book_and_questions(req.goal, req.challenge)
    return ProfileRecommendationResponse(**result)

@app.post("/ai/ask", response_model=AIAskResponse)
def ai_ask(req: AIAskRequest):
    """
    Seçilen kitap + kullanıcının sorusu üzerinden
    oyunlaştırılmış, özet + görev odaklı cevap döner.
    """
    book = get_book(req.book_id)

    if not OPENAI_API_KEY or client is None:
        # API anahtarı yoksa, anlamlı ama sabit bir cevap verelim
        fallback = (
            "Sunucuda OPENAI_API_KEY tanımlı olmadığı için gerçek yapay zeka cevabı "
            "üretemiyorum, ama genel bir çerçeve çizebilirim.\n\n"
        )
        if book:
            fallback += f"Kitap: {book.title}\n\nÖzet: {book.description}\n\n"
        fallback += (
            "Kendine şu mini görevleri verebilirsin:\n"
            "1) Kitaptan bir prensip seç ve bugün hayatında test et.\n"
            "2) Akşam, gerçekten fark yaratıp yaratmadığını 3 cümleyle yaz."
        )
        return AIAskResponse(answer=fallback)

    if not book:
        return AIAskResponse(
            answer="Bu kitap ID'si bulunamadı. Lütfen geçerli bir kitap seç."
        )

    system_msg = (
        "Sen VideoKitapAI adlı bir uygulamada çalışan akıllı bir asistansın. "
        "Görevin, kullanıcıya seçtiği kitabı oyunlaştırılmış bir şekilde anlatmak.\n\n"
        "Cevaplarında şu yapıyı kullan:\n"
        "1) Kısa ve net bir genel özet (2–3 cümle).\n"
        "2) Kitabın ana prensiplerini madde madde yaz.\n"
        "3) 'Mini Görevler' başlığı altında, kullanıcının bugün deneyebileceği "
        "2–3 küçük görev (quest) ver. Görevler mümkünse somut ve ölçülebilir olsun.\n"
        "4) Tonun motive edici, sade ve Türkçe olsun."
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
        answer_text = f"AI isteğinde bir hata oluştu: {e}"

    return AIAskResponse(answer=answer_text)
