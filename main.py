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
    Book(
        id=5,
        title="Düşün ve Zengin Ol",
        description=(
            "Napoleon Hill'in klasik eseri. Başarı ve zenginliği sadece para değil, "
            "düşünce gücü, inanç, kararlılık ve eylem üzerinden anlatır. "
            "Hedef belirleme, motivasyon ve özgüven üzerine güçlü prensipler sunar."
        ),
    ),
    Book(
        id=6,
        title="Alışkanlıkların Gücü",
        description=(
            "Gündelik hayat ve iş hayatındaki alışkanlıkların, ipucu-rutin-ödül döngüsü üzerinden "
            "nasıl oluştuğunu ve bu döngüyü değiştirerek hem bireysel hem kurumsal dönüşümün "
            "nasıl sağlanabileceğini anlatır."
        ),
    ),
    Book(
        id=7,
        title="İknanın Psikolojisi",
        description=(
            "İkna ve etkilemenin psikolojik prensiplerini (karşılıklılık, kıtlık, sosyal kanıt, "
            "otorite, tutarlılık vb.) açıklayan; satış, pazarlama ve günlük iletişime uygulanabilen "
            "klasik bir sosyal psikoloji kitabı."
        ),
    ),
    Book(
        id=8,
        title="Pür Dikkat (Deep Work)",
        description=(
            "Dağılmış dikkati toparlayıp, kesintisiz odaklanma ile kaliteli işler üretmeyi "
            "anlatan; dikkat ekonomisi çağında derin çalışmanın önemini ve pratik yöntemlerini sunar."
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
    text = f"{goal} {challenge}"

    # --- ZENGİN BABA YOKSUL BABA (para / finans / borç) ---
    if any(k in text for k in ["para", "finans", "borç", "gelir", "yatırım", "maddi", "fatura", "borçlar"]):
        return {
            "book_id": 4,
            "book_title": "Zengin Baba Yoksul Baba",
            "questions": [
                "Bu kitap, para ve zenginlik hakkında bakış açımı nasıl değiştirebilir?",
                "Finansal özgürlük için bugün başlayabileceğim en basit adımlar neler?",
                "Gelirimi artırmak veya daha bilinçli harcama yapmak için bu kitaptan hangi prensipleri uygulayabilirim?"
            ],
        }

    # --- DÜŞÜN VE ZENGİN OL (başarı / kariyer / finansal hedef / motivasyon) ---
    if any(k in text for k in [
        "zengin olmak", "finansal özgürlük", "para kazanmak",
        "finansal hedef", "finansal hedefim", "başarı", "kariyer", "motivasyon"
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

    # --- ALIŞKANLIKLARIN GÜCÜ (alışkanlık + iş/kurum) ---
    if "alışkanlık" in text and any(k in text for k in ["iş", "ofis", "şirket", "kurumsal", "organizasyon", "ekip"]):
        return {
            "book_id": 6,
            "book_title": "Alışkanlıkların Gücü",
            "questions": [
                "Bu kitap, alışkanlık döngüsünü (ipucu–rutin–ödül) anlamamda nasıl yardımcı olabilir?",
                "İş ve özel hayatımdaki kötü alışkanlıkları değiştirmek için hangi adımları uygulamalıyım?",
                "Bugün hayatımda test edebileceğim küçük bir alışkanlık deneyi önerir misin?"
            ],
        }

    # --- ATOMİK ALIŞKANLIKLAR (kişisel alışkanlık / erteleme) ---
    if any(k in text for k in ["alışkanlık", "disiplin", "düzen", "ertelemek", "erteleme", "rutin"]):
        return {
            "book_id": 1,
            "book_title": "Atomik Alışkanlıklar",
            "questions": [
                "Hedefime ulaşmak için hangi küçük alışkanlıklarla başlayabilirim?",
                "Erteleme sorunumu bu kitabı kullanarak nasıl çözebilirim?",
                "Bugün uygulayabileceğim 3 küçük alışkanlık önerir misin?"
            ],
        }

    # --- PÜR DİKKAT (odak / verimlilik / derin çalışma) ---
    if any(k in text for k in ["odak", "odaklanmak", "verimlilik", "derin çalışma", "deep work", "üretkenlik", "dikkat dağınıklığı", "sosyal medya"]):
        return {
            "book_id": 8,
            "book_title": "Pür Dikkat (Deep Work)",
            "questions": [
                "Bu kitap, dağılmış dikkatimi toparlayıp daha derin çalışmam için bana nasıl yol gösterir?",
                "Sosyal medya ve bildirimler yüzünden odaklanamıyorum, Pür Dikkat'e göre nereden başlamalıyım?",
                "Bugün uygulayabileceğim 2–3 'derin çalışma' seansı planı verebilir misin?"
            ],
        }

    # --- AKIŞ (FLOW) (mutluluk / keyif / yaratıcılık) ---
    if any(k in text for k in ["akış hali", "akış", "flow", "yaratıcılık", "yaratıcı", "mutlu olmak", "tutku", "keyif almak"]):
        return {
            "book_id": 3,
            "book_title": "Akış (Flow)",
            "questions": [
                "Akış hali nedir ve benim hayatımda daha çok akış yaşayabilmem için nereden başlamalıyım?",
                "Odaklanma veya motivasyon düşüşü yaşadığımda akış haline dönmek için neler yapabilirim?",
                "Bugün akış haline yaklaşmak için yapabileceğim 2–3 küçük egzersiz ne?"
            ],
        }

    # --- İKNANIN PSİKOLOJİSİ (ikna / satış / pazarlama / iletişim) ---
    if any(k in text for k in ["ikna", "satış", "pazarlama", "insanları ikna", "insanları etkilemek", "sosyal kanıt", "otorite", "müşteri"]):
        return {
            "book_id": 7,
            "book_title": "İknanın Psikolojisi",
            "questions": [
                "İknanın Psikolojisi'ndeki temel ikna prensiplerini günlük hayatımda nasıl kullanabilirim?",
                "Satış veya pazarlama alanında çalışıyorum, bu kitap bana müşterilerle iletişimde nasıl avantaj sağlar?",
                "İnsanları manipüle etmeden, etik şekilde daha etkileyici olmak için neler yapabilirim?"
            ],
        }

    # --- SAVAŞ SANATI (strateji / rekabet / liderlik) ---
    if any(k in text for k in ["strateji", "taktik", "rekabet", "liderlik", "savaş", "rakip"]):
        return {
            "book_id": 2,
            "book_title": "Savaş Sanatı",
            "questions": [
                "Stratejik düşünme becerimi geliştirmek için nereden başlamalıyım?",
                "Şu an yaşadığım zorluklara hangi stratejiler daha uygun olur?",
                "Bugün uygulayabileceğim 2-3 basit strateji örneği verebilir misin?"
            ],
        }

    # --- HİÇBİRİ UYMAZSA GENEL (Atomik Alışkanlıklar) ---
    return {
        "book_id": 1,
        "book_title": "Atomik Alışkanlıklar",
        "questions": [
            "Hedefime ulaşmak için hangi temel adımlarla başlamalıyım?",
            "Bu kitap bana nasıl yardımcı olabilir?",
            "Bugün uygulayabileceğim 2-3 küçük adım önerir misin?"
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

    # DEBUG: Cevabın başına seçilen kitabı yaz
    debug_prefix = f"[Seçilen kitap: {book.title}] "
    return AIAskResponse(answer=debug_prefix + answer_text)
