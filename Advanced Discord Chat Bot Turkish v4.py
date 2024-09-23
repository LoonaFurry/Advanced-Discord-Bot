import asyncio
import io
import json
import logging
import os
import pickle
import random
import re
import traceback
import sys
import time
import hashlib
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import aiohttp
import aiosqlite
import backoff
import discord
import networkx as nx
import numpy as np
import requests
import spacy
from bs4 import BeautifulSoup
from discord.ext import tasks
from duckduckgo_search import AsyncDDGS
from google.api_core.exceptions import GoogleAPIError
from google.generativeai import configure, GenerativeModel
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import google.generativeai as genai
from transitions import Machine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from PIL import Image
import functools  # for error tracker
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
import torch
from huggingface_hub import login
from typing import Optional

# Set your Hugging Face token here
HUGGINGFACE_TOKEN = "your-huggingface-token"  # Replace with your actual token
login(HUGGINGFACE_TOKEN)

# --- FLUX.1-schnell ---
FLUX_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
try:
    image_generator = DiffusionPipeline.from_pretrained(
        FLUX_MODEL_PATH,
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
    ).to("cuda")
except Exception as e:
    logging.error(f"Failed to load FLUX.1-schnell: {e}")
    raise  # Don't continue if image generator fails

# --- Negative Prompts for Image Generation ---
negative_prompts = [
    "Ugly", "Bad anatomy", "Bad proportions", "Bad quality", "Blurry", "Cropped",
    "Deformed", "Disconnected limbs", "Out of frame", "Out of focus", "Dehydrated",
    "Error", "Disfigured", "Disgusting", "Extra arms", "Extra limbs", "Extra hands",
    "Fused fingers", "Gross proportions", "Long neck", "Low res", "Low quality",
    "Jpeg", "Jpeg artifacts", "Malformed limbs", "Mutated", "Mutated hands",
    "Mutated limbs", "Missing arms", "Missing fingers", "Picture frame",
    "Poorly drawn hands", "Poorly drawn face", "Text", "Signature", "Username",
    "Watermark", "Worst quality", "Collage", "Pixel", "Pixelated", "Grainy",
    "Amputee", "Missing fingers", "Missing hands", "Missing limbs", "Missing arms",
    "Extra fingers", "Extra hands", "Extra limbs", "Mutated hands", "Mutated",
    "Mutation", "Multiple heads", "Malformed limbs", "Disfigured", "Poorly drawn hands",
    "Poorly drawn face", "Long neck", "Fused fingers", "Fused hands", "Dismembered",
    "Duplicate", "Improper scale", "Ugly body", "Cloned face", "Cloned body",
    "Gross proportions", "Body horror", "Cartoon", "CGI", "Render", "3D",
    "Artwork", "Illustration", "3D render", "Cinema 4D", "Artstation",
    "Octane render", "Painting", "Oil painting", "Anime", "2D", "Sketch",
    "Drawing", "Bad photography", "Bad photo", "Deviant art", "Nsfw",
    "Uncensored", "Cleavage", "Nude", "Nipples", "Overexposed", "Simple background",
    "Plain background", "Grainy", "Portrait", "Grayscale", "Monochrome",
    "Underexposed", "Low contrast", "Low quality", "Dark", "Distorted",
    "White spots", "Deformed structures", "Macro", "Multiple angles", "Asymmetry",
    "Parts", "Components", "Design", "Broken", "Cartoon", "Distorted",
    "Extra pieces", "Bad proportion", "Inverted", "Misaligned", "Macabre",
    "Missing parts", "Oversized", "Tilted", "Too many fingers"
]

# --- Load Pre-trained Models for NLP ---
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    logging.warning(
        "Multilingual spaCy model not found. Downloading xx_ent_wiki_sm model..."
    )
    spacy.cli.download("xx_ent_wiki_sm")
    nlp = spacy.load("xx_ent_wiki_sm")

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except OSError:
    logging.warning("Sentiment analysis model not found. Downloading...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# --- Advanced AI Assistant Class ---
class AdvancedAIAssistant:
    def __init__(self):
        self.memory_network = nx.DiGraph()
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.ethical_guidelines = self._load_ethical_guidelines()

    def _load_ethical_guidelines(self) -> Dict[str, List[str]]:
        return {
            "privacy": ["respect user data", "minimize data collection"],
            "fairness": ["avoid bias", "ensure equal treatment"],
            "transparency": ["explain decisions", "provide clear information"],
        }


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hata.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# --- Bot Instance and Environment Variables ---
intents = discord.Intents.all()
intents.message_content = True
intents.members = True

discord_token = ("your-discord-token")
gemini_api_key = ("your-gemini-key")
if not discord_token or not gemini_api_key:
    raise ValueError(
        "DISCORD_TOKEN and GEMINI_API_KEY environment variables must be set."
    )

# --- Gemini AI Configuration ---
configure(api_key=gemini_api_key)
üretim_yapılandırması = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
model = GenerativeModel("gemini-pro", generation_config=üretim_yapılandırması)

# --- Discord Bot Configuration ---
bot = discord.Client(intents=intents)

# --- Directory and Database Settings ---
KOD_DİZİNİ = os.path.dirname(__file__)
VERİTABANI_DOSYASI = os.path.join(KOD_DİZİNİ, "sohbet_gecmisi.db")
KULLANICI_PROFİLLERİ_DOSYASI = os.path.join(KOD_DİZİNİ, "kullanici_profilleri.json")
BİLGİ_GRAFİĞİ_DOSYASI = os.path.join(KOD_DİZİNİ, "bilgi_grafi.pkl")
IMAGE_DATABASE = os.path.join(KOD_DİZİNİ, "image_data.db")

# --- Context Window and User Profiles ---
BAĞLAM_PENCERESİ_BOYUTU = 10000
kullanıcı_profilleri = defaultdict(
    lambda: {
        "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
        "demografi": {"yaş": None, "konum": None},
        "geçmiş_özeti": "",
        "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU),
        "kişilik": {
            "mizah": 0.5,
            "nezaket": 0.8,
            "iddialılık": 0.6,
            "yaratıcılık": 0.5,
        },
        "diyalog_durumu": "karşılama",
        "uzun_süreli_hafıza": [],
        "son_bot_eylemi": None,
        "ilgiler": [],
        "sorgu": "",
        "planlama_durumu": {},
        "etkileşim_geçmişi": [],
        "feedback_topics": [],
        "feedback_keywords": [],
        "satisfaction": 0,
        "duygusal_durum": "nötr",
        "çıkarımlar": [],
    }
)

# --- Dialogue and Action Types ---
DİYALOG_DURUMLARI = [
    "karşılama",
    "soru_cevap",
    "hikaye_anlatma",
    "genel_konuşma",
    "planlama",
    "çıkış",
]
BOT_EYLEMLERİ = [
    "bilgilendirici_yanıt",
    "yaratıcı_yanıt",
    "açıklayıcı_soru",
    "diyalog_durumunu_değiştir",
    "yeni_konu_başlat",
    "plan_oluştur",
    "planı_uygula",
]

# --- NLP Tools ---
tfidf_vektörleştirici = TfidfVectorizer()

# --- Error Counter ---
hata_sayacı = 0

# --- Active User Counter ---
aktif_kullanıcılar = 0

# --- Response Time Histogram ---
yanıt_süresi_histogramı = []

# --- Response Time Summary ---
yanıt_süresi_özeti = []

# --- Feedback Counter ---
geri_bildirim_sayısı = 0

# --- Database Ready Flag ---
veritabanı_hazır = False

# --- Database Lock ---
veritabanı_kilidi = None


# --- Knowledge Graph Class ---
class BilgiGrafiği:
    def __init__(self):
        self.grafik = nx.DiGraph()
        self.düğüm_kimliği_sayacı = 0

    def _düğüm_kimliği_oluştur(self):
        self.düğüm_kimliği_sayacı += 1
        return str(self.düğüm_kimliği_sayacı)

    def düğüm_ekle(self, düğüm_türü, düğüm_kimliği=None, veri=None):
        if düğüm_kimliği is None:
            düğüm_kimliği = self._düğüm_kimliği_oluştur()
        self.grafik.add_node(
            düğüm_kimliği, tür=düğüm_türü, veri=veri if veri is not None else {}
        )

    def düğüm_al(self, düğüm_kimliği):
        return self.grafik.nodes.get(düğüm_kimliği)

    def kenar_ekle(self, kaynak_kimliği, ilişki, hedef_kimliği, özellikler=None):
        self.grafik.add_edge(
            kaynak_kimliği,
            hedef_kimliği,
            ilişki=ilişki,
            özellikler=özellikler if özellikler is not None else {},
        )

    def ilgili_düğümleri_al(self, düğüm_kimliği, ilişki=None, yön="giden"):
        ilgili_düğümler = []
        if yön == "giden" or yön == "her ikisi":
            for komşu in self.grafik.successors(düğüm_kimliği):
                kenar_verisi = self.grafik.get_edge_data(düğüm_kimliği, komşu)
                if ilişki is None or kenar_verisi["ilişki"] == ilişki:
                    ilgili_düğümler.append(self.düğüm_al(komşu))
        if yön == "gelen" or yön == "her ikisi":
            for komşu in self.grafik.predecessors(düğüm_kimliği):
                kenar_verisi = self.grafik.get_edge_data(komşu, düğüm_kimliği)
                if ilişki is None or kenar_verisi["ilişki"] == ilişki:
                    ilgili_düğümler.append(self.düğüm_al(komşu))
        return ilgili_düğümler

    async def metni_göm(self, metin: str) -> List[float]:
        """Generates an embedding for the given text using Gemini."""
        istem = f"""
        Generate a numerical vector representation (embedding) for the following text:

        ```
        {metin}
        ```

        The embedding should capture the meaning and context of the text. 
        Return the embedding as a JSON array of floating-point numbers.
        """
        try:
            yanıt = await gemini_ile_yanıt_oluştur(istem)
            gömme = json.loads(yanıt)
            return gömme
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error occurred while generating embedding: {e}")
            return [0.0] * 768  # Return a default embedding of zeros

    async def düğümleri_ara(self, sorgu, üst_k=3, düğüm_türü=None):
        sorgu_gömmesi = await self.metni_göm(sorgu)
        sonuçlar = []
        for düğüm_kimliği, düğüm_verisi in self.grafik.nodes(data=True):
            if düğüm_türü is None or düğüm_verisi["tür"] == düğüm_türü:
                düğüm_gömmesi = await self.metni_göm(str(düğüm_verisi["veri"]))
                benzerlik = cosine_similarity([sorgu_gömmesi], [düğüm_gömmesi])[0][
                    0
                ]
                sonuçlar.append(
                    (düğüm_verisi["tür"], düğüm_kimliği, düğüm_verisi["veri"], benzerlik)
                )

        sonuçlar.sort(key=lambda x: x[3], reverse=True)
        return sonuçlar[:üst_k]

    def düğümü_güncelle(self, düğüm_kimliği, yeni_veri):
        self.grafik.nodes[düğüm_kimliği]["veri"].update(yeni_veri)

    def düğümü_sil(self, düğüm_kimliği):
        self.grafik.remove_node(düğüm_kimliği)

    def dosyaya_kaydet(self, dosya_adı):
        with open(dosya_adı, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def dosyadan_yükle(dosya_adı):
        with open(dosya_adı, "rb") as f:
            return pickle.load(f)


# --- Create/Load Knowledge Graph ---
bilgi_grafiği = BilgiGrafiği()
if os.path.exists(BİLGİ_GRAFİĞİ_DOSYASI):
    bilgi_grafiği = BilgiGrafiği.dosyadan_yükle(BİLGİ_GRAFİĞİ_DOSYASI)


# --- Long-Term Memory Functions ---
async def uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, bilgi_türü, bilgi):
    bilgi_grafiği.düğüm_ekle(
        bilgi_türü, veri={"kullanıcı_kimliği": kullanıcı_kimliği, "bilgi": bilgi}
    )
    bilgi_grafiği.kenar_ekle(
        kullanıcı_kimliği,
        "sahiptir_" + bilgi_türü,
        str(bilgi_grafiği.düğüm_kimliği_sayacı - 1),
    )
    bilgi_grafiği.dosyaya_kaydet(BİLGİ_GRAFİĞİ_DOSYASI)


async def uzun_süreli_hafızadan_al(kullanıcı_kimliği, bilgi_türü, sorgu=None, üst_k=3):
    if sorgu:
        arama_sonuçları = await bilgi_grafiği.düğümleri_ara(
            sorgu, üst_k=üst_k, düğüm_türü=bilgi_türü
        )
        return [
            (düğüm_türü, düğüm_kimliği, düğüm_verisi)
            for düğüm_türü, düğüm_kimliği, düğüm_verisi, skor in arama_sonuçları
        ]
    else:
        ilgili_düğümler = bilgi_grafiği.ilgili_düğümleri_al(
            kullanıcı_kimliği, "sahiptir_" + bilgi_türü
        )
        return [düğüm["veri"]["bilgi"] for düğüm in ilgili_düğümler]


# --- Plan Execution and Monitoring ---
async def plan_adımını_yürüt(
    plan: Dict, adım_indeksi: int, kullanıcı_kimliği: str, mesaj: discord.Message
) -> str:
    adım = plan["adımlar"][adım_indeksi]
    yürütme_istemi = f"""
    You are an AI assistant helping a user carry out a plan.
    Here is the plan step: {adım["açıklama"]}
    The user said: {mesaj.content}

    If the user's message indicates they are ready to proceed with this step, provide a simulated response as if they completed it.
    If the user requests clarification or changes, accept their request and provide helpful information or guidance.
    Be specific and relevant to the plan step.
    """
    try:
        yürütme_yanıtı = await gemini_ile_yanıt_oluştur(
            yürütme_istemi, kullanıcı_kimliği
        )
    except Exception as e:
        logging.error(f"Error occurred while executing plan step: {e}")
        return "An error occurred while trying to execute this step. Please try again later."

    adım["durum"] = "devam_ediyor"
    await uzun_süreli_hafızaya_kaydet(
        kullanıcı_kimliği,
        "plan_uygulama_sonucu",
        {
            "adım_açıklaması": adım["açıklama"],
            "sonuç": "devam_ediyor",
            "zaman_damgası": datetime.now(timezone.utc).isoformat(),
        },
    )
    return yürütme_yanıtı


async def plan_yürütmesini_izle(
    plan: Dict, kullanıcı_kimliği: str, mesaj: discord.Message
) -> str:
    geçerli_adım_indeksi = next(
        (i for i, adım in enumerate(plan["adımlar"]) if adım["durum"] == "devam_ediyor"),
        None,
    )

    if geçerli_adım_indeksi is not None:
        if "bitti" in mesaj.content.lower() or "tamamlandı" in mesaj.content.lower():
            plan["adımlar"][geçerli_adım_indeksi]["durum"] = "tamamlandı"
            await mesaj.channel.send(
                f"Great! Step {geçerli_adım_indeksi + 1} has been completed."
            )
            if geçerli_adım_indeksi + 1 < len(plan["adımlar"]):
                sonraki_adım_yanıtı = await plan_adımını_yürüt(
                    plan, geçerli_adım_indeksi + 1, kullanıcı_kimliği, mesaj
                )
                return f"Moving on to the next step: {sonraki_adım_yanıtı}"
            else:
                return "Congratulations! You have completed all the steps in the plan."
        else:
            return await plan_adımını_yürüt(
                plan, geçerli_adım_indeksi, kullanıcı_kimliği, mesaj
            )


async def plan_oluştur(
    hedef: str,
    tercihler: Dict,
    kullanıcı_kimliği: str,
    mesaj: discord.Message,
) -> Dict:
    planlama_istemi = f"""
    You are an AI assistant specialized in planning.
    A user needs help with the following goal: {hedef}
    What the user said about the plan: {tercihler.get('kullanıcı_girdisi')}

    Based on this information, create a detailed and actionable plan by identifying key steps and considerations.
    Ensure the plan is:
    * **Specific:** Each step should be clearly defined.
    * **Measurable:** Add ways to track progress.
    * **Achievable:** Steps should be realistic and actionable.
    * **Relevant:** Align with the user's goal.
    * **Time-bound:** Include estimated timelines or deadlines.

    Analyze potential risks and dependencies for each step.

    Format the plan as a JSON object with the following structure:
    ```json
    {{
      "hedef": "User's goal",
      "adımlar": [
        {{
          "açıklama": "Step description",
          "son_tarih": "Optional deadline for the step",
          "bağımlılıklar": ["List of dependencies (other step descriptions)"],
          "riskler": ["List of potential risks"],
          "durum": "waiting"
        }},
        // ... more steps
      ],
      "tercihler": {{
        // User preferences related to the plan
      }}
    }}
    ```
    """
    try:
        plan_metni = await gemini_ile_yanıt_oluştur(planlama_istemi, kullanıcı_kimliği)
        plan = json.loads(plan_metni)
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"Error occurred while parsing JSON or creating plan: {e}")
        return {"hedef": hedef, "adımlar": [], "tercihler": tercihler}

    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan", plan)
    return plan


async def planı_değerlendir(plan: Dict, kullanıcı_kimliği: str) -> Dict:
    değerlendirme_istemi = f"""
    You are an AI assistant tasked with evaluating a plan, including identifying potential risks and dependencies.
    Here is the plan:

    Goal: {plan["hedef"]}
    Steps:
    {json.dumps(plan["adımlar"], indent=2)}

    Evaluate this plan based on the following criteria:
    * **Feasibility:** Is the plan realistically achievable?
    * **Completeness:** Does the plan cover all necessary steps?
    * **Efficiency:** Is the plan optimally structured? Are there unnecessary or redundant steps?
    * **Risks:** Analyze the risks identified for each step. Are they significant? How can they be mitigated?
    * **Dependencies:** Are the dependencies between steps clear and well defined? Are there potential conflicts or bottlenecks?
    * **Improvements:** Suggest any improvements or alternative approaches considering the risks and dependencies.

    Provide a structured evaluation summarizing your assessment for each criterion. Be as specific as possible in your analysis.
    """
    try:
        değerlendirme_metni = await gemini_ile_yanıt_oluştur(
            değerlendirme_istemi, kullanıcı_kimliği
        )
    except Exception as e:
        logging.error(f"Error occurred while evaluating plan: {e}")
        return {
            "değerlendirme_metni": "An error occurred while evaluating the plan. Please try again later."
        }

    await uzun_süreli_hafızaya_kaydet(
        kullanıcı_kimliği, "plan_değerlendirmesi", değerlendirme_metni
    )
    değerlendirme = {"değerlendirme_metni": değerlendirme_metni}
    return değerlendirme


async def planı_doğrula(plan: Dict, kullanıcı_kimliği: str) -> Tuple[bool, str]:
    doğrulama_istemi = f"""
    You are an AI assistant specialized in evaluating the feasibility and safety of plans. 
    Carefully analyze the following plan and identify any potential issues, flaws, or missing information that could lead to failure or undesirable outcomes.

    Goal: {plan["hedef"]}
    Steps:
    {json.dumps(plan["adımlar"], indent=2)}

    Consider the following points:
    * **Clarity and Specificity:** Are the steps clear and specific enough to be actionable?
    * **Realism and Feasibility:** Are the steps realistic and achievable considering the user's context and resources?
    * **Dependencies:** Are the dependencies between steps clearly stated and logical? Are there cyclic dependencies?
    * **Time Constraints:** Are the deadlines realistic and achievable? Are there potential time conflicts?
    * **Resource Availability:** Are the necessary resources available for each step?
    * **Risk Assessment:** Are potential risks sufficiently identified and analyzed? Are there mitigation strategies?
    * **Safety and Ethics:** Does the plan comply with safety and ethical standards? Are there potential negative outcomes?

    Provide a detailed analysis of the plan highlighting any weaknesses or areas for improvement. 
    Indicate if the plan is solid and well-structured, or provide specific recommendations for making it more robust and effective.
    """

    try:
        doğrulama_sonucu = await gemini_ile_yanıt_oluştur(
            doğrulama_istemi, kullanıcı_kimliği
        )
    except Exception as e:
        logging.error(f"Error occurred while validating plan: {e}")
        return (
            False,
            "An error occurred while validating the plan. Please try again later.",
        )

    logging.info(f"Plan validation result: {doğrulama_sonucu}")

    if "valid" in doğrulama_sonucu.lower():
        return True, doğrulama_sonucu
    else:
        return False, doğrulama_sonucu


async def plan_geri_bildirimini_işle(kullanıcı_kimliği: str, mesaj: str) -> str:
    geri_bildirim_istemi = f"""
    You are an AI assistant analyzing user feedback on a plan.
    The user said: {mesaj}

    Is the user accepting the plan?
    Respond with "ACCEPT" if yes.
    If no, identify parts of the plan the user wants to change and suggest how the plan might be revised.
    """
    try:
        geri_bildirim_analizi = await gemini_ile_yanıt_oluştur(
            geri_bildirim_istemi, kullanıcı_kimliği
        )
        if "accept" in geri_bildirim_analizi.lower():
            return "accept"
        else:
            return geri_bildirim_analizi  # Return suggestions for revisions
    except Exception as e:
        logging.error(f"Error occurred while processing plan feedback: {e}")
        return "An error occurred while processing your feedback. Please try again later."


# --- User Interest Determination ---
kullanıcı_mesaj_tamponu = defaultdict(list)


async def kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği: str, mesaj: str):
    kullanıcı_mesaj_tamponu[kullanıcı_kimliği].append(mesaj)
    if len(kullanıcı_mesaj_tamponu[kullanıcı_kimliği]) >= 5:  # Process every 5 messages
        mesajlar = kullanıcı_mesaj_tamponu[kullanıcı_kimliği]
        kullanıcı_mesaj_tamponu[kullanıcı_kimliği] = []  # Clear the buffer
        gömmeler = [
            await bilgi_grafiği.metni_göm(mesaj) for mesaj in mesajlar
        ]  # Generate embeddings using Gemini
        konu_sayısı = 3  # Adjust number of topics
        kmeans = KMeans(n_clusters=konu_sayısı, random_state=0)
        kmeans.fit(gömmeler)
        konu_etiketleri = kmeans.labels_

        for i, mesaj in enumerate(mesajlar):
            kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"].append(
                {"mesaj": mesaj, "gömme": gömmeler[i], "konu": konu_etiketleri[i]}
            )
        kullanıcı_profillerini_kaydet()


async def yeni_konu_öner(kullanıcı_kimliği: str) -> str:
    if kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"]:
        ilgiler = kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"]
        konu_sayıları = defaultdict(int)
        for ilgi in ilgiler:
            konu_sayıları[ilgi["konu"]] += 1
        en_sık_konu = max(konu_sayıları, key=konu_sayıları.get)
        önerilen_ilgi = random.choice(
            [ilgi for ilgi in ilgiler if ilgi["konu"] == en_sık_konu]
        )
        return (
            f"Hey, maybe we could talk more about '{önerilen_ilgi['mesaj']}'? I'd love to hear your thoughts."
        )
    else:
        return "I'm not sure what to talk about next. What are you interested in?"


# --- Advanced Dialogue State Monitoring ---
class DiyalogDurumuİzleyici:
    durumlar = {
        "karşılama": {"giriş_eylemi": "kullanıcıyı_karşıla"},
        "genel_konuşma": {},  # No specific entry action
        "hikaye_anlatma": {},
        "soru_cevap": {},
        "planlama": {"giriş_eylemi": "planlamaya_başla"},
        "çıkış": {"giriş_eylemi": "çıkışı_işle"},
        "hata": {"giriş_eylemi": "hatayı_yönet"},  # Add an error state
    }

    def __init__(self):
        self.makine = Machine(
            model=self,
            states=list(DiyalogDurumuİzleyici.durumlar.keys()),
            initial="karşılama",
        )
        # Define conditional transitions
        self.makine.add_transition(
            "karşıla",
            "karşılama",
            "genel_konuşma",
            conditions=["kullanıcı_merhaba_diyor"],
        )
        self.makine.add_transition(
            "soru_sor", "*", "soru_cevap", conditions=["kullanıcı_soru_soruyor"]
        )
        self.makine.add_transition(
            "hikaye_anlat",
            "*",
            "hikaye_anlatma",
            conditions=["kullanıcı_hikaye_istiyor"],
        )
        self.makine.add_transition(
            "planla", "*", "planlama", conditions=["kullanıcı_plan_istiyor"]
        )
        self.makine.add_transition(
            "çıkışı_işle", "*", "çıkış", conditions=["kullanıcı_çıkış_istiyor"]
        )
        self.makine.add_transition("hata", "*", "hata")  # Add transition to error state

    def kullanıcı_merhaba_diyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            karşılama in kullanıcı_girdisi.lower()
            for karşılama in ["merhaba", "selam", "hey"]
        )

    def kullanıcı_soru_soruyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            soru_kelimesi in kullanıcı_girdisi.lower()
            for soru_kelimesi in ["ne", "kim", "nerede", "ne zaman", "nasıl", "neden"]
        )

    def kullanıcı_hikaye_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            hikaye_anahtar_kelimesi in kullanıcı_girdisi.lower()
            for hikaye_anahtar_kelimesi in [
                "bana bir hikaye anlat",
                "bir hikaye anlat",
                "hikaye zamanı",
            ]
        )

    def kullanıcı_plan_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            plan_anahtar_kelimesi in kullanıcı_girdisi.lower()
            for plan_anahtar_kelimesi in [
                "bir plan yap",
                "bir şey planla",
                "planlamama yardım et",
            ]
        )

    def kullanıcı_çıkış_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            çıkış in kullanıcı_girdisi.lower()
            for çıkış in ["hoşçakal", "görüşürüz", "sonra görüşürüz", "çıkış"]
        )

    def kullanıcıyı_karşıla(self, kullanıcı_kimliği: str) -> str:
        karşılamalar = [
            f"Merhaba <@{kullanıcı_kimliği}>! Bugün sana nasıl yardımcı olabilirim?",
            f"Selam <@{kullanıcı_kimliği}>, aklında ne var?",
            f"Hey <@{kullanıcı_kimliği}>! Senin için ne yapabilirim?",
        ]
        return random.choice(karşılamalar)

    def planlamaya_başla(self, kullanıcı_kimliği: str) -> str:
        kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"] = {}
        return "Tamam, planlamaya başlayalım. Neyi planlamaya çalışıyorsun?"

    def çıkışı_işle(self, kullanıcı_kimliği: str) -> str:
        çıkışlar = [
            f"Hoşçakal, <@{kullanıcı_kimliği}>! İyi günler!",
            f"Görüşürüz, <@{kullanıcı_kimliği}>!",
            f"Sonra konuşuruz, <@{kullanıcı_kimliği}>!",
            f"Çıkış yapılıyor, <@{kullanıcı_kimliği}>!",
        ]
        return random.choice(çıkışlar)

    def hatayı_yönet(self, kullanıcı_kimliği: str) -> str:
        return "Anlamadım. Lütfen isteğinizi yeniden ifade eder misiniz?"

    async def diyalog_eylemini_sınıflandır(self, kullanıcı_girdisi: str) -> str:
        """Classifies the dialogue action using Gemini."""
        for deneme in range(3):  # Retry up to 3 times
            try:
                istem = (
                    f"Classify the following user input into one of the dialogue actions: "
                    f"karşılama, soru_cevap, hikaye_anlatma, genel_konuşma, planlama, çıkış.\n\n"
                    f"User input: {kullanıcı_girdisi}\n\n"
                    f"Classify the dialogue action by stating your answer as a single word on the first line:"
                )
                logging.info(f"Dialogue Action Classification Request: {istem}")
                yanıt = await gemini_ile_yanıt_oluştur(istem, None)
                diyalog_eylemi = yanıt.strip().split("\n")[0].lower()
                logging.info(
                    f"Raw Gemini response for Dialogue Action Classification: {yanıt}"
                )
                logging.info(f"Extracted Dialogue Action: {diyalog_eylemi}")
                return diyalog_eylemi
            except Exception as e:
                logging.error(
                    f"Error occurred while extracting dialogue action from Gemini response: {e}, Attempt: {deneme + 1}"
                )
                await asyncio.sleep(2)  # Wait before retrying

        # Transition to error state after 3 attempts
        self.makine.trigger("hata")
        return self.makine.state

    async def durumu_geçiş_yap(
        self,
        geçerli_durum: str,
        kullanıcı_girdisi: str,
        kullanıcı_kimliği: str,
        konuşma_geçmişi: List,
    ) -> str:
        """Transitions the dialogue state based on user input and conditions."""
        if self.makine.trigger("karşıla", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("soru_sor", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("hikaye_anlat", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("planla", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("çıkışı_işle", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        # Default transition if no condition is met
        return "genel_konuşma"


# --- Initialize the Dialogue State Tracker ---
diyalog_durumu_izleyici = DiyalogDurumuİzleyici()

# --- Gemini Rate Limit Handling ---
ORAN_SINIRI_DAKİKADA_GEMINI = 60
ORAN_SINIRI_PENCERESİ_GEMINI = 60
kullanıcı_son_istek_zamanı_gemini = defaultdict(lambda: 0)
global_son_istek_zamanı_gemini = 0
global_istek_sayısı_gemini = 0


@backoff.on_exception(
    backoff.expo, (requests.exceptions.RequestException, GoogleAPIError), max_time=600
)
async def gemini_ile_yanıt_oluştur(istem: str, kullanıcı_kimliği: str = None) -> str:
    """
    Handles rate limits and retries while generating a response with Gemini.
    
    Args:
        istem: The prompt to send to Gemini.
        kullanıcı_kimliği: The ID of the user making the request.

    Returns:
        The response from Gemini.
    """
    global global_son_istek_zamanı_gemini, global_istek_sayısı_gemini
    geçerli_zaman = time.time()

    # Global rate limit for Gemini
    if geçerli_zaman - global_son_istek_zamanı_gemini < ORAN_SINIRI_PENCERESİ_GEMINI:
        global_istek_sayısı_gemini += 1
        if global_istek_sayısı_gemini > ORAN_SINIRI_DAKİKADA_GEMINI:
            bekleme_süresi = ORAN_SINIRI_PENCERESİ_GEMINI - (
                geçerli_zaman - global_son_istek_zamanı_gemini
            )
            await asyncio.sleep(bekleme_süresi)
            global_istek_sayısı_gemini = 0
    else:
        global_istek_sayısı_gemini = 0
    global_son_istek_zamanı_gemini = geçerli_zaman

    # User-specific rate limit for Gemini (if kullanıcı_kimliği is provided)
    if kullanıcı_kimliği:
        son_istekten_bu_yana_geçen_süre = (
            geçerli_zaman - kullanıcı_son_istek_zamanı_gemini[kullanıcı_kimliği]
        )
        if (
            son_istekten_bu_yana_geçen_süre
            < ORAN_SINIRI_PENCERESİ_GEMINI / ORAN_SINIRI_DAKİKADA_GEMINI
        ):
            bekleme_süresi = (
                ORAN_SINIRI_PENCERESİ_GEMINI / ORAN_SINIRI_DAKİKADA_GEMINI
                - son_istekten_bu_yana_geçen_süre
            )
            await asyncio.sleep(bekleme_süresi)
            kullanıcı_son_istek_zamanı_gemini[kullanıcı_kimliği] = time.time()

    # Generate response with Gemini
    if isinstance(istem, str):
        response = model.generate_content(istem)
        logging.info(f"Raw Gemini response: {response}")
        return response.text
    elif isinstance(istem, list) and len(istem) == 2 and isinstance(istem[1], Image.Image):
        prompt, image = istem
        response = model.generate_content([prompt, {"image": image}])
        logging.info(f"Raw Gemini response (with image): {response}")
        return response.text  # Assuming the response still primarily contains text
    else:
        raise TypeError("Invalid input format for gemini_ile_yanıt_oluştur()")

# --- Gemini Search and Summarization ---
async def gemini_arama_ve_özetleme(sorgu: str) -> Optional[str]:
    """
    Searches the web using DuckDuckGo and summarizes the results with Gemini.
    
    Args:
        sorgu: The user's search query.

    Returns:
        A summarized version of the search results or None if an error occurs.
    """
    try:
        ddg = AsyncDDGS()
        arama_sonuçları = await asyncio.to_thread(ddg.text, sorgu, max_results=3)

        arama_sonuçları_metni = ""
        for indeks, sonuç in enumerate(arama_sonuçları):
            arama_sonuçları_metni += (
                f'[{indeks}] Başlık: {sonuç["title"]}\nÖzet: {sonuç["body"]}\n\n'
            )

        istem = (
            f"You are a helpful AI assistant. A user asked about '{sorgu}'. Here are some relevant web search results:\n\n"
            f"{arama_sonuçları_metni}\n\n"
            f"Please provide a concise and informative summary of these search results."
        )

        yanıt = await gemini_ile_yanıt_oluştur(istem)
        return yanıt
    except Exception as e:
        logging.error(f"Error during Gemini search and summarization: {e}")
        return None


# --- URL Extraction from Description ---
async def açıklamadan_url_çıkar(açıklama: str) -> Optional[str]:
    """
    Extracts a URL from a description using DuckDuckGo searches.
    Prioritizes links from YouTube, Twitch, Instagram, and Twitter.

    Args:
        açıklama: The description to search for a URL.

    Returns:
        The URL extracted from the description or None if no URL is found.
    """
    arama_sorgusu = (
        f"{açıklama} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"
    )

    async with aiohttp.ClientSession() as oturum:
        async with oturum.get(
            f"https://duckduckgo.com/html/?q={arama_sorgusu}"
        ) as yanıt:
            html = await yanıt.text()
            soup = BeautifulSoup(html, "html.parser")

            ilk_sonuç = soup.find("a", class_="result__a")
            if ilk_sonuç:
                return ilk_sonuç["href"]
            else:
                return None


# --- Clean Up URL Format ---
async def url_temizle(url: str, açıklama: str = None) -> Optional[str]:
    """
    Cleans a URL, adding https:// if it's missing and validating it.

    Args:
        url: The URL to clean.
        açıklama: Optional description that can be used to refine the search if the URL is invalid.

    Returns:
        The cleaned URL or None if the URL is invalid and cannot be cleaned.
    """
    if url is None:
        return None

    # 1. Normalize the URL (lowercase, strip whitespace)
    temizlenmiş_url = url.lower().strip()

    # 2. Add https:// if the protocol is missing
    if not temizlenmiş_url.startswith(("https://", "http://")):
        temizlenmiş_url = "https://" + temizlenmiş_url

    # 3. Handle specific website patterns (You can add more as needed)
    if "youtube.com" in temizlenmiş_url and not "www.youtube.com" in temizlenmiş_url:
        temizlenmiş_url = re.sub(
            r"(youtube\.com/)(.*)", r"www.youtube.com/\2", temizlenmiş_url
        )
    elif "twitch.tv" in temizlenmiş_url and not "www.twitch.tv" in temizlenmiş_url:
        temizlenmiş_url = re.sub(
            r"(twitch\.tv/)(.*)", r"www.twitch.tv/\2", temizlenmiş_url
        )
    elif (
        "instagram.com" in temizlenmiş_url
        and not "www.instagram.com" in temizlenmiş_url
    ):
        temizlenmiş_url = re.sub(
            r"(instagram\.com/)(.*)", r"www.instagram.com/\2", temizlenmiş_url
        )
    elif "twitter.com" in temizlenmiş_url and not "www.twitter.com" in temizlenmiş_url:
        temizlenmiş_url = re.sub(
            r"(twitter\.com/)(.*)", r"www.twitter.com/\2", temizlenmiş_url
        )

    # 4. Remove unnecessary characters (preserve query parameters)
    temizlenmiş_url = re.sub(
        r"^[^:]+://([^/]+)(.*)$", r"\1\2", temizlenmiş_url
    )  # Remove domain and path
    temizlenmiş_url = re.sub(
        r"[^a-zA-Z0-9./?=-]", "", temizlenmiş_url
    )  # Remove invalid characters

    # 5. Check if the cleaned URL is valid
    try:
        yanıt = requests.get(temizlenmiş_url)
        if yanıt.status_code == 200:
            return temizlenmiş_url
        else:
            logging.warning(
                f"Cleaned URL ({temizlenmiş_url}) is not valid (status code: {yanıt.status_code})."
            )
            return None
    except requests.exceptions.RequestException:
        logging.warning(f"Cleaned URL ({temizlenmiş_url}) is not valid.")
        return None


# --- Complex Dialogue Manager ---
async def karmaşık_diyalog_yöneticisi(
    kullanıcı_profilleri: Dict, kullanıcı_kimliği: str, mesaj: discord.Message
) -> str:
    """Manages complex dialogue flows, particularly for planning."""
    if kullanıcı_profilleri[kullanıcı_kimliği]["diyalog_durumu"] == "planlama":
        if "aşama" not in kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]:
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                "aşama"
            ] = "ilk_istek"

        if (
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"]
            == "ilk_istek"
        ):
            (
                hedef,
                sorgu_türü,
            ) = await hedefi_çıkar(kullanıcı_profilleri[kullanıcı_kimliği]["sorgu"])
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["hedef"] = hedef
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                "sorgu_türü"
            ] = sorgu_türü
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                "aşama"
            ] = "bilgi_toplama"
            return await açıklayıcı_sorular_sor(hedef, sorgu_türü)

        elif (
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"]
            == "bilgi_toplama"
        ):
            await planlama_bilgisini_işle(kullanıcı_kimliği, mesaj)
            if await yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği):
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                    "aşama"
                ] = "plan_oluşturma"
                plan = await plan_oluştur(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["hedef"],
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                        "tercihler"
                    ],
                    kullanıcı_kimliği,
                    mesaj,
                )
                geçerli_mi, doğrulama_sonucu = await planı_doğrula(
                    plan, kullanıcı_kimliği
                )
                if geçerli_mi:
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                        "plan"
                    ] = plan
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                        "aşama"
                    ] = "planı_sunma"
                    return await planı_sun_ve_geri_bildirim_iste(plan)
                else:
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                        "aşama"
                    ] = "bilgi_toplama"
                    return (
                        f"There are some issues with the plan: {doğrulama_sonucu} Please provide more information or adjust your preferences."
                    )
            else:
                return await daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği)

        elif (
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"]
            == "planı_sunma"
        ):
            geri_bildirim_sonucu = await plan_geri_bildirimini_işle(
                kullanıcı_kimliği, mesaj.content
            )
            if geri_bildirim_sonucu == "accept":
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                    "aşama"
                ] = "planı_değerlendirme"
                değerlendirme = await planı_değerlendir(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"],
                    kullanıcı_kimliği,
                )
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                    "değerlendirme"
                ] = değerlendirme
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                    "aşama"
                ] = "planı_yürütme"
                ilk_yürütme_mesajı = await plan_adımını_yürüt(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"],
                    0,
                    kullanıcı_kimliği,
                    mesaj,
                )
                return (
                    await yanıt_oluştur(
                        kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                            "plan"
                        ],
                        değerlendirme,
                        {},
                        kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                            "tercihler"
                        ],
                    )
                    + "\n\n"
                    + ilk_yürütme_mesajı
                )
            else:
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"][
                    "aşama"
                ] = "bilgi_toplama"
                return (
                    f"Okay, let's revise the plan. Here are some suggestions: {geri_bildirim_sonucu} What changes would you like to make?"
                )

        elif (
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"]
            == "planı_yürütme"
        ):
            yürütme_sonucu = await plan_yürütmesini_izle(
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"],
                kullanıcı_kimliği,
                mesaj,
            )
            return yürütme_sonucu


# --- Planning Helper Functions ---
async def açıklayıcı_sorular_sor(hedef: str, sorgu_türü: str) -> str:
    return (
        "I need some more details to create an effective plan. Could you please tell me:\n"
        f"- What is the desired outcome of this plan?\n"
        f"- What are the key steps or milestones involved?\n"
        f"- Are there any constraints or limitations I should be aware of?\n"
        f"- What resources or tools are available?\n"
        f"- What is the timeline for completing this plan?"
    )


async def planlama_bilgisini_işle(kullanıcı_kimliği: str, mesaj: discord.Message):
    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"][
        "kullanıcı_girdisi"
    ] = mesaj.content


async def yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği: str) -> bool:
    return (
        "kullanıcı_girdisi"
        in kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]
    )


async def daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği: str) -> str:
    return (
        "Please provide more details to help me create a better plan. "
        "For example, more information about steps, constraints, resources, or the time frame."
    )


async def planı_sun_ve_geri_bildirim_iste(plan: Dict) -> str:
    plan_metni = ""
    for i, adım in enumerate(plan["adımlar"]):
        plan_metni += f"{i + 1}. {adım['açıklama']}\n"
    return (
        f"Based on your input, a draft plan looks like this:\n\n{plan_metni}\n\n"
        f"What do you think? Are there any changes you would like to make? (Type 'accept' to proceed)"
    )


async def yanıt_oluştur(plan, değerlendirme, ek_bilgi, tercihler) -> str:
    """
    Creates a user-friendly response based on the plan, evaluation, additional information, and preferences.
    """
    yanıt = f"I've created a plan for your goal: **{plan['hedef']}**\n\n"

    yanıt += "**Steps:**\n"
    for i, adım in enumerate(plan["adımlar"]):
        yanıt += f"{i + 1}. {adım['açıklama']}"
        if "son_tarih" in adım:
            yanıt += f" (Deadline: {adım['son_tarih']})"
        yanıt += "\n"

    if değerlendirme:
        yanıt += f"\n**Evaluation:**\n{değerlendirme.get('değerlendirme_metni', '')}\n"

    if ek_bilgi:
        yanıt += "\n**Additional Information:**\n"
        for bilgi_türü, bilgi in ek_bilgi.items():
            yanıt += f"- {bilgi_türü}: {bilgi}\n"

    if tercihler:
        yanıt += "\n**Your Preferences:**\n"
        for tercih_adı, tercih_değeri in tercihler.items():
            yanıt += f"- {tercih_adı}: {tercih_değeri}\n"

    return yanıt


# --- Goal Extraction ---
async def hedefi_çıkar(sorgu: str) -> Tuple[str, str]:
    """Extracts the user's goal from their query."""
    istem = f"""
    You are an AI assistant capable of understanding user goals.
    What is the user trying to achieve with the following query?

    User Query: {sorgu}

    Please specify the goal in a concise sentence.
    """
    try:
        hedef = await gemini_ile_yanıt_oluştur(istem, None)
    except Exception as e:
        logging.error(f"Error occurred while extracting the goal: {e}")
        return (
            "I couldn't understand your goal. Please express it differently.",
            "general",
        )
    return hedef.strip(), "general"


# --- Multi-stage Sentiment Analysis ---
async def çok_aşamalı_duygu_analizi(sorgu: str, kullanıcı_kimliği: str) -> Dict:
    """Performs sentiment analysis using multiple methods."""
    sonuçlar = []

    # TextBlob sentiment analysis
    try:
        blob = TextBlob(sorgu)
        textblob_sentiment = blob.sentiment.polarity
        sonuçlar.append(textblob_sentiment)
    except Exception as e:
        logging.error(f"Error in TextBlob sentiment analysis: {str(e)}")

    # VADER sentiment analysis
    try:
        vader = SentimentIntensityAnalyzer()
        vader_sentiment = vader.polarity_scores(sorgu)
        sonuçlar.append(vader_sentiment["compound"])
    except Exception as e:
        logging.error(f"Error in VADER sentiment analysis: {str(e)}")

    # Transformer-based sentiment analysis (requires transformers library)
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        transformer_sentiment = sentiment_pipeline(sorgu)[0]
        sonuçlar.append(
            transformer_sentiment["score"]
            if transformer_sentiment["label"] == "POSITIVE"
            else -transformer_sentiment["score"]
        )
    except ImportError:
        logging.warning(
            "Transformers library not found. Transformer-based sentiment analysis skipped."
        )
    except Exception as e:
        logging.error(f"Error in transformer sentiment analysis: {str(e)}")

    # Gemini-based sentiment analysis
    try:
        duygu_istemi = f"Analyze the sentiment and intensity of: {sorgu}. Return only the sentiment value as a float between -1 and 1."
        gemini_sentiment = await gemini_ile_yanıt_oluştur(
            duygu_istemi, kullanıcı_kimliği
        )
        sentiment_match = re.search(r"-?\d+(\.\d+)?", gemini_sentiment)
        if sentiment_match:
            gemini_score = float(sentiment_match.group())
            sonuçlar.append(gemini_score)
        else:
            logging.error(
                f"Unable to extract sentiment value from Gemini response: {gemini_sentiment}"
            )
    except Exception as e:
        logging.error(f"Error in Gemini sentiment analysis: {str(e)}")

    # Calculate average sentiment
    if sonuçlar:
        ortalama_duygu = np.mean(sonuçlar)
    else:
        logging.error("No valid sentiment scores obtained")
        ortalama_duygu = 0.0

    return {
        "duygu_etiketi": "olumlu"
        if ortalama_duygu > 0.05
        else "olumsuz"
        if ortalama_duygu < -0.05
        else "nötr",
        "duygu_yoğunluğu": abs(ortalama_duygu),
    }

# --- Error Tracker Decorator ---
def error_tracker(func):
    """Decorator to track and log errors in asynchronous functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            # You might want to handle the error differently here, like sending a message to the user
            # For now, we'll just re-raise the exception
            raise

    return wrapper

# --- Image Generation Intent Detection ---
async def image_generation_intent_detection(kullanıcı_kimliği: str, mesaj: str) -> Optional[str]:
    """Detects if the user wants to generate an image and extracts the image description."""
    istem = f"""
    You are an AI assistant helping to detect user intent.
    The user said: {mesaj}

    Does the user want to generate an image?
    If yes, return the description of the image they want to create.
    If no, return "NO_IMAGE_GENERATION". 
    """
    try:
        yanıt = await gemini_ile_yanıt_oluştur(istem, kullanıcı_kimliği)
        if yanıt.strip() == "NO_IMAGE_GENERATION":
            return None
        else:
            return yanıt.strip()  # The image description
    except Exception as e:
        logging.error(f"Error occurred during image generation intent detection: {e}")
        return None

# --- Generate and Send Image ---
async def generate_and_send_image(description: str, mesaj: discord.Message):
    """Generates an image using FLUX.1-schnell and sends it to the user."""
    await mesaj.channel.send(
        "Image generation in progress... This might take a few minutes."
    )
    try:
        # Generate image with negative prompts
        full_prompt = f"{' '.join(negative_prompts)} {description}"
        generated_image = image_generator(full_prompt, guidance_scale=9, num_inference_steps=50).images[0] 

        generated_image.save("generated_image.png")
        await mesaj.channel.send(file=discord.File("generated_image.png"))
    except Exception as e:
        logging.error(f"Error occurred during image generation: {e}")
        await mesaj.channel.send(
            "I'm sorry, I couldn't generate an image based on your description. Please try again."
        )

# --- Image Analysis ---
async def analyze_image(
    image: Image.Image, user_id: str
) -> Tuple[str, str, List[str], List[str], str, List[float]]:
    """Analyzes an image using a combination of models and APIs."""
    analysis_results = f"Image Analysis Results:\n\n"

    # 1. Image Description (using Gemini)
    description_prompt = f"Describe this image in detail."
    image_description = await gemini_ile_yanıt_oluştur(
        [description_prompt, image], user_id
    )
    analysis_results += f"Description: {image_description}\n\n"

    # 2. Image Caption (using Gemini)
    caption_prompt = f"Generate a short, engaging caption for this image."
    image_caption = await gemini_ile_yanıt_oluştur([caption_prompt, image], user_id)
    analysis_results += f"Caption: {image_caption}\n\n"

    # 3. Object Detection (replace with your preferred model/API)
    detected_objects = ["object1", "object2"]  
    analysis_results += f"Detected Objects: {', '.join(detected_objects)}\n\n"

    # 4. Dominant Colors (replace with your preferred library/API)
    dominant_colors = ["color1", "color2"] 
    analysis_results += f"Dominant Colors: {', '.join(dominant_colors)}\n\n"

    # 5. Visual Sentiment (replace with your preferred model/API)
    visual_sentiment = "positive" 
    analysis_results += f"Visual Sentiment: {visual_sentiment}\n\n"

    # 6. Image Embedding (using Gemini)
    embedding_prompt = f"Generate an embedding for this image."
    image_embedding = await bilgi_grafiği.metni_göm(
        await gemini_ile_yanıt_oluştur([embedding_prompt, image], user_id)
    )

    return (
        analysis_results,
        image_caption,
        detected_objects,
        dominant_colors,
        visual_sentiment,
        image_embedding,
    )


# --- Reinforcement Learning System (Placeholder) ---
async def pekiştirmeli_öğrenme_sistemi(
    durum: Dict, eylem: str, ödül: float, yeni_durum: Dict
) -> None:
    """
    This function is a placeholder for a more complex reinforcement learning system.
    
    You'll need to implement the actual logic for updating your agent's policy 
    based on the state, action, reward, and new state. 
    
    This could involve using Q-learning, SARSA, or other RL algorithms.
    """
    logging.info(
        f"Reinforcement learning update: State={durum}, Action={eylem}, Reward={ödül}, New State={yeni_durum}"
    )
    # Implement your RL logic here


# --- Complex Reasoning and Response Generation ---
async def çok_gelişmiş_muhakeme_gerçekleştir(
    sorgu: str,
    ilgili_geçmiş: str,
    özetlenmiş_arama: str,
    kullanıcı_kimliği: str,
    mesaj: discord.Message,
    içerik: str,
) -> Tuple[str, str]:
    """
    Performs multi-faceted reasoning using various AI techniques.

    Args:
        sorgu: The user's query.
        ilgili_geçmiş: Relevant conversation history.
        özetlenmiş_arama: Summarized web search results.
        kullanıcı_kimliği: The ID of the user.
        mesaj: The Discord message object.
        içerik: The message content.

    Returns:
        A tuple containing the generated response and the detected sentiment.
    """
    try:
        durum = {"dinamik_bellek_ağı": nx.DiGraph()}
        bağlam = {
            "sorgu": sorgu,
            "ilgili_geçmiş": ilgili_geçmiş,
            "özetlenmiş_arama": özetlenmiş_arama,
            "kullanıcı_kimliği": kullanıcı_kimliği,
            "mesaj": mesaj,
            "zaman_damgası": datetime.now(timezone.utc).isoformat(),
        }

        # --- Reasoning Steps ---
        duygu_analizi = await çok_aşamalı_duygu_analizi(sorgu, kullanıcı_kimliği)
        hedefler = await gelişmiş_hedef_çıkarımı(bağlam, durum)
        bağlam_analizi = await çok_boyutlu_bağlam_analizi(bağlam, durum)
        uzun_vadeli_hedefler = await uzun_vadeli_hedef_izleme_ve_optimizasyon(
            bağlam, durum
        )
        nedensel_çıkarımlar = await nedensel_ve_karşı_olgusal_çıkarım(bağlam, durum)
        eylem_önerileri = await dinamik_eylem_önerisi_oluşturma(bağlam, durum)
        ton_ayarı = await adaptif_kişilik_bazlı_ton_ayarlama(bağlam, durum)
        bilişsel_yük = await bilişsel_yük_değerlendirmesi(bağlam, durum)
        belirsizlik = await belirsizlik_yönetimi(bağlam, durum)
        çok_modlu_analiz = await çok_modlu_içerik_analizi(bağlam, durum)
        durum["dinamik_bellek_ağı"] = await dinamik_bellek_ağı_güncelleme(
            bağlam, durum
        )
        anlam_çıkarımı = await derin_anlam_çıkarımı(bağlam, durum)
        metabilişsel_analiz_sonucu = await metabilişsel_analiz(bağlam, durum)
        etik_değerlendirme_sonucu = await etik_değerlendirme(bağlam, durum)
        yaratıcı_çözümler = await yaratıcı_problem_çözme(bağlam, durum)
        duygusal_zeka = await duygusal_zeka_analizi(bağlam, durum)
        
        # --- Response Generation ---
        yanıt = await gelişmiş_yanıt_oluştur(bağlam, durum)
        
        # (Optional) Log the generated response for debugging
        logging.info(f"Generated Response: {yanıt}")

        # --- Update Reinforcement Learning System (Placeholder) ---
        # This is where you'd call your RL system with the relevant information:
        # await pekiştirmeli_öğrenme_sistemi(durum, eylem, ödül, yeni_durum)

        return yanıt, duygu_analizi["duygu_etiketi"]
    except Exception as e:
        logging.error(
            f"Error in çok_gelişmiş_muhakeme_gerçekleştir: {str(e)}", exc_info=True
        )
        return "An error occurred while processing your request. Please try again.", None


# --- Advanced Reasoning Helper Functions ---
async def çok_aşamalı_duygu_analizi(
    sorgu: str, kullanıcı_kimliği: str
) -> Dict[str, Any]:
    """Simulates a complex multi-stage emotion analysis process."""
    return {"duygu_etiketi": "meraklı", "duygu_yoğunluğu": 0.75}


async def gelişmiş_hedef_çıkarımı(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> list:
    """Simulates advanced goal inference."""
    return ["bilgi_edinme", "problem_çözme"]


async def çok_boyutlu_bağlam_analizi(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates multi-dimensional context analysis."""
    return {
        "konu": "yapay_zeka",
        "karmaşıklık_seviyesi": "yüksek",
        "kullanıcı_uzmanlığı": "orta",
    }


async def uzun_vadeli_hedef_izleme_ve_optimizasyon(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates long-term goal tracking and optimization."""
    return {"ana_hedef": "AI_öğrenme", "ilerleme": 0.6, "tahmini_tamamlanma": "2 ay"}


async def nedensel_ve_karşı_olgusal_çıkarım(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates causal and counterfactual reasoning."""
    return {
        "nedensel_ilişkiler": ["öğrenme_hızı -> bilgi_derinliği"],
        "karşı_olgusal_senaryolar": ["daha_fazla_uygulama -> daha_hızlı_öğrenme"],
    }


async def dinamik_eylem_önerisi_oluşturma(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> list:
    """Simulates dynamic action suggestion."""
    return ["kod_örnekleri_inceleme", "pratik_yapma", "kaynak_okuma"]


async def adaptif_kişilik_bazlı_ton_ayarlama(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates adaptive personality-based tone adjustment."""
    return {"ton": "bilgilendirici_ve_destekleyici", "resmiyet_seviyesi": "orta"}


async def bilişsel_yük_değerlendirmesi(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates cognitive load assessment."""
    return {
        "bilişsel_yük": "orta",
        "karmaşıklık_seviyesi": "yüksek",
        "önerilen_adım": "konuyu_alt_bölümlere_ayırma",
    }


async def belirsizlik_yönetimi(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates uncertainty management."""
    return {"belirsizlik_seviyesi": "düşük", "güven_aralığı": "0.85-0.95"}


async def çok_modlu_içerik_analizi(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates multi-modal content analysis."""
    return {
        "metin_analizi": "tamamlandı",
        "görsel_analiz": "uygulanamaz",
        "ses_analizi": "uygulanamaz",
    }


async def dinamik_bellek_ağı_güncelleme(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> nx.DiGraph:
    """Simulates dynamic memory network update."""
    G = durum["dinamik_bellek_ağı"]
    G.add_edge("yapay_zeka", "makine_öğrenmesi")
    G.add_edge("makine_öğrenmesi", "derin_öğrenme")
    return G


async def derin_anlam_çıkarımı(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates deep semantic inference."""
    return {
        "ana_kavramlar": ["AI", "karmaşık_sistemler", "optimizasyon"],
        "kavram_ilişkileri": ["AI -> karmaşık_sistemler", "karmaşık_sistemler -> optimizasyon"],
    }


async def metabilişsel_analiz(bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates metacognitive analysis."""
    return {
        "öğrenme_stratejisi": "aktif_öğrenme",
        "bilgi_boşlukları": ["ileri_düzey_optimizasyon_teknikleri"],
        "önerilen_yaklaşım": "pratik_uygulama_artırma",
    }


async def etik_değerlendirme(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates ethical evaluation."""
    return {
        "etik_sorunlar": [],
        "öneriler": ["şeffaflık_artırma", "veri_gizliliğine_dikkat"],
    }


async def yaratıcı_problem_çözme(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> list:
    """Simulates creative problem solving."""
    return ["hibrit_model_kullanımı", "transfer_öğrenme_uygulaması"]


async def duygusal_zeka_analizi(
    bağlam: Dict[str, Any], durum: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulates emotional intelligence analysis."""
    return {
        "empati_seviyesi": "yüksek",
        "motivasyon_faktörleri": ["başarı", "merak"],
        "duygusal_destek_stratejisi": "teşvik_edici_geri_bildirim",
    }


async def gelişmiş_yanıt_oluştur(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> str:
    """Generates a comprehensive response based on the reasoning results."""
    sorgu = bağlam["sorgu"]
    istem = f"""
    You are a sophisticated AI assistant. 
    A user asked the following: {sorgu}

    Provide a comprehensive and informative response considering the following analysis:

    Sentiment: {durum.get("duygu_analizi", {}).get("duygu_etiketi", "neutral")}
    Goals: {", ".join(durum.get("hedefler", []))}
    Context: 
        Topic: {durum.get("bağlam_analizi", {}).get("konu", "unknown")}
        Complexity: {durum.get("bağlam_analizi", {}).get("karmaşıklık_seviyesi", "unknown")}
        User Expertise: {durum.get("bağlam_analizi", {}).get("kullanıcı_uzmanlığı", "unknown")}
    Long-Term Goals:
        Main Goal: {durum.get("uzun_vadeli_hedefler", {}).get("ana_hedef", "unknown")}
        Progress: {durum.get("uzun_vadeli_hedefler", {}).get("ilerleme", "unknown")}
        Estimated Completion: {durum.get("uzun_vadeli_hedefler", {}).get("tahmini_tamamlanma", "unknown")}
    Causal Inferences: {", ".join(durum.get("nedensel_çıkarımlar", {}).get("nedensel_ilişkiler", []))}
    Counterfactual Scenarios: {", ".join(durum.get("nedensel_çıkarımlar", {}).get("karşı_olgusal_senaryolar", []))}
    Action Suggestions: {", ".join(durum.get("eylem_önerileri", []))}
    Tone: {durum.get("ton_ayarı", {}).get("ton", "neutral")}
    Formality: {durum.get("ton_ayarı", {}).get("resmiyet_seviyesi", "medium")}
    Cognitive Load: {durum.get("bilişsel_yük", {}).get("bilişsel_yük", "medium")}
    Uncertainty: {durum.get("belirsizlik", {}).get("belirsizlik_seviyesi", "low")}
    Confidence Interval: {durum.get("belirsizlik", {}).get("güven_aralığı", "unknown")}
    Multi-modal Analysis:
        Text Analysis: {durum.get("çok_modlu_analiz", {}).get("metin_analizi", "not performed")}
        Image Analysis: {durum.get("çok_modlu_analiz", {}).get("görsel_analiz", "not performed")}
        Audio Analysis: {durum.get("çok_modlu_analiz", {}).get("ses_analizi", "not performed")}
    Dynamic Memory Network: {json_graph.node_link_data(durum.get("dinamik_bellek_ağı", nx.DiGraph()))}
    Deep Semantic Inference:
        Key Concepts: {", ".join(durum.get("anlam_çıkarımı", {}).get("ana_kavramlar", []))}
        Concept Relationships: {", ".join(durum.get("anlam_çıkarımı", {}).get("kavram_ilişkileri", []))}
    Metacognitive Analysis:
        Learning Strategy: {durum.get("metabilişsel_analiz_sonucu", {}).get("öğrenme_stratejisi", "unknown")}
        Knowledge Gaps: {", ".join(durum.get("metabilişsel_analiz_sonucu", {}).get("bilgi_boşlukları", []))}
        Suggested Approach: {durum.get("metabilişsel_analiz_sonucu", {}).get("önerilen_yaklaşım", "unknown")}
    Ethical Considerations:
        Ethical Issues: {", ".join(durum.get("etik_değerlendirme_sonucu", {}).get("etik_sorunlar", []))}
        Suggestions: {", ".join(durum.get("etik_değerlendirme_sonucu", {}).get("öneriler", []))}
    Creative Solutions: {", ".join(durum.get("yaratıcı_çözümler", []))}
    Emotional Intelligence:
        Empathy Level: {durum.get("duygusal_zeka", {}).get("empati_seviyesi", "unknown")}
        Motivation Factors: {", ".join(durum.get("duygusal_zeka", {}).get("motivasyon_faktörleri", []))}
        Emotional Support Strategy: {durum.get("duygusal_zeka", {}).get("duygusal_destek_stratejisi", "unknown")}

    Ensure the response is clear, concise, and tailored to the user's query and the provided analysis.
    """
    yanıt = await gemini_ile_yanıt_oluştur(istem, bağlam["kullanıcı_kimliği"])
    return yanıt

# --- Database Interaction Functions ---
veritabanı_kuyruğu = asyncio.Queue()


async def init_image_database():
    """Initializes the database for image data."""
    async with aiosqlite.connect(IMAGE_DATABASE) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                analysis_results TEXT,
                image_caption TEXT,
                detected_objects TEXT,
                dominant_colors TEXT,
                visual_sentiment TEXT,
                image_embedding TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        await db.commit()


async def save_image_data(
    user_id: str,
    image_hash: str,
    analysis_results: str,
    image_caption: str,
    detected_objects: List[str],
    dominant_colors: List[str],
    visual_sentiment: str,
    image_embedding: List[float],
):
    """Saves image data and analysis to the database."""
    async with aiosqlite.connect(IMAGE_DATABASE) as db:
        await db.execute(
            """
            INSERT INTO images (user_id, image_hash, analysis_results, image_caption, 
                                detected_objects, dominant_colors, visual_sentiment, image_embedding, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                image_hash,
                analysis_results,
                image_caption,
                json.dumps(detected_objects),
                json.dumps(dominant_colors),
                visual_sentiment,
                json.dumps(image_embedding),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()


async def get_relevant_images(
    user_id: str, query: str
) -> List[Tuple[str, str, float]]:
    """Retrieves relevant image data from the database based on user query."""
    async with aiosqlite.connect(IMAGE_DATABASE) as db:
        cursor = await db.execute(
            """
            SELECT image_hash, analysis_results, image_caption, detected_objects, dominant_colors, visual_sentiment, image_embedding 
            FROM images 
            WHERE user_id = ?
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()

        relevant_images = []
        for (
            image_hash,
            analysis_results,
            image_caption,
            detected_objects_str,
            dominant_colors_str,
            visual_sentiment,
            image_embedding_str,
        ) in rows:
            detected_objects = json.loads(detected_objects_str)
            dominant_colors = json.loads(dominant_colors_str)
            image_embedding = json.loads(image_embedding_str)

            relevance_score = 0

            # 1. Keyword Matching in Analysis Results:
            if any(
                word in analysis_results.lower() for word in query.lower().split()
            ):
                relevance_score += 2

            # 2. Caption Similarity:
            caption_similarity = cosine_similarity(
                [await bilgi_grafiği.metni_göm(query)],
                [await bilgi_grafiği.metni_göm(image_caption)],
            )[0][0]
            relevance_score += caption_similarity

            # 3. Object Detection Relevance:
            if any(
                keyword in obj
                for keyword in query.lower().split()
                for obj in detected_objects
            ):
                relevance_score += 1

            # 4. Color Matching (Implementation needed: Translate color names to numerical values)
            # Example (you'll need a color library or a mapping):
            # color_similarity = calculate_color_similarity(query_colors, dominant_colors)
            # relevance_score += color_similarity

            # 5. Sentiment Matching:
            if visual_sentiment and (
                ("positive" in query.lower() and visual_sentiment == "positive")
                or ("negative" in query.lower() and visual_sentiment == "negative")
            ):
                relevance_score += 0.5

            # 6. Embedding Similarity:
            embedding_similarity = cosine_similarity(
                [await bilgi_grafiği.metni_göm(query)], [image_embedding]
            )[0][0]
            relevance_score += embedding_similarity

            if relevance_score > 0:
                relevant_images.append(
                    (image_hash, analysis_results, relevance_score)
                )

        # Sort by relevance score
        relevant_images.sort(key=lambda x: x[2], reverse=True)

        return relevant_images[:3]  # Return top 3 most relevant images


async def sohbet_geçmişini_kaydet(
    kullanıcı_kimliği: str,
    mesaj: str,
    kullanıcı_adı: str,
    bot_kimliği: str,
    bot_adı: str,
):
    """Saves a chat message to the database."""
    await veritabanı_kuyruğu.put(
        (kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı)
    )


async def veritabanı_kuyruğunu_işle():
    """Processes the queue of database operations."""
    while True:
        while not veritabanı_hazır:
            await asyncio.sleep(1)  # Wait until the database is ready
        (
            kullanıcı_kimliği,
            mesaj,
            kullanıcı_adı,
            bot_kimliği,
            bot_adı,
        ) = await veritabanı_kuyruğu.get()
        try:
            async with veritabanı_kilidi:
                async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
                    await db.execute(
                        "INSERT INTO sohbet_gecmisi (kullanıcı_kimliği, mesaj, zaman_damgası, kullanıcı_adı, bot_kimliği, bot_adı) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            kullanıcı_kimliği,
                            mesaj,
                            datetime.now(timezone.utc).isoformat(),
                            kullanıcı_adı,
                            bot_kimliği,
                            bot_adı,
                        ),
                    )
                    await db.commit()
        except Exception as e:
            logging.error(f"Error occurred while saving to the database: {e}")
        finally:
            veritabanı_kuyruğu.task_done()


async def geri_bildirimi_veritabanına_kaydet(
    kullanıcı_kimliği: str, geri_bildirim: str
):
    """Saves user feedback to the database."""
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            await db.execute(
                "INSERT INTO geri_bildirimler (kullanıcı_kimliği, geri_bildirim, zaman_damgası) VALUES (?, ?, ?)",
                (
                    kullanıcı_kimliği,
                    geri_bildirim,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()


async def ilgili_geçmişi_al(kullanıcı_kimliği: str, geçerli_mesaj: str) -> str:
    """Retrieves relevant conversation history from the database."""
    global veritabanı_kilidi
    if veritabanı_kilidi is None:
        veritabanı_kilidi = asyncio.Lock()

    async with veritabanı_kilidi:
        geçmiş_metni = ""
        mesajlar = []
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            async with db.execute(
                "SELECT mesaj FROM sohbet_gecmisi WHERE kullanıcı_kimliği = ? ORDER BY id DESC LIMIT ?",
                (
                    kullanıcı_kimliği,
                    50,
                ),  # Get the last 50 messages
            ) as cursor:
                async for satır in cursor:
                    mesajlar.append(satır[0])

        mesajlar.reverse()  # Reverse to chronological order
        if not mesajlar:
            return ""  # Return empty string if no history

        tfidf_matrisi = tfidf_vektörleştirici.fit_transform(mesajlar + [geçerli_mesaj])
        geçerli_mesaj_vektörü = tfidf_matrisi[-1]
        benzerlikler = cosine_similarity(
            geçerli_mesaj_vektörü, tfidf_matrisi[:-1]
        ).flatten()
        en_benzer_indeksler = np.argsort(benzerlikler)[-3:]  # Top 3 similar messages

        for indeks in en_benzer_indeksler:
            geçmiş_metni += mesajlar[indeks] + "\n"
        return geçmiş_metni


# --- Database Table Creation ---
async def sohbet_geçmişi_tablosu_oluştur():
    """Creates the chat history and feedback tables in the database."""
    async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS sohbet_gecmisi (
            id INTEGER PRIMARY KEY,
            kullanıcı_kimliği TEXT,
            mesaj TEXT,
            zaman_damgası TEXT,
            kullanıcı_adı TEXT,
            bot_kimliği TEXT,
            bot_adı TEXT
        )
        """
        )
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS geri_bildirimler (
            id INTEGER PRIMARY KEY,
            kullanıcı_kimliği TEXT,
            geri_bildirim TEXT,
            zaman_damgası TEXT
        )
        """
        )
        await db.commit()


# --- Database Initialization ---
async def veritabanını_başlat():
    """Initializes the database and sets the ready flag."""
    global veritabanı_hazır
    async with veritabanı_kilidi:
        await sohbet_geçmişi_tablosu_oluştur()
        await init_image_database()
        veritabanı_hazır = True


# --- User Profile Management ---
def kullanıcı_profillerini_yükle() -> Dict:
    """Loads user profiles from a JSON file."""
    try:
        with open(KULLANICI_PROFİLLERİ_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning("User profile file not found or corrupted. Starting new.")
        return defaultdict(
            lambda: {
                "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
                "demografi": {"yaş": None, "konum": None},
                "geçmiş_özeti": "",
                "bağlam": [],
                "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6},
                "diyalog_durumu": "karşılama",
                "uzun_süreli_hafıza": [],
                "son_bot_eylemi": None,
                "ilgiler": [],
                "sorgu": "",
                "planlama_durumu": {},
                "etkileşim_geçmişi": [],
                "feedback_topics": [],
                "feedback_keywords": [],
                "satisfaction": 0,
            }
        )


def kullanıcı_profillerini_kaydet():
    """Saves user profiles to a JSON file."""
    profiller_kopyası = defaultdict(
        lambda: {
            "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
            "demografi": {"yaş": None, "konum": None},
            "geçmiş_özeti": "",
            "bağlam": [],
            "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6},
            "diyalog_durumu": "karşılama",
            "uzun_süreli_hafıza": [],
            "son_bot_eylemi": None,
            "ilgiler": [],
            "sorgu": "",
            "planlama_durumu": {},
            "etkileşim_geçmişi": [],
            "feedback_topics": [],
            "feedback_keywords": [],
            "satisfaction": 0,
        }
    )

    for kullanıcı_kimliği, profil in kullanıcı_profilleri.items():
        profiller_kopyası[kullanıcı_kimliği].update(profil)
        profiller_kopyası[kullanıcı_kimliği]["bağlam"] = list(
            profil["bağlam"]
        )  # Convert deque to list

        # Convert NumPy arrays in "ilgiler" to lists for JSON serialization
        for ilgi in profiller_kopyası[kullanıcı_kimliği]["ilgiler"]:
            if isinstance(ilgi.get("gömme"), np.ndarray):
                ilgi["gömme"] = ilgi["gömme"].tolist()

    try:
        with open(KULLANICI_PROFİLLERİ_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(profiller_kopyası, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error occurred while saving user profiles: {e}")


# --- Feedback Analysis ---
async def veritabanından_geri_bildirimi_analiz_et():
    """Analyzes user feedback from the database using sentiment analysis and topic modeling."""
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            async with db.execute("SELECT * FROM geri_bildirimler") as cursor:
                async for satır in cursor:
                    kullanıcı_kimliği, geri_bildirim, zaman_damgası = satır

                    # 1. Sentiment Analysis (using Gemini)
                    duygu_istemi = f"""
                    Analyze the sentiment of the following feedback:

                    Feedback: {geri_bildirim}

                    Indicate the sentiment as one of the following: positive, negative, or neutral.
                    """
                    try:
                        duygu_yanıtı = await gemini_ile_yanıt_oluştur(
                            duygu_istemi, kullanıcı_kimliği
                        )
                        duygu_etiketi = duygu_yanıtı.strip().lower()
                        logging.info(
                            f"Sentiment Analysis of Feedback (User {kullanıcı_kimliği}): {duygu_etiketi}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error occurred during sentiment analysis of feedback: {e}"
                        )
                        duygu_etiketi = "neutral"  # Default to neutral if error

                    # 2. Topic Modeling (using LDA)
                    try:
                        işlenmiş_geri_bildirim = geri_bildirimi_ön_işle(
                            geri_bildirim
                        )
                        tfidf = TfidfVectorizer().fit_transform(
                            [işlenmiş_geri_bildirim]
                        )
                        lda = LatentDirichletAllocation(n_components=3, random_state=0)
                        lda.fit(tfidf)
                        dominant_topic = np.argmax(lda.transform(tfidf))
                        logging.info(
                            f"Dominant Topic for Feedback (User {kullanıcı_kimliği}): {dominant_topic}"
                        )
                        top_keywords = get_top_keywords_for_topic(
                            lda, TfidfVectorizer().get_feature_names_out(), 5
                        )
                        logging.info(
                            f"Top Keywords for Topic {dominant_topic}: {top_keywords}"
                        )

                        # Store topic and keywords in user profile
                        if (
                            "feedback_topics"
                            not in kullanıcı_profilleri[kullanıcı_kimliği]
                        ):
                            kullanıcı_profilleri[kullanıcı_kimliği][
                                "feedback_topics"
                            ] = []
                        if (
                            "feedback_keywords"
                            not in kullanıcı_profilleri[kullanıcı_kimliği]
                        ):
                            kullanıcı_profilleri[kullanıcı_kimliği][
                                "feedback_keywords"
                            ] = []
                        kullanıcı_profilleri[kullanıcı_kimliği][
                            "feedback_topics"
                        ].append(dominant_topic)
                        kullanıcı_profilleri[kullanıcı_kimliği][
                            "feedback_keywords"
                        ].extend(top_keywords)
                    except Exception as e:
                        logging.error(f"Error occurred during topic modeling: {e}")

                    # 3. Update User Profiles Based on Feedback (Example)
                    if duygu_etiketi == "positive":
                        kullanıcı_profilleri[kullanıcı_kimliği]["satisfaction"] = (
                            kullanıcı_profilleri[kullanıcı_kimliği].get(
                                "satisfaction", 0
                            )
                            + 1
                        )
                    elif duygu_etiketi == "negative":
                        logging.warning(
                            f"Negative feedback received from User {kullanıcı_kimliği}: {geri_bildirim}"
                        )


# --- Feedback Analysis Helper Functions ---
def geri_bildirimi_ön_işle(geri_bildirim):
    """Preprocesses feedback text. Needs to be implemented based on your requirements."""
    # Your preprocessing logic here (tokenization, stop word removal, etc.)
    return geri_bildirim


def get_top_keywords_for_topic(model, feature_names, num_top_words):
    """Gets top keywords for a topic from the LDA model."""
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_keywords_idx = topic.argsort()[:-num_top_words - 1 : -1]
        topic_keywords.append([feature_names[i] for i in top_keywords_idx])
    return topic_keywords[topic_idx]

# --- Discord Event Handlers ---
@bot.event
@error_tracker  # Apply the error tracker to the on_message event
async def on_message(mesaj: discord.Message):
    global aktif_kullanıcılar, hata_sayacı, yanıt_süresi_histogramı, yanıt_süresi_özeti
    if mesaj.author == bot.user:
        return

    aktif_kullanıcılar += 1
    kullanıcı_kimliği = str(mesaj.author.id)
    içerik = mesaj.content.strip()

    try:
        # 1. Image Generation Intent Detection
        image_generation_intent = await image_generation_intent_detection(
            kullanıcı_kimliği, içerik
        )
        if image_generation_intent:
            await generate_and_send_image(image_generation_intent, mesaj)
            return

        # 2. Image Analysis (If image attachment exists)
        if mesaj.attachments:
            for attachment in mesaj.attachments:
                if attachment.content_type.startswith("image"):
                    await mesaj.channel.send(
                        "Resmi analiz ediyorum... Bu işlem bir dakika kadar sürebilir."
                    )
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as response:
                            if response.status == 200:
                                image_data = await response.read()
                                image = Image.open(io.BytesIO(image_data))
                                image_hash = hashlib.md5(image_data).hexdigest()
                                (
                                    analysis_results,
                                    image_caption,
                                    detected_objects,
                                    dominant_colors,
                                    visual_sentiment,
                                    image_embedding,
                                ) = await analyze_image(image, kullanıcı_kimliği)
                                await save_image_data(
                                    kullanıcı_kimliği,
                                    image_hash,
                                    analysis_results,
                                    image_caption,
                                    detected_objects,
                                    dominant_colors,
                                    visual_sentiment,
                                    image_embedding,
                                )
                                await mesaj.channel.send(analysis_results)
                                return
                            else:
                                logging.error(
                                    f"Failed to download image: HTTP status {response.status}"
                                )
                                await mesaj.channel.send(
                                    "Resmi indirirken bir hata oluştu. Lütfen tekrar deneyin."
                                )
                                return

        # 3. Load/Initialize User Profile
        if kullanıcı_kimliği not in kullanıcı_profilleri:
            kullanıcı_profilleri[kullanıcı_kimliği] = {
                "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
                "demografi": {"yaş": None, "konum": None},
                "geçmiş_özeti": "",
                "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU),
                "kişilik": {
                    "mizah": 0.5,
                    "nezaket": 0.8,
                    "iddialılık": 0.6,
                    "yaratıcılık": 0.5,
                },
                "diyalog_durumu": "karşılama",
                "uzun_süreli_hafıza": [],
                "son_bot_eylemi": None,
                "ilgiler": [],
                "sorgu": "",
                "planlama_durumu": {},
                "etkileşim_geçmişi": [],
                "feedback_topics": [],
                "feedback_keywords": [],
                "satisfaction": 0,
                "duygusal_durum": "nötr",
                "çıkarımlar": [],
            }
        else:
            if "bağlam" not in kullanıcı_profilleri[kullanıcı_kimliği]:
                kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"] = deque(
                    maxlen=BAĞLAM_PENCERESİ_BOYUTU
                )

        # 4. Update User Context
        kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"].append(
            {"rol": "kullanıcı", "içerik": içerik}
        )
        kullanıcı_profilleri[kullanıcı_kimliği]["sorgu"] = içerik

        # 5. Determine User Interests
        await kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği, içerik)

        # 6. Retrieve Relevant Data
        ilgili_geçmiş = await ilgili_geçmişi_al(kullanıcı_kimliği, içerik)
        özetlenmiş_arama = await gemini_arama_ve_özetleme(içerik)

        # 7. Perform Complex Reasoning
        başlangıç_zamanı = time.time()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await çok_gelişmiş_muhakeme_gerçekleştir(
                    içerik,
                    ilgili_geçmiş,
                    özetlenmiş_arama,
                    kullanıcı_kimliği,
                    mesaj,
                    içerik,
                )
                if result:
                    yanıt_metni, duygu = result
                    logging.info(
                        f"Response generated successfully on attempt {attempt + 1}"
                    )
                    break
                else:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Response generation failed (attempt {attempt + 1}). Result was None. Retrying..."
                        )
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    else:
                        yanıt_metni = "Üzgünüm, şu anda bir yanıt oluşturmakta zorlanıyorum. Lütfen mesajınızı yeniden ifade edebilir misiniz veya birazdan tekrar deneyebilir misiniz?"
                        duygu = None
                        logging.error("All response generation attempts failed")
            except Exception as e:
                logging.error(
                    f"Error in response generation (attempt {attempt + 1}): {str(e)}"
                )
                logging.error(f"Error type: {type(e).__name__}")
                logging.error(f"Error traceback: {traceback.format_exc()}")
                if attempt == max_retries - 1:
                    yanıt_metni = "İsteğinizi işlerken bir sorunla karşılaştım. Lütfen tekrar deneyin veya sorun devam ederse destek ekibimizle iletişime geçin."
                    duygu = None

        # 8. Manage Complex Dialogue (If in planning state)
        if kullanıcı_profilleri[kullanıcı_kimliği]["diyalog_durumu"] == "planlama":
            yanıt_metni = await karmaşık_diyalog_yöneticisi(
                kullanıcı_profilleri, kullanıcı_kimliği, mesaj
            )

        # 9. Log Response Time
        yanıt_süresi = time.time() - başlangıç_zamanı
        yanıt_süresi_histogramı.append(yanıt_süresi)
        yanıt_süresi_özeti.append(yanıt_süresi)

        # 10. Update User Context with Bot's Response
        kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"].append(
            {"rol": "asistan", "içerik": yanıt_metni}
        )

        # 11. Retrieve Relevant Images
        relevant_images = await get_relevant_images(kullanıcı_kimliği, içerik)
        relevant_image_info = ""
        if relevant_images:
            relevant_image_info = "İşte ilgili resimlerle ilgili önceki analizlerim:\n"
            for image_hash, analysis_results, relevance_score in relevant_images:
                relevant_image_info += f"- {analysis_results}\n"

        # 12. Send Response (Split if too long)
        maks_mesaj_uzunluğu = 2000
        for i in range(0, len(yanıt_metni), maks_mesaj_uzunluğu):
            await mesaj.channel.send(
                yanıt_metni[i : i + maks_mesaj_uzunluğu] + "\n" + relevant_image_info
            )

        # 13. Save Chat History and User Profiles
        await sohbet_geçmişini_kaydet(
            kullanıcı_kimliği,
            içerik,
            mesaj.author.name,
            bot.user.id,
            bot.user.name,
        )
        kullanıcı_profillerini_kaydet()

    except Exception as e:
        logging.exception(f"Mesaj işlenirken bir hata oluştu: {e}")
        hata_sayacı += 1
        await mesaj.channel.send(
            "Beklenmedik bir hata oluştu. Ekibimiz bilgilendirildi ve sorunu çözmek için çalışıyoruz. Lütfen daha sonra tekrar deneyin."
        )
    finally:
        aktif_kullanıcılar -= 1


# --- Bot Startup Event ---
@bot.event
async def on_ready():
    global veritabanı_kilidi
    logging.info(f"{bot.user} olarak giriş yapıldı!")
    veritabanı_kilidi = asyncio.Lock()
    bot.loop.create_task(veritabanını_başlat())
    bot.loop.create_task(veritabanı_kuyruğunu_işle())
    # geri_bildirim_analiz_görevi.start()  # Start feedback analysis task if needed

# --- Run the Bot ---
bot.run(discord_token)
