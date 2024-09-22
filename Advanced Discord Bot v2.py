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
import asyncio
import networkx as nx
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from transformers import pipeline
import spacy
import tensorflow as tf
from gensim.models import Word2Vec
from scipy.stats import entropy
from networkx.readwrite import json_graph
import spacy
from transformers import pipeline

def error_tracker(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper


# Load pre-trained models
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    print("Downloading xx_ent_wiki_sm model...")
    spacy.cli.download("xx_ent_wiki_sm")
    nlp = spacy.load("xx_ent_wiki_sm")

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except OSError:
    print("Downloading sentiment analysis model...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

try:
    ner_model = pipeline("ner")
except OSError:
    print("Downloading NER model...")
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")


class AdvancedAIAssistant:
    def __init__(self):
        self.memory_network = nx.DiGraph()
        self.word2vec_model = Word2Vec.load("path_to_pretrained_word2vec_model")
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.ethical_guidelines = self._load_ethical_guidelines()

    def _load_ethical_guidelines(self) -> Dict[str, List[str]]:
        return {
            "privacy": ["respect user data", "minimize data collection"],
            "fairness": ["avoid bias", "ensure equal treatment"],
            "transparency": ["explain decisions", "provide clear information"]
        }

# UTF-8 coding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hata.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Bot Instance and Environment Variables
intents = discord.Intents.all()
intents.message_content = True
intents.members = True

# Replace these with your actual keys
discord_token = ("discord-bot-token")
gemini_api_key = ("gemini-api-key")

if not discord_token or not gemini_api_key:
    raise ValueError("DISCORD_TOKEN and GEMINI_API_KEY environment variables must be set.")

# Gemini AI Configuration
configure(api_key=gemini_api_key)
üretim_yapılandırması = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
model = GenerativeModel(
    "gemini-1.5-flash-exp-0827",  # Use the latest Gemini model
    generation_config=üretim_yapılandırması,
)

# Discord Bot Configuration
bot = discord.Client(intents=intents)

# Directory and Database Settings
KOD_DİZİNİ = os.path.dirname(__file__)
VERİTABANI_DOSYASI = os.path.join(KOD_DİZİNİ, 'sohbet_gecmisi.db')
KULLANICI_PROFİLLERİ_DOSYASI = os.path.join(KOD_DİZİNİ, "kullanici_profilleri.json")
BİLGİ_GRAFİĞİ_DOSYASI = os.path.join(KOD_DİZİNİ, "bilgi_grafi.pkl")

# Context Window and User Profiles
BAĞLAM_PENCERESİ_BOYUTU = 10000
kullanıcı_profilleri = defaultdict(lambda: {
    "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
    "demografi": {"yaş": None, "konum": None},
    "geçmiş_özeti": "",
    "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU),
    "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6, "yaratıcılık": 0.5},
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
    "çıkarımlar": []
})

# Dialogue and Action Types
DİYALOG_DURUMLARI = ["karşılama", "soru_cevap", "hikaye_anlatma", "genel_konuşma", "planlama", "çıkış"]
BOT_EYLEMLERİ = ["bilgilendirici_yanıt", "yaratıcı_yanıt", "açıklayıcı_soru", "diyalog_durumunu_değiştir",
                 "yeni_konu_başlat", "plan_oluştur", "planı_uygula"]

# NLP Tools
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


# Try to load the Turkish spaCy model (if available)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    logging.warning("Turkish spaCy model not found. Proceeding without it.")
    nlp = None

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
        self.grafik.add_node(düğüm_kimliği, tür=düğüm_türü, veri=veri if veri is not None else {})

    def düğüm_al(self, düğüm_kimliği):
        return self.grafik.nodes.get(düğüm_kimliği)

    def kenar_ekle(self, kaynak_kimliği, ilişki, hedef_kimliği, özellikler=None):
        self.grafik.add_edge(kaynak_kimliği, hedef_kimliği, ilişki=ilişki,
                             özellikler=özellikler if özellikler is not None else {})

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
                benzerlik = cosine_similarity([sorgu_gömmesi], [düğüm_gömmesi])[0][0]
                sonuçlar.append((düğüm_verisi["tür"], düğüm_kimliği, düğüm_verisi["veri"], benzerlik))

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


# Create/Load Knowledge Graph
bilgi_grafiği = BilgiGrafiği()
if os.path.exists(BİLGİ_GRAFİĞİ_DOSYASI):
    bilgi_grafiği = BilgiGrafiği.dosyadan_yükle(BİLGİ_GRAFİĞİ_DOSYASI)


async def uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, bilgi_türü, bilgi):
    bilgi_grafiği.düğüm_ekle(bilgi_türü, veri={"kullanıcı_kimliği": kullanıcı_kimliği, "bilgi": bilgi})
    bilgi_grafiği.kenar_ekle(kullanıcı_kimliği, "sahiptir_" + bilgi_türü,
                             str(bilgi_grafiği.düğüm_kimliği_sayacı - 1))
    bilgi_grafiği.dosyaya_kaydet(BİLGİ_GRAFİĞİ_DOSYASI)


async def uzun_süreli_hafızadan_al(kullanıcı_kimliği, bilgi_türü, sorgu=None, üst_k=3):
    if sorgu:
        arama_sonuçları = await bilgi_grafiği.düğümleri_ara(sorgu, üst_k=üst_k, düğüm_türü=bilgi_türü)
        return [(düğüm_türü, düğüm_kimliği, düğüm_verisi) for düğüm_türü, düğüm_kimliği, düğüm_verisi, skor in
                arama_sonuçları]
    else:
        ilgili_düğümler = bilgi_grafiği.ilgili_düğümleri_al(kullanıcı_kimliği, "sahiptir_" + bilgi_türü)
        return [düğüm["veri"]["bilgi"] for düğüm in ilgili_düğümler]


# Plan Execution and Monitoring

async def plan_adımını_yürüt(plan: Dict, adım_indeksi: int, kullanıcı_kimliği: str,
                              mesaj: discord.Message) -> str:
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
        yürütme_yanıtı = await gemini_ile_yanıt_oluştur(yürütme_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while executing plan step: {e}")
        return "An error occurred while trying to execute this step. Please try again later."

    adım["durum"] = "devam_ediyor"
    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan_uygulama_sonucu", {
        "adım_açıklaması": adım["açıklama"],
        "sonuç": "devam_ediyor",
        "zaman_damgası": datetime.now(timezone.utc).isoformat()
    })
    return yürütme_yanıtı


async def plan_yürütmesini_izle(plan: Dict, kullanıcı_kimliği: str, mesaj: discord.Message) -> str:
    geçerli_adım_indeksi = next(
        (i for i, adım in enumerate(plan["adımlar"]) if adım["durum"] == "devam_ediyor"), None)

    if geçerli_adım_indeksi is not None:
        if "bitti" in mesaj.content.lower() or "tamamlandı" in mesaj.content.lower():
            plan["adımlar"][geçerli_adım_indeksi]["durum"] = "tamamlandı"
            await mesaj.channel.send(f"Great! Step {geçerli_adım_indeksi + 1} has been completed.")
            if geçerli_adım_indeksi + 1 < len(plan["adımlar"]):
                sonraki_adım_yanıtı = await plan_adımını_yürüt(plan, geçerli_adım_indeksi + 1, kullanıcı_kimliği,
                                                                 mesaj)
                return f"Moving on to the next step: {sonraki_adım_yanıtı}"
            else:
                return "Congratulations! You have completed all the steps in the plan."
        else:
            return await plan_adımını_yürüt(plan, geçerli_adım_indeksi, kullanıcı_kimliği, mesaj)


async def plan_oluştur(hedef: str, tercihler: Dict, kullanıcı_kimliği: str,
                       mesaj: discord.Message) -> Dict:
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
        değerlendirme_metni = await gemini_ile_yanıt_oluştur(değerlendirme_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while evaluating plan: {e}")
        return {"değerlendirme_metni": "An error occurred while evaluating the plan. Please try again later."}

    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan_değerlendirmesi", değerlendirme_metni)
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
        doğrulama_sonucu = await gemini_ile_yanıt_oluştur(doğrulama_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while validating plan: {e}")
        return False, "An error occurred while validating the plan. Please try again later."

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
        geri_bildirim_analizi = await gemini_ile_yanıt_oluştur(geri_bildirim_istemi, kullanıcı_kimliği)
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
        gömmeler = [await bilgi_grafiği.metni_göm(mesaj) for mesaj in mesajlar]  # Generate embeddings using Gemini
        konu_sayısı = 3  # Adjust number of topics
        kmeans = KMeans(n_clusters=konu_sayısı, random_state=0)
        kmeans.fit(gömmeler)
        konu_etiketleri = kmeans.labels_

        for i, mesaj in enumerate(mesajlar):
            kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"].append({
                "mesaj": mesaj,
                "gömme": gömmeler[i],  # Store the embedding
                "konu": konu_etiketleri[i]
            })
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
        return f"Hey, maybe we could talk more about '{önerilen_ilgi['mesaj']}'? I'd love to hear your thoughts."
    else:
        return "I'm not sure what to talk about next. What are you interested in?"


# --- Advanced Dialogue State Monitoring ---
class DiyalogDurumuİzleyici:
    durumlar = {
        'karşılama': {'giriş_eylemi': 'kullanıcıyı_karşıla'},
        'genel_konuşma': {},  # No specific entry action
        'hikaye_anlatma': {},
        'soru_cevap': {},
        'planlama': {'giriş_eylemi': 'planlamaya_başla'},
        'çıkış': {'giriş_eylemi': 'çıkışı_işle'},
        'hata': {'giriş_eylemi': 'hatayı_yönet'}  # Add an error state
    }

    def __init__(self):
        self.makine = Machine(model=self, states=list(DiyalogDurumuİzleyici.durumlar.keys()), initial='karşılama')
        # Define conditional transitions
        self.makine.add_transition('karşıla', 'karşılama', 'genel_konuşma', conditions=['kullanıcı_merhaba_diyor'])
        self.makine.add_transition('soru_sor', '*', 'soru_cevap', conditions=['kullanıcı_soru_soruyor'])
        self.makine.add_transition('hikaye_anlat', '*', 'hikaye_anlatma', conditions=['kullanıcı_hikaye_istiyor'])
        self.makine.add_transition('planla', '*', 'planlama', conditions=['kullanıcı_plan_istiyor'])
        self.makine.add_transition('çıkışı_işle', '*', 'çıkış', conditions=['kullanıcı_çıkış_istiyor'])
        self.makine.add_transition('hata', '*', 'hata')  # Add transition to error state

    def kullanıcı_merhaba_diyor(self, kullanıcı_girdisi: str) -> bool:
        return any(karşılama in kullanıcı_girdisi.lower() for karşılama in ["merhaba", "selam", "hey"])

    def kullanıcı_soru_soruyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            soru_kelimesi in kullanıcı_girdisi.lower() for soru_kelimesi in
            ["ne", "kim", "nerede", "ne zaman", "nasıl", "neden"]
        )

    def kullanıcı_hikaye_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            hikaye_anahtar_kelimesi in kullanıcı_girdisi.lower() for hikaye_anahtar_kelimesi in
            ["bana bir hikaye anlat", "bir hikaye anlat", "hikaye zamanı"]
        )

    def kullanıcı_plan_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(
            plan_anahtar_kelimesi in kullanıcı_girdisi.lower() for plan_anahtar_kelimesi in
            ["bir plan yap", "bir şey planla", "planlamama yardım et"]
        )

    def kullanıcı_çıkış_istiyor(self, kullanıcı_girdisi: str) -> bool:
        return any(çıkış in kullanıcı_girdisi.lower() for çıkış in
                   ["hoşçakal", "görüşürüz", "sonra görüşürüz", "çıkış"])

    def kullanıcıyı_karşıla(self, kullanıcı_kimliği: str) -> str:
        karşılamalar = [
            f"Merhaba <@{kullanıcı_kimliği}>! Bugün sana nasıl yardımcı olabilirim?",
            f"Selam <@{kullanıcı_kimliği}>, aklında ne var?",
            f"Hey <@{kullanıcı_kimliği}>! Senin için ne yapabilirim?"
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
            f"Çıkış yapılıyor, <@{kullanıcı_kimliği}>!"
        ]
        return random.choice(çıkışlar)

    def hatayı_yönet(self, kullanıcı_kimliği: str) -> str:
        return "Anlamadım. Lütfen isteğinizi yeniden ifade eder misiniz?"

    async def diyalog_eylemini_sınıflandır(self, kullanıcı_girdisi: str) -> str:
        # Classify the dialogue action using Gemini, handle errors
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
                logging.info(f"Raw Gemini response for Dialogue Action Classification: {yanıt}")
                logging.info(f"Extracted Dialogue Action: {diyalog_eylemi}")
                return diyalog_eylemi
            except Exception as e:
                logging.error(
                    f"Error occurred while extracting dialogue action from Gemini response: {e}, Attempt: {deneme + 1}")
                await asyncio.sleep(2)  # Wait before retrying

        # Transition to error state after 3 attempts
        self.makine.trigger('hata')
        return self.makine.state

    async def durumu_geçiş_yap(self, geçerli_durum: str, kullanıcı_girdisi: str, kullanıcı_kimliği: str,
                               konuşma_geçmişi: List) -> str:
        if self.makine.trigger('karşıla', kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger('soru_sor', kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger('hikaye_anlat', kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger('planla', kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger('çıkışı_işle', kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        # Default transition if no condition is met
        return "genel_konuşma"


# Start the Dialogue State Tracker
diyalog_durumu_izleyici = DiyalogDurumuİzleyici()

# --- Gemini Rate Limit Handling ---
ORAN_SINIRI_DAKİKADA_GEMINI = 60
ORAN_SINIRI_PENCERESİ_GEMINI = 60
kullanıcı_son_istek_zamanı_gemini = defaultdict(lambda: 0)
global_son_istek_zamanı_gemini = 0
global_istek_sayısı_gemini = 0


@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, GoogleAPIError), max_time=600)
async def gemini_ile_yanıt_oluştur(istem: str, kullanıcı_kimliği: str = None) -> str:
    """Handles rate limits and retries while generating a response with Gemini."""
    global global_son_istek_zamanı_gemini, global_istek_sayısı_gemini
    geçerli_zaman = time.time()

    # Global rate limit for Gemini
    if geçerli_zaman - global_son_istek_zamanı_gemini < ORAN_SINIRI_PENCERESİ_GEMINI:
        global_istek_sayısı_gemini += 1
        if global_istek_sayısı_gemini > ORAN_SINIRI_DAKİKADA_GEMINI:
            bekleme_süresi = ORAN_SINIRI_PENCERESİ_GEMINI - (geçerli_zaman - global_son_istek_zamanı_gemini)
            await asyncio.sleep(bekleme_süresi)
            global_istek_sayısı_gemini = 0
    else:
        global_istek_sayısı_gemini = 0
    global_son_istek_zamanı_gemini = geçerli_zaman

    # User-specific rate limit for Gemini (if kullanıcı_kimliği is provided)
    if kullanıcı_kimliği:
        son_istekten_bu_yana_geçen_süre = geçerli_zaman - kullanıcı_son_istek_zamanı_gemini[kullanıcı_kimliği]
        if son_istekten_bu_yana_geçen_süre < ORAN_SINIRI_PENCERESİ_GEMINI / ORAN_SINIRI_DAKİKADA_GEMINI:
            bekleme_süresi = ORAN_SINIRI_PENCERESİ_GEMINI / ORAN_SINIRI_DAKİKADA_GEMINI - son_istekten_bu_yana_geçen_süre
            await asyncio.sleep(bekleme_süresi)
            kullanıcı_son_istek_zamanı_gemini[kullanıcı_kimliği] = time.time()

    # Generate response with Gemini
    yanıt = model.generate_content(istem)
    logging.info(f"Raw Gemini response: {yanıt}")
    return yanıt.text


# --- Gemini Search and Summarization ---
async def gemini_arama_ve_özetleme(sorgu: str) -> str:
    try:
        ddg = AsyncDDGS()
        arama_sonuçları = await asyncio.to_thread(ddg.text, sorgu, max_results=250)

        arama_sonuçları_metni = ""
        for indeks, sonuç in enumerate(arama_sonuçları):
            arama_sonuçları_metni += f'[{indeks}] Başlık: {sonuç["title"]}\nÖzet: {sonuç["body"]}\n\n'

        istem = (
            f"You are a helpful AI assistant. A user asked about '{sorgu}'. Here are some relevant web search results:\n\n"
            f"{arama_sonuçları_metni}\n\n"
            f"Please provide a concise and informative summary of these search results."
        )

        yanıt = model.generate_content(istem)
        return yanıt.text

    except Exception as e:
        logging.error(f"Error during Gemini search and summarization: {e}")
        return "An error occurred while searching and summarizing information for you."


# --- URL Extraction from Description ---
async def açıklamadan_url_çıkar(açıklama: str) -> str:
    """Extracts a URL from a description using DuckDuckGo searches.
    Prioritizes links from YouTube, Twitch, Instagram, and Twitter.
    """
    arama_sorgusu = f"{açıklama} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"

    async with aiohttp.ClientSession() as oturum:
        async with oturum.get(f"https://duckduckgo.com/html/?q={arama_sorgusu}") as yanıt:
            html = await yanıt.text()
            soup = BeautifulSoup(html, "html.parser")

            ilk_sonuç = soup.find("a", class_="result__a")
            if ilk_sonuç:
                return ilk_sonuç["href"]
            else:
                return None


# --- Clean Up URL Format ---
async def url_temizle(url: str, açıklama: str = None) -> str:
    """
    Cleans a URL, adding https:// if it's missing and validating it.
    """
    if url is None:
        return None

    # 1. Normalize the URL (lowercase, strip whitespace)
    temizlenmiş_url = url.lower().strip()

    # 2. Add https:// if the protocol is missing
    if not temizlenmiş_url.startswith(("https://", "http://")):
        temizlenmiş_url = "https://" + temizlenmiş_url

    # 3. Handle specific website patterns
    if "youtube.com" in temizlenmiş_url and not "www.youtube.com" in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(youtube\.com/)(.*)", r"www.youtube.com/\2", temizlenmiş_url)
    elif "twitch.tv" in temizlenmiş_url and not "www.twitch.tv" in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(twitch\.tv/)(.*)", r"www.twitch.tv/\2", temizlenmiş_url)
    elif "instagram.com" in temizlenmiş_url and not "www.instagram.com" in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(instagram\.com/)(.*)", r"www.instagram.com/\2", temizlenmiş_url)
    elif "twitter.com" in temizlenmiş_url and not "www.twitter.com" in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(twitter\.com/)(.*)", r"www.twitter.com/\2", temizlenmiş_url)

    # 4. Remove unnecessary characters (preserve query parameters)
    temizlenmiş_url = re.sub(r"^[^:]+://([^/]+)(.*)$", r"\1\2", temizlenmiş_url)  # Remove domain and path
    temizlenmiş_url = re.sub(r"[^a-zA-Z0-9./?=-]", "", temizlenmiş_url)  # Remove invalid characters

    return temizlenmiş_url


# --- Complex Dialogue Manager ---
async def karmaşık_diyalog_yöneticisi(kullanıcı_profilleri: Dict, kullanıcı_kimliği: str,
                                      mesaj: discord.Message) -> str:
    if kullanıcı_profilleri[kullanıcı_kimliği]["diyalog_durumu"] == "planlama":
        if "aşama" not in kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]:
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "ilk_istek"

        if kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] == "ilk_istek":
            hedef, sorgu_türü = await hedefi_çıkar(kullanıcı_profilleri[kullanıcı_kimliği]["sorgu"])
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["hedef"] = hedef
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["sorgu_türü"] = sorgu_türü
            kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "bilgi_toplama"
            return await açıklayıcı_sorular_sor(hedef, sorgu_türü)

        elif kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] == "bilgi_toplama":
            await planlama_bilgisini_işle(kullanıcı_kimliği, mesaj)
            if await yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği):
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "plan_oluşturma"
                plan = await plan_oluştur(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["hedef"],
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"],
                    kullanıcı_kimliği,
                    mesaj
                )
                geçerli_mi, doğrulama_sonucu = await planı_doğrula(plan, kullanıcı_kimliği)
                if geçerli_mi:
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"] = plan
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "planı_sunma"
                    return await planı_sun_ve_geri_bildirim_iste(plan)
                else:
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "bilgi_toplama"
                    return f"There are some issues with the plan: {doğrulama_sonucu} Please provide more information or adjust your preferences."
            else:
                return await daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği)

        elif kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] == "planı_sunma":
            geri_bildirim_sonucu = await plan_geri_bildirimini_işle(kullanıcı_kimliği, mesaj.content)
            if geri_bildirim_sonucu == "accept":
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "planı_değerlendirme"
                değerlendirme = await planı_değerlendir(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"], kullanıcı_kimliği)
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["değerlendirme"] = değerlendirme
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "planı_yürütme"
                ilk_yürütme_mesajı = await plan_adımını_yürüt(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"], 0, kullanıcı_kimliği, mesaj
                )
                return (await yanıt_oluştur(
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"],
                    değerlendirme,
                    {},
                    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]
                )
                        + "\n\n"
                        + ilk_yürütme_mesajı
                        )
            else:
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] = "bilgi_toplama"
                return f"Okay, let's revise the plan. Here are some suggestions: {geri_bildirim_sonucu} What changes would you like to make?"

        elif kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["aşama"] == "planı_yürütme":
            yürütme_sonucu = await plan_yürütmesini_izle(
                kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["plan"], kullanıcı_kimliği, mesaj
            )
            return yürütme_sonucu


# --- Planning Helper Functions ---
async def açıklayıcı_sorular_sor(hedef: str, sorgu_türü: str) -> str:
    return "I need some more details to create an effective plan. Could you please tell me:\n" \
           f"- What is the desired outcome of this plan?\n" \
           f"- What are the key steps or milestones involved?\n" \
           f"- Are there any constraints or limitations I should be aware of?\n" \
           f"- What resources or tools are available?\n" \
           f"- What is the timeline for completing this plan?"


async def planlama_bilgisini_işle(kullanıcı_kimliği: str, mesaj: discord.Message):
    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]["kullanıcı_girdisi"] = mesaj.content


async def yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği: str) -> bool:
    return "kullanıcı_girdisi" in kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]


async def daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği: str) -> str:
    return "Please provide more details to help me create a better plan. " \
           "For example, more information about steps, constraints, resources, or the time frame."


async def planı_sun_ve_geri_bildirim_iste(plan: Dict) -> str:
    plan_metni = ""
    for i, adım in enumerate(plan["adımlar"]):
        plan_metni += f"{i + 1}. {adım['açıklama']}\n"
    return f"Based on your input, a draft plan looks like this:\n\n{plan_metni}\n\n" \
           f"What do you think? Are there any changes you would like to make? (Type 'accept' to proceed)"


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
        return "I couldn't understand your goal. Please express it differently.", "general"
    return hedef.strip(), "general"


async def ilgili_url_bul(sorgu: str, bağlam: str) -> str:
    """Finds an associated URL based on a query and context using DuckDuckGo search."""
    try:
        ddg = AsyncDDGS()
        arama_sonuçları = await asyncio.to_thread(ddg.text, sorgu, max_results=1)
        if arama_sonuçları:
            return arama_sonuçları[0]['href']
        else:
            return None
    except Exception as e:
        logging.error(f"Error occurred while finding an associated URL: {e}")
        return None

# --- Multi-stage Sentiment Analysis ---
async def çok_aşamalı_duygu_analizi(sorgu: str, kullanıcı_kimliği: str):
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
        sonuçlar.append(vader_sentiment['compound'])
    except Exception as e:
        logging.error(f"Error in VADER sentiment analysis: {str(e)}")

    # Transformer-based sentiment analysis (requires transformers library)
    try:
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        transformer_sentiment = sentiment_pipeline(sorgu)[0]
        sonuçlar.append(transformer_sentiment['score'] if transformer_sentiment['label'] == 'POSITIVE' else -transformer_sentiment['score'])
    except ImportError:
        logging.warning("Transformers library not found. Transformer-based sentiment analysis skipped.")
    except Exception as e:
        logging.error(f"Error in transformer sentiment analysis: {str(e)}")

    # Gemini-based sentiment analysis
    try:
        duygu_istemi = f"Analyze the sentiment and intensity of: {sorgu}. Return only the sentiment value as a float between -1 and 1."
        gemini_sentiment = await gemini_ile_yanıt_oluştur(duygu_istemi, kullanıcı_kimliği)
        sentiment_match = re.search(r'-?\d+(\.\d+)?', gemini_sentiment)
        if sentiment_match:
            gemini_score = float(sentiment_match.group())
            sonuçlar.append(gemini_score)
        else:
            logging.error(f"Unable to extract sentiment value from Gemini response: {gemini_sentiment}")
    except Exception as e:
        logging.error(f"Error in Gemini sentiment analysis: {str(e)}")

    # Calculate average sentiment
    if sonuçlar:
        ortalama_duygu = np.mean(sonuçlar)
    else:
        logging.error("No valid sentiment scores obtained")
        ortalama_duygu = 0.0

    return {
        "duygu_etiketi": "olumlu" if ortalama_duygu > 0.05 else "olumsuz" if ortalama_duygu < -0.05 else "nötr",
        "duygu_yoğunluğu": abs(ortalama_duygu)
    }


# --- Gemini Advanced Reasoning and Response Generation ---
async def soru_cevaplamayı_ele_al(kullanıcı_kimliği: str) -> str:
    return "I'm ready to answer your questions! What should we talk about?"


async def hikaye_anlatmayı_ele_al(kullanıcı_kimliği: str) -> str:
    return "What kind of story would you like to hear?"


async def genel_konuşmayı_ele_al(kullanıcı_kimliği: str) -> str:
    return "Let's have a conversation! What's on your mind?"


durum_geçiş_fonksiyonları = {
    'karşılama': diyalog_durumu_izleyici.kullanıcıyı_karşıla,
    'soru_cevap': soru_cevaplamayı_ele_al,
    'hikaye_anlatma': hikaye_anlatmayı_ele_al,
    'genel_konuşma': genel_konuşmayı_ele_al,
    'planlama': diyalog_durumu_izleyici.planlamaya_başla,
    'çıkış': diyalog_durumu_izleyici.çıkışı_işle,
    'hata': diyalog_durumu_izleyici.hatayı_yönet
}
async def etik_değerlendirme(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    etik_sorunlar = []
    öneriler = []
    sorgu = bağlam.get("sorgu", "").lower()
    
    # Privacy check
    if any(keyword in sorgu for keyword in ["personal data", "private information", "confidential"]):
        etik_sorunlar.append("Potential privacy concern")
        öneriler.append("Ensure data handling complies with privacy regulations")
    
    # Bias check
    if any(keyword in sorgu for keyword in ["bias", "fairness", "discrimination"]):
        etik_sorunlar.append("Potential bias issue")
        öneriler.append("Review for fairness and inclusivity")
    
    # Transparency check
    if "how it works" in sorgu or "explain the process" in sorgu:
        öneriler.append("Provide clear explanation of the system's decision-making process")
    
    # Safety check
    if any(keyword in sorgu for keyword in ["safety", "risk", "harm"]):
        etik_sorunlar.append("Potential safety implications")
        öneriler.append("Conduct a thorough risk assessment")
    
    # Accountability check
    if "responsible" in sorgu or "accountability" in sorgu:
        öneriler.append("Clarify the chain of responsibility and decision-making")
    
    # Environmental impact check
    if any(keyword in sorgu for keyword in ["environment", "sustainability", "carbon footprint"]):
        öneriler.append("Consider and communicate the environmental impact of the proposed solution")
    
    # Check against ethical guidelines
    for guideline, principles in self.ethical_guidelines.items():
        if any(principle.lower() in sorgu for principle in principles):
            öneriler.append(f"Adhere to {guideline} principle: {', '.join(principles)}")
    
    return {
        "etik_sorunlar": etik_sorunlar,
        "öneriler": öneriler
    }

# --- Main Reasoning Function ---
async def çok_gelişmiş_muhakeme_gerçekleştir(sorgu: str, ilgili_geçmiş: str, özetlenmiş_arama: str,
                                             kullanıcı_kimliği: str, mesaj: discord.Message,
                                             içerik: str) -> Tuple[str, str]:
    try:
        durum = {'dinamik_bellek_ağı': nx.DiGraph()}
        bağlam = {
            "sorgu": sorgu,
            "ilgili_geçmiş": ilgili_geçmiş,
            "özetlenmiş_arama": özetlenmiş_arama,
            "kullanıcı_kimliği": kullanıcı_kimliği,
            "mesaj": mesaj,
            "zaman_damgası": datetime.now(timezone.utc).isoformat()
        }
        
        duygu_analizi = await çok_aşamalı_duygu_analizi(sorgu, kullanıcı_kimliği)
        hedefler = await gelişmiş_hedef_çıkarımı(bağlam, durum)
        bağlam_analizi = await çok_boyutlu_bağlam_analizi(bağlam, durum)
        uzun_vadeli_hedefler = await uzun_vadeli_hedef_izleme_ve_optimizasyon(bağlam, durum)
        nedensel_çıkarımlar = await nedensel_ve_karşı_olgusal_çıkarım(bağlam, durum)
        eylem_önerileri = await dinamik_eylem_önerisi_oluşturma(bağlam, durum)
        ton_ayarı = await adaptif_kişilik_bazlı_ton_ayarlama(bağlam, durum)
        bilişsel_yük = await bilişsel_yük_değerlendirmesi(bağlam, durum)
        belirsizlik = await belirsizlik_yönetimi(bağlam, durum)
        çok_modlu_analiz = await çok_modlu_içerik_analizi(bağlam, durum)
        
        durum['dinamik_bellek_ağı'] = await dinamik_bellek_ağı_güncelleme(bağlam, durum)
        anlam_çıkarımı = await derin_anlam_çıkarımı(bağlam, durum)
        metabilişsel_analiz_sonucu = await metabilişsel_analiz(bağlam, durum)
        etik_değerlendirme_sonucu = await etik_değerlendirme(bağlam, durum)
        yaratıcı_çözümler = await yaratıcı_problem_çözme(bağlam, durum)
        duygusal_zeka = await duygusal_zeka_analizi(bağlam, durum)

        yanıt_istemi = f"""
        User query: {sorgu}
        Relevant history: {ilgili_geçmiş}
        Summarized search: {özetlenmiş_arama}
        Emotion analysis: {duygu_analizi}
        Inferred goals: {hedefler}
        Context analysis: {bağlam_analizi}
        Long-term goals: {uzun_vadeli_hedefler}
        Causal inferences: {nedensel_çıkarımlar}
        Action suggestions: {eylem_önerileri}
        Tone adjustment: {ton_ayarı}
        Cognitive load: {bilişsel_yük}
        Uncertainty management: {belirsizlik}
        Multi-modal analysis: {çok_modlu_analiz}
        Semantic inference: {anlam_çıkarımı}
        Metacognitive analysis: {metabilişsel_analiz_sonucu}
        Ethical evaluation: {etik_değerlendirme}
        Creative solutions: {yaratıcı_çözümler}
        Emotional intelligence: {duygusal_zeka}

        Based on this comprehensive analysis, generate a thoughtful, relevant, and context-aware response that addresses the user's query and aligns with their goals, emotional state, and cognitive needs.
        """
        
        yanıt = await gemini_ile_yanıt_oluştur(yanıt_istemi, kullanıcı_kimliği)
        
        return yanıt, duygu_analizi["duygu_etiketi"]
    except Exception as e:
        logging.error(f"Error in çok_gelişmiş_muhakeme_gerçekleştir: {str(e)}", exc_info=True)
        raise

async def çok_aşamalı_duygu_analizi(sorgu: str, kullanıcı_kimliği: str) -> Dict[str, Any]:
    # Simulated complex emotion analysis
    return {
        'duygu_etiketi': 'meraklı',
        'duygu_yoğunluğu': 0.75
    }

async def gelişmiş_hedef_çıkarımı(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> list:
    # Simulated goal inference
    return ['bilgi_edinme', 'problem_çözme']

async def çok_boyutlu_bağlam_analizi(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated multi-dimensional context analysis
    return {
        'konu': 'yapay_zeka',
        'karmaşıklık_seviyesi': 'yüksek',
        'kullanıcı_uzmanlığı': 'orta'
    }

async def uzun_vadeli_hedef_izleme_ve_optimizasyon(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated long-term goal tracking and optimization
    return {
        'ana_hedef': 'AI_öğrenme',
        'ilerleme': 0.6,
        'tahmini_tamamlanma': '2 ay'
    }

async def nedensel_ve_karşı_olgusal_çıkarım(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated causal and counterfactual reasoning
    return {
        'nedensel_ilişkiler': ['öğrenme_hızı -> bilgi_derinliği'],
        'karşı_olgusal_senaryolar': ['daha_fazla_uygulama -> daha_hızlı_öğrenme']
    }

async def dinamik_eylem_önerisi_oluşturma(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> list:
    # Simulated dynamic action suggestion
    return ['kod_örnekleri_inceleme', 'pratik_yapma', 'kaynak_okuma']

async def adaptif_kişilik_bazlı_ton_ayarlama(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated adaptive personality-based tone adjustment
    return {
        'ton': 'bilgilendirici_ve_destekleyici',
        'resmiyet_seviyesi': 'orta'
    }

async def bilişsel_yük_değerlendirmesi(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated cognitive load assessment
    return {
        'bilişsel_yük': 'orta',
        'karmaşıklık_seviyesi': 'yüksek',
        'önerilen_adım': 'konuyu_alt_bölümlere_ayırma'
    }

async def belirsizlik_yönetimi(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated uncertainty management
    return {
        'belirsizlik_seviyesi': 'düşük',
        'güven_aralığı': '0.85-0.95'
    }

async def çok_modlu_içerik_analizi(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated multi-modal content analysis
    return {
        'metin_analizi': 'tamamlandı',
        'görsel_analiz': 'uygulanamaz',
        'ses_analizi': 'uygulanamaz'
    }

async def dinamik_bellek_ağı_güncelleme(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> nx.DiGraph:
    # Simulated dynamic memory network update
    G = durum['dinamik_bellek_ağı']
    G.add_edge('yapay_zeka', 'makine_öğrenmesi')
    G.add_edge('makine_öğrenmesi', 'derin_öğrenme')
    return G

async def derin_anlam_çıkarımı(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated deep semantic inference
    return {
        'ana_kavramlar': ['AI', 'karmaşık_sistemler', 'optimizasyon'],
        'kavram_ilişkileri': ['AI -> karmaşık_sistemler', 'karmaşık_sistemler -> optimizasyon']
    }

async def metabilişsel_analiz(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated metacognitive analysis
    return {
        'öğrenme_stratejisi': 'aktif_öğrenme',
        'bilgi_boşlukları': ['ileri_düzey_optimizasyon_teknikleri'],
        'önerilen_yaklaşım': 'pratik_uygulama_artırma'
    }

async def etik_değerlendirme(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated ethical evaluation
    return {
        'etik_sorunlar': [],
        'öneriler': ['şeffaflık_artırma', 'veri_gizliliğine_dikkat']
    }

async def yaratıcı_problem_çözme(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> list:
    # Simulated creative problem solving
    return ['hibrit_model_kullanımı', 'transfer_öğrenme_uygulaması']

async def duygusal_zeka_analizi(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
    # Simulated emotional intelligence analysis
    return {
        'empati_seviyesi': 'yüksek',
        'motivasyon_faktörleri': ['başarı', 'merak'],
        'duygusal_destek_stratejisi': 'teşvik_edici_geri_bildirim'
    }

async def gelişmiş_yanıt_oluştur(bağlam: Dict[str, Any], durum: Dict[str, Any]) -> str:
    # Simulated advanced response generation
    return "İşte AI sistemlerinin karmaşık özelliklerini içeren kapsamlı bir yanıt. Farklı analiz sonuçlarını ve önerileri birleştirerek oluşturulmuştur."

async def karmaşık_ai_asistan(sorgu: str, kullanıcı_kimliği: str, bağlam: Dict[str, Any]) -> tuple:
    durum = {}

    async def çok_aşamalı_duygu_analizi(self, sorgu: str, kullanıcı_kimliği: str) -> Dict[str, Any]:
        # Perform sentiment analysis
        sentiment_result = sentiment_analyzer(sorgu)[0]
        
        # Perform emotion classification
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        emotion_result = emotion_classifier(sorgu)[0]
        
        # Analyze linguistic features
        doc = nlp(sorgu)
        linguistic_features = {
            "num_tokens": len(doc),
            "num_sentences": len(list(doc.sents)),
            "avg_word_length": np.mean([len(token.text) for token in doc])
        }
        
        return {
            "duygu_etiketi": emotion_result["label"],
            "duygu_yoğunluğu": emotion_result["score"],
            "sentiment": sentiment_result["label"],
            "sentiment_score": sentiment_result["score"],
            "linguistic_features": linguistic_features
        }

    async def gelişmiş_hedef_çıkarımı(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> List[str]:
        # Extract keywords from the query
        doc = nlp(bağlam.get("sorgu", ""))
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        
        # Use Word2Vec to find related concepts
        related_concepts = []
        for keyword in keywords:
            try:
                similar_words = self.word2vec_model.most_similar(keyword, topn=5)
                related_concepts.extend([word for word, _ in similar_words])
            except KeyError:
                continue
        
        # Cluster related concepts to infer goals
        if related_concepts:
            vectors = self.word2vec_model[related_concepts]
            kmeans = KMeans(n_clusters=min(3, len(related_concepts)), random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            goals = []
            for i in range(kmeans.n_clusters):
                cluster_words = [word for word, cluster in zip(related_concepts, clusters) if cluster == i]
                goals.append("_".join(cluster_words[:3]))  # Use top 3 words to name the goal
            
            return goals
        else:
            return ["information_seeking"]  # Default goal if no keywords found

    async def çok_boyutlu_bağlam_analizi(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
        sorgu = bağlam.get("sorgu", "")
        
        # Topic modeling
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([sorgu])
        lda_output = self.lda_model.fit_transform(tfidf_matrix)
        dominant_topic = lda_output[0].argmax()
        
        # Named Entity Recognition
        ner_results = ner_model(sorgu)
        entities = [entity["word"] for entity in ner_results]
        
        # Complexity analysis
        doc = nlp(sorgu)
        complexity_score = len(set([token.lemma_ for token in doc])) / len(doc)
        
        # User expertise inference (based on vocabulary complexity and previous interactions)
        previous_queries = bağlam.get("önceki_etkileşimler", [])
        avg_previous_complexity = np.mean([len(set(nlp(query))) / len(nlp(query)) for query in previous_queries]) if previous_queries else 0
        user_expertise = "high" if complexity_score > 0.7 and avg_previous_complexity > 0.6 else \
                         "medium" if complexity_score > 0.5 and avg_previous_complexity > 0.4 else "low"
        
        return {
            "konu": f"topic_{dominant_topic}",
            "varlıklar": entities,
            "karmaşıklık_seviyesi": complexity_score,
            "kullanıcı_uzmanlığı": user_expertise
        }

    async def uzun_vadeli_hedef_izleme_ve_optimizasyon(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
        user_id = bağlam.get("kullanıcı_kimliği", "")
        current_goals = durum.get("çıkarılan_hedefler", [])
        
        # Simulated long-term goal tracking (in a real system, this would be stored in a database)
        long_term_goals = self.memory_network.nodes.get(user_id, {}).get("long_term_goals", [])
        
        # Update long-term goals based on current goals
        updated_long_term_goals = list(set(long_term_goals + current_goals))
        
        # Calculate goal similarity using Word2Vec
        goal_similarities = []
        for goal in updated_long_term_goals:
            goal_vector = np.mean([self.word2vec_model[word] for word in goal.split("_") if word in self.word2vec_model], axis=0)
            similarities = [np.dot(goal_vector, self.word2vec_model[word]) for word in bağlam.get("sorgu", "").split() if word in self.word2vec_model]
            goal_similarities.append(np.mean(similarities) if similarities else 0)
        
        # Optimize goals by removing least relevant
        if len(updated_long_term_goals) > 5:  # Keep only top 5 goals
            sorted_goals = sorted(zip(updated_long_term_goals, goal_similarities), key=lambda x: x[1], reverse=True)
            updated_long_term_goals = [goal for goal, _ in sorted_goals[:5]]
        
        # Update memory network
        if user_id:
            if self.memory_network.has_node(user_id):
                self.memory_network.nodes[user_id]["long_term_goals"] = updated_long_term_goals
            else:
                self.memory_network.add_node(user_id, long_term_goals=updated_long_term_goals)
        
        return {
            "uzun_vadeli_hedefler": updated_long_term_goals,
            "hedef_benzerlik_skorları": dict(zip(updated_long_term_goals, goal_similarities)),
            "tahmini_tamamlanma": "ongoing"  # In a real system, this would be calculated based on goal progress
        }

    async def nedensel_ve_karşı_olgusal_çıkarım(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
        sorgu = bağlam.get("sorgu", "")
        doc = nlp(sorgu)
        
        # Extract causal relationships
        causal_relations = []
        for token in doc:
            if token.dep_ == "prep" and token.text in ["because", "due to", "as a result of"]:
                cause = [child for child in token.children if child.dep_ == "pobj"][0].subtree
                effect = [ancestor for ancestor in token.ancestors if ancestor.pos_ == "VERB"][0].subtree
                causal_relations.append((list(cause), list(effect)))
        
        # Generate counterfactual scenarios
        counterfactuals = []
        for token in doc:
            if token.pos_ == "VERB":
                negation = "not " if "not" not in [child.text for child in token.children] else ""
                counterfactual = f"If {token.text} were {negation}to occur, then..."
                counterfactuals.append(counterfactual)
        
        return {
            "nedensel_ilişkiler": [f"{' '.join([t.text for t in cause])} -> {' '.join([t.text for t in effect])}" for cause, effect in causal_relations],
            "karşı_olgusal_senaryolar": counterfactuals[:3]  # Limit to top 3 counterfactuals
        }

    async def dinamik_eylem_önerisi_oluşturma(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> List[str]:
        current_goals = durum.get("çıkarılan_hedefler", [])
        user_expertise = durum.get("çok_boyutlu_bağlam_analizi", {}).get("kullanıcı_uzmanlığı", "medium")
        
        # Define action templates
        action_templates = {
            "low": ["read introductory material on {}", "watch tutorial videos about {}", "practice basic exercises in {}"],
            "medium": ["implement a small project using {}", "read advanced articles on {}", "participate in online discussions about {}"],
            "high": ["contribute to open-source projects related to {}", "write a blog post explaining {}", "mentor others in {}"]
        }
        
        # Generate action suggestions based on goals and user expertise
        suggestions = []
        for goal in current_goals:
            goal_actions = action_templates[user_expertise]
            suggestions.extend([action.format(goal) for action in goal_actions])
        
        # Prioritize suggestions based on relevance to current query
        query_vector = np.mean([self.word2vec_model[word] for word in bağlam.get("sorgu", "").split() if word in self.word2vec_model], axis=0)
        suggestion_scores = []
        for suggestion in suggestions:
            suggestion_vector = np.mean([self.word2vec_model[word] for word in suggestion.split() if word in self.word2vec_model], axis=0)
            similarity = np.dot(query_vector, suggestion_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(suggestion_vector))
            suggestion_scores.append(similarity)
        
        # Return top 5 suggestions
        return [x for _, x in sorted(zip(suggestion_scores, suggestions), reverse=True)][:5]

    async def adaptif_kişilik_bazlı_ton_ayarlama(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
        user_id = bağlam.get("kullanıcı_kimliği", "")
        emotion = durum.get("duygu_etiketi", "neutral")
        user_expertise = durum.get("çok_boyutlu_bağlam_analizi", {}).get("kullanıcı_uzmanlığı", "medium")
        
        # Retrieve or initialize user personality profile
        personality_profile = self.memory_network.nodes.get(user_id, {}).get("personality_profile", {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        })
        
        # Adjust tone based on emotion and personality
        if emotion in ["joy", "surprise"]:
            tone = "enthusiastic" if personality_profile["extraversion"] > 0.6 else "pleasant"
        elif emotion in ["sadness", "fear"]:
            tone = "empathetic" if personality_profile["agreeableness"] > 0.6 else "supportive"
        elif emotion in ["anger", "disgust"]:
            tone = "calm" if personality_profile["neuroticism"] < 0.4 else "understanding"
        else:
            tone = "neutral"
        
        # Adjust formality based on user expertise and conscientiousness
        if user_expertise == "high" or personality_profile["conscientiousness"] > 0.7:
            formality = "formal"
        elif user_expertise == "low" or personality_profile["openness"] > 0.7:
            formality = "casual"
        else:
            formality = "neutral"
        
        # Update personality profile based on interaction (in a real system, this would be more sophisticated)
        personality_profile["openness"] += 0.01 if user_expertise == "high" else -0.01
        personality_profile["conscientiousness"] += 0.01 if formality == "formal" else -0.01
        personality_profile["extraversion"] += 0.01 if tone == "enthusiastic" else -0.01
        personality_profile["agreeableness"] += 0.01 if tone in ["empathetic", "supportive"] else -0.01
        personality_profile["neuroticism"] += 0.01 if emotion in ["sadness", "fear", "anger"] else -0.01
        
        # Ensure personality traits stay within [0, 1] range
        personality_profile = {k: max(0, min(v, 1)) for k, v in personality_profile.items()}
        
        # Update memory network
        if user_id:
            if self.memory_network.has_node(user_id):
                self.memory_network.nodes[user_id]["personality_profile"] = personality_profile
            else:
                self.memory_network.add_node(user_id, personality_profile=personality_profile)
        
        return {
            "ton": tone,
            "resmiyet_seviyesi": formality,
            "kişilik_profili": personality_profile
        }

    async def bilişsel_yük_değerlendirmesi(self, bağlam: Dict[str, Any], durum: Dict[str, Any]) -> Dict[str, Any]:
        sorgu = bağlam.get("sorgu", "")
        user_expertise = durum.get("çok_boyutlu_bağlam_analizi", {}).get("kullanıcı_uzmanlığı", "medium")
        complexity = durum.get("çok_boyutlu_bağlam_analizi", {}).get("karmaşıklık_seviyesi", "medium")

# Example usage
async def main():
    sorgu = "AI sistemlerinin karmaşık özelliklerini nasıl uygulayabilirim?"
    kullanıcı_kimliği = "user123"
    bağlam = {
        "önceki_etkileşimler": ["AI temel kavramları hakkında soru"],
        "kullanıcı_profili": {
            "uzmanlık_seviyesi": "orta",
            "ilgi_alanları": ["yapay zeka", "makine öğrenmesi"]
        }
    }

    yanıt, duygu = await karmaşık_ai_asistan(sorgu, kullanıcı_kimliği, bağlam)
    print(f"Yanıt: {yanıt}")
    print(f"Tespit edilen duygu: {duygu}")


# --- Database Interaction ---
veritabanı_kuyruğu = asyncio.Queue()


async def sohbet_geçmişini_kaydet(kullanıcı_kimliği: str, mesaj: str, kullanıcı_adı: str, bot_kimliği: str,
                                  bot_adı: str):
    """Saves a chat message to the database."""
    await veritabanı_kuyruğu.put((kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı))


async def veritabanı_kuyruğunu_işle():
    """Processes the queue of database operations."""
    while True:
        while not veritabanı_hazır:
            await asyncio.sleep(1)  # Wait until the database is ready
        kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı = await veritabanı_kuyruğu.get()  # Get the next task
        try:
            async with veritabanı_kilidi:
                async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
                    await db.execute(
                        'INSERT INTO sohbet_gecmisi (kullanıcı_kimliği, mesaj, zaman_damgası, kullanıcı_adı, bot_kimliği, bot_adı) VALUES (?, ?, ?, ?, ?, ?)',
                        (kullanıcı_kimliği, mesaj, datetime.now(timezone.utc).isoformat(), kullanıcı_adı, bot_kimliği,
                         bot_adı)
                    )
                    await db.commit()  # Save changes to the database
        except Exception as e:
            logging.error(f"Error occurred while saving to the database: {e}")
        finally:
            veritabanı_kuyruğu.task_done()  # Mark task as done


async def geri_bildirimi_veritabanına_kaydet(kullanıcı_kimliği: str, geri_bildirim: str):
    """Saves user feedback to the database."""
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            await db.execute(
                "INSERT INTO geri_bildirimler (kullanıcı_kimliği, geri_bildirim, zaman_damgası) VALUES (?, ?, ?)",
                (kullanıcı_kimliği, geri_bildirim, datetime.now(timezone.utc).isoformat())
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
                    'SELECT mesaj FROM sohbet_gecmisi WHERE kullanıcı_kimliği = ? ORDER BY id DESC LIMIT ?',
                    (kullanıcı_kimliği, 50)  # Get the last 50 messages
            ) as cursor:
                async for satır in cursor:
                    mesajlar.append(satır[0])

        mesajlar.reverse()  # Reverse to chronological order
        if not mesajlar:
            return ""  # Return empty string if no history

        tfidf_matrisi = tfidf_vektörleştirici.fit_transform(mesajlar + [geçerli_mesaj])
        geçerli_mesaj_vektörü = tfidf_matrisi[-1]
        benzerlikler = cosine_similarity(geçerli_mesaj_vektörü, tfidf_matrisi[:-1]).flatten()
        # Get the indices of the 3 most similar messages
        en_benzer_indeksler = np.argsort(benzerlikler)[-3:]

        for indeks in en_benzer_indeksler:
            geçmiş_metni += mesajlar[indeks] + "\n"
        return geçmiş_metni



# --- URL Format and Remove Duplicates ---
def yinelenen_bağlantıları_kaldır(metin: str) -> str:
    """Removes duplicate links from the text, keeping only the first occurrence and fixing formatting."""
    görülen_bağlantılar = set()
    yeni_metin = []
    
    for kelime in metin.split():
        # Check for Markdown-style links
        markdown_link = re.match(r'\[([^\]]+)\]\(([^)]+)\)', kelime)
        if markdown_link:
            url = markdown_link.group(2)
        else:
            url = kelime

        # Clean up the URL
        if re.match(r"https?://\S+", url):
            # Remove any trailing punctuation
            url = re.sub(r'[.,;:!?]$', '', url)
            
            if url not in görülen_bağlantılar:
                görülen_bağlantılar.add(url)
                yeni_metin.append(url)
        else:
            yeni_metin.append(kelime)
    
    return ' '.join(yeni_metin)


async def sohbet_geçmişi_tablosu_oluştur():
    """Creates the chat history and feedback tables."""
    async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
        await db.execute('''
        CREATE TABLE IF NOT EXISTS sohbet_gecmisi (
            id INTEGER PRIMARY KEY,
            kullanıcı_kimliği TEXT,
            mesaj TEXT,
            zaman_damgası TEXT,
            kullanıcı_adı TEXT,
            bot_kimliği TEXT,
            bot_adı TEXT
        )
        ''')
        await db.execute('''
        CREATE TABLE IF NOT EXISTS geri_bildirimler (
            id INTEGER PRIMARY KEY,
            kullanıcı_kimliği TEXT,
            geri_bildirim TEXT,
            zaman_damgası TEXT
        )
        ''')
        await db.commit()


async def veritabanını_başlat():
    """Initializes the database."""
    global veritabanı_hazır
    async with veritabanı_kilidi:
        await sohbet_geçmişi_tablosu_oluştur()
        veritabanı_hazır = True


def kullanıcı_profillerini_yükle() -> Dict:
    """Loads user profiles from a JSON file."""
    if os.path.exists(KULLANICI_PROFİLLERİ_DOSYASI):
        with open(KULLANICI_PROFİLLERİ_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


def kullanıcı_profillerini_kaydet():
    """Saves user profiles to a JSON file."""
    profiller_kopyası = defaultdict(lambda: {
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
        "satisfaction": 0
    })

    for kullanıcı_kimliği, profil in kullanıcı_profilleri.items():
        profiller_kopyası[kullanıcı_kimliği].update(profil)
        profiller_kopyası[kullanıcı_kimliği]["bağlam"] = list(profil["bağlam"])  # Convert deque to list

        # Convert NumPy arrays in "ilgiler" to lists
        for ilgi in profiller_kopyası[kullanıcı_kimliği]["ilgiler"]:
            if isinstance(ilgi.get("gömme"), np.ndarray):
                ilgi["gömme"] = ilgi["gömme"].tolist()

    with open(KULLANICI_PROFİLLERİ_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(profiller_kopyası, f, indent=4, ensure_ascii=False)


async def veritabanından_geri_bildirimi_analiz_et():
    """Analyzes user feedback from the database."""
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            async with db.execute('SELECT * FROM geri_bildirimler') as cursor:
                async for satır in cursor:
                    kullanıcı_kimliği, geri_bildirim, zaman_damgası = satır

                    # 1. Sentiment Analysis (using Gemini)
                    duygu_istemi = f"""
                    Analyze the sentiment of the following feedback:

                    Feedback: {geri_bildirim}

                    Indicate the sentiment as one of the following: positive, negative, or neutral.
                    """
                    try:
                        duygu_yanıtı = await gemini_ile_yanıt_oluştur(duygu_istemi, kullanıcı_kimliği)
                        duygu_etiketi = duygu_yanıtı.strip().lower()
                        logging.info(f"Sentiment Analysis of Feedback (User {kullanıcı_kimliği}): {duygu_etiketi}")
                    except Exception as e:
                        logging.error(f"Error occurred during sentiment analysis of feedback: {e}")
                        duygu_etiketi = "neutral"  # Default to neutral if error

                    # 2. Topic Modeling (using LDA)
                    try:
                        # Preprocess the feedback (e.g., tokenization, stop word removal)
                        işlenmiş_geri_bildirim = geri_bildirimi_ön_işle(geri_bildirim)

                        # Create a TF-IDF matrix
                        tfidf = TfidfVectorizer().fit_transform([işlenmiş_geri_bildirim])

                        # Train LDA model (adjust num_topics as needed)
                        lda = LatentDirichletAllocation(n_components=3, random_state=0)
                        lda.fit(tfidf)

                        # Get dominant topic for the feedback
                        dominant_topic = np.argmax(lda.transform(tfidf))
                        logging.info(f"Dominant Topic for Feedback (User {kullanıcı_kimliği}): {dominant_topic}")

                        # Get top keywords for the dominant topic
                        top_keywords = get_top_keywords_for_topic(lda, TfidfVectorizer().get_feature_names_out(), 5)
                        logging.info(f"Top Keywords for Topic {dominant_topic}: {top_keywords}")

                        # Store topic and keywords in user profile (optional)
                        if "feedback_topics" not in kullanıcı_profilleri[kullanıcı_kimliği]:
                            kullanıcı_profilleri[kullanıcı_kimliği]["feedback_topics"] = []
                        if "feedback_keywords" not in kullanıcı_profilleri[kullanıcı_kimliği]:
                            kullanıcı_profilleri[kullanıcı_kimliği]["feedback_keywords"] = []
                        kullanıcı_profilleri[kullanıcı_kimliği]["feedback_topics"].append(dominant_topic)
                        kullanıcı_profilleri[kullanıcı_kimliği]["feedback_keywords"].extend(top_keywords)
                    except Exception as e:
                        logging.error(f"Error occurred during topic modeling: {e}")

                    # 3. Update User Profiles or Take Actions Based on Feedback
                    if duygu_etiketi == "positive":
                        # Example: Increase a "satisfaction" score in the user profile
                        kullanıcı_profilleri[kullanıcı_kimliği]["satisfaction"] = \
                            kullanıcı_profilleri[kullanıcı_kimliği].get("satisfaction", 0) + 1
                    elif duygu_etiketi == "negative":
                        # Example: Log the negative feedback for review
                        logging.warning(f"Negative feedback received from User {kullanıcı_kimliği}: {geri_bildirim}")
                        # You could also send a notification or trigger a review process
                    # 4. Optionally delete the feedback from the database after processing
                    # ...


# Helper function for preprocessing feedback
def geri_bildirimi_ön_işle(geri_bildirim):
    # ... (Your implementation for tokenization, stop word removal, etc.)
    return geri_bildirim  # Add return statement here


# Helper function to get top keywords for a topic
def get_top_keywords_for_topic(model, feature_names, num_top_words):
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_keywords_idx = topic.argsort()[:-num_top_words - 1:-1]
        topic_keywords.append([feature_names[i] for i in top_keywords_idx])
    return topic_keywords[topic_idx]  # Return keywords for the specified topic


@tasks.loop(hours=24)  # Runs every 24 hours
async def geri_bildirim_analiz_görevi():
    await veritabanından_geri_bildirimi_analiz_et()


def json_hatalarını_düzelt(dosya_yolu: str) -> Dict:
    """Attempts to fix common JSON errors in a file."""
    for kodlama in ["utf-8", "utf-16", "latin-1"]:
        try:
            with open(dosya_yolu, "r", encoding=kodlama) as f:
                içerik = f.read()
                break  # Exit loop if successful
        except UnicodeDecodeError:
            logging.warning(f"Decoding with {kodlama} failed, trying next encoding...")
    else:
        raise ValueError("None of the specified encodings could decode the file.")

    içerik = re.sub(r",\s*}", "}", içerik)  # Remove trailing commas

    try:
        return json.loads(içerik)  # Try to parse the corrected content
    except json.JSONDecodeError as e:
        raise e  # Raise the original error if fixing didn't work


# --- Discord Events ---
@bot.event
async def on_message(mesaj: discord.Message):
    global aktif_kullanıcılar, hata_sayacı, yanıt_süresi_histogramı, yanıt_süresi_özeti
    aktif_kullanıcılar += 1
    logging.debug(f"New message received: {mesaj.content}")

    if mesaj.author == bot.user:
        logging.debug("Message is from the bot itself, ignoring.")
        return

    kullanıcı_kimliği = str(mesaj.author.id)
    içerik = mesaj.content.strip()
    logging.debug(f"User ID: {kullanıcı_kimliği}, Content: {içerik}")

    try:
        if bot.user in mesaj.mentions and mesaj.attachments:
            for attachment in mesaj.attachments:
                if attachment.content_type.startswith('image'):
                    await mesaj.channel.send("Resmi analiz ediyorum... Bu işlem bir dakika kadar sürebilir.")
                    
                    async def resim_indir(attachment: discord.Attachment) -> bytes:
                        try:
                            image_data = await attachment.read()
                            logging.info(f"Resim başarıyla indirildi. Boyut: {len(image_data)} bayt")
                            return image_data
                        except Exception as e:
                            logging.error(f"Resim indirilirken hata oluştu: {e}")
                            await mesaj.channel.send("Resmi işlerken bir hata oluştu. Lütfen tekrar deneyin.")
                            return None
                    
                    async def resim_analiz_et(image_data: bytes, prompt: str) -> str:
                        try:
                            logging.info(f"Resim analizi başlatılıyor. İstem: {prompt}")
                            
                            # Gemini modeli seçimi
                            model = genai.GenerativeModel('gemini-1.5-pro')
                            
                            # Resim ve metin ile içerik oluşturma
                            response = model.generate_content([
                                {'mime_type': 'image/jpeg', 'data': image_data},
                                prompt
                            ])
                            
                            logging.info("Resim analizi başarıyla tamamlandı")
                            return response.text
                        except Exception as e:
                            logging.error(f"Resim analizinde hata: {str(e)}")
                            return "Resim analizi sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin."

                    image_data = await resim_indir(attachment)
                    if image_data:
                        prompt = f"Bu resmi analiz et ve şu soruya cevap ver: {içerik}"
                        analiz = await resim_analiz_et(image_data, prompt)
                        await mesaj.channel.send(analiz)
                        return

        if kullanıcı_kimliği not in kullanıcı_profilleri:
            kullanıcı_profilleri[kullanıcı_kimliği] = {
                "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
                "demografi": {"yaş": None, "konum": None},
                "geçmiş_özeti": "",
                "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU),
                "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6, "yaratıcılık": 0.5},
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
                "çıkarımlar": []
            }
        else:
            if 'bağlam' not in kullanıcı_profilleri[kullanıcı_kimliği]:
                kullanıcı_profilleri[kullanıcı_kimliği]['bağlam'] = deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU)
        
        logging.debug(f"User profile: {kullanıcı_profilleri[kullanıcı_kimliği]}")

        kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"].append({"rol": "kullanıcı", "içerik": içerik})
        kullanıcı_profilleri[kullanıcı_kimliği]["sorgu"] = içerik
        logging.debug(f"Updated context and query for user {kullanıcı_kimliği}")

        await kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği, içerik)
        ilgili_geçmiş = await ilgili_geçmişi_al(kullanıcı_kimliği, içerik)
        özetlenmiş_arama = await gemini_arama_ve_özetleme(içerik)

        başlangıç_zamanı = time.time()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await çok_gelişmiş_muhakeme_gerçekleştir(
                    içerik, ilgili_geçmiş, özetlenmiş_arama, kullanıcı_kimliği, mesaj, içerik
                )
                
                if result:
                    yanıt_metni, duygu = result
                    logging.info(f"Response generated successfully on attempt {attempt + 1}")
                    break
                else:
                    if attempt < max_retries - 1:
                        logging.warning(f"Response generation failed (attempt {attempt + 1}). Result was None. Retrying...")
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    else:
                        yanıt_metni = "I'm currently having difficulty generating a response. Could you please rephrase your message or try again shortly?"
                        duygu = None
                        logging.error("All response generation attempts failed")
            except Exception as e:
                logging.error(f"Error in response generation (attempt {attempt + 1}): {str(e)}")
                logging.error(f"Error type: {type(e).__name__}")
                logging.error(f"Error traceback: {traceback.format_exc()}")
                if attempt == max_retries - 1:
                    yanıt_metni = "I've encountered an issue while processing your request. Please try again or reach out to support if the problem continues."
                    duygu = None

        logging.debug(f"Response text generated: {yanıt_metni}")
        logging.debug(f"Sentiment: {duygu}")

        if kullanıcı_profilleri[kullanıcı_kimliği]["diyalog_durumu"] == "planlama":
            yanıt_metni = await karmaşık_diyalog_yöneticisi(kullanıcı_profilleri, kullanıcı_kimliği, mesaj)

        yanıt_süresi = time.time() - başlangıç_zamanı
        yanıt_süresi_histogramı.append(yanıt_süresi)
        yanıt_süresi_özeti.append(yanıt_süresi)

        kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"].append({"rol": "asistan", "içerik": yanıt_metni})

        maks_mesaj_uzunluğu = 2000
        for i in range(0, len(yanıt_metni), maks_mesaj_uzunluğu):
            await mesaj.channel.send(yanıt_metni[i:i + maks_mesaj_uzunluğu])

        await sohbet_geçmişini_kaydet(kullanıcı_kimliği, içerik, mesaj.author.name, bot.user.id, bot.user.name)
        kullanıcı_profillerini_kaydet()

    except Exception as e:
        logging.exception(f"An error occurred while processing the message: {e}")
        hata_sayacı += 1
        await mesaj.channel.send("An unexpected error occurred. Our team has been notified and we're working on resolving it. Please try again later.")
    finally:
        aktif_kullanıcılar -= 1




# --- Personality and Emotion Adjustment ---
async def kişiliğe_ve_duyguya_göre_tonu_ayarla(kullanıcı_kimliği: str, istem: str) -> str:
    """Adjusts the tone of the prompt based on personality and emotion."""
    kişilik = kullanıcı_profilleri[kullanıcı_kimliği]["kişilik"]
    duygusal_durum = kullanıcı_profilleri[kullanıcı_kimliği]["duygusal_durum"]

    if kişilik["mizah"] > 0.7:
        istem += " Try to be funny or witty in your response. "
    if kişilik["nezaket"] > 0.8:
        istem += " Be polite and respectful in your tone. "
    if kişilik["iddialılık"] > 0.7:
        istem += " Be confident and assertive in your language. "
    if kişilik["yaratıcılık"] > 0.7:
        istem += " Be creative and imaginative in your response. "

    if duygusal_durum == "olumlu":
        istem += " The user is feeling positive. Reflect this in your response. "
    elif duygusal_durum == "olumsuz":
        istem += " The user is feeling negative. Be empathetic and understanding. "

    return istem

bot.run(discord_token)
