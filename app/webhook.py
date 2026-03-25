import os
import logging
import threading
import time
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
import requests
from app.cliente_openai import OpenAIClient

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# KONFIGURASI
# =====================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_CLOUD_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_HOOK_TOKEN")

# Inisialisasi OpenAI Client
openai_client = OpenAIClient()

# =====================================
# MANAJEMEN HISTORY PER PENGGUNA
# =====================================
user_histories = {}  # Format: {wa_id: [list_of_messages]}

# Set untuk menyimpan ID pesan yang sudah diproses (cegah duplikat)
processed_messages = set()

def get_user_history(wa_id: str, max_length: int = 10):
    """Ambil history percakapan user"""
    if wa_id not in user_histories:
        user_histories[wa_id] = []
    return user_histories[wa_id][-max_length:]

def add_to_history(wa_id: str, role: str, content: str):
    """Tambahkan pesan ke history user"""
    if wa_id not in user_histories:
        user_histories[wa_id] = []
    user_histories[wa_id].append({"role": role, "content": content})
    
    # Batasi panjang history (hapus yang lama)
    if len(user_histories[wa_id]) > 20:
        user_histories[wa_id] = user_histories[wa_id][-20:]

# =====================================
# CLEANER UNTUK HAPUS ID PESAN LAMA
# =====================================
def clean_processed_messages():
    """Bersihkan set processed_messages setiap 1 jam"""
    while True:
        time.sleep(3600)  # 1 jam
        count = len(processed_messages)
        processed_messages.clear()
        logger.info(f"🧹 Cleaned {count} old message IDs from cache")

# Jalankan thread cleaner
cleaner_thread = threading.Thread(target=clean_processed_messages, daemon=True)
cleaner_thread.start()
logger.info("🔄 Message ID cleaner thread started (clears every 1 hour)")

# =====================================
# FUNGSI KIRIM PESAN WHATSAPP
# =====================================
def send_whatsapp_message(to: str, message: str):
    """Send text message via WhatsApp Cloud API"""
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info(f"✅ Message sent to {to}")
        return response.json()
    except Exception as e:
        logger.error(f"❌ Error sending message: {e}")
        return None

# =====================================
# FUNGSI TYPING INDICATOR
# =====================================
def send_typing_indicator(to: str, message_id: str):
    """
    Send typing indicator to WhatsApp user.
    Shows "typing..." status while bot processes the response.
    Lasts up to 25 seconds or until message is sent.
    """
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "body": "",
            "typing_indicator": True
        }
    }
    
    # Alternative approach using message_id
    # Some versions use status endpoint instead
    url_alt = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    data_alt = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }
    
    try:
        # Try both methods (first one is simpler)
        response = requests.post(url, headers=headers, json={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "typing"
        })
        
        if response.status_code == 200:
            logger.info(f"⌨️ Typing indicator sent to {to}")
        else:
            # Try alternative method
            response2 = requests.post(url_alt, headers=headers, json=data_alt)
            if response2.status_code == 200:
                logger.info(f"⌨️ Typing indicator sent (alt method) to {to}")
            else:
                logger.warning(f"⚠️ Typing indicator failed: {response2.status_code}")
                
    except Exception as e:
        logger.error(f"❌ Error sending typing indicator: {e}")

# =====================================
# ENDPOINT HEALTH CHECK
# =====================================
@app.get("/")
async def root():
    return {
        "status": "active",
        "bot": "SIMBAH",
        "service": "Smart Information Bot Assistant for Hospital",
        "hospital": "RS Muhammadiyah Bandung"
    }

# =====================================
# WEBHOOK VERIFICATION
# =====================================
@app.get("/webhook/")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """Verify webhook endpoint for Meta"""
    logger.info(f"Webhook verification: mode={hub_mode}, token={hub_verify_token}")
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")
    else:
        logger.warning(f"❌ Verification failed. Expected: {VERIFY_TOKEN}, Got: {hub_verify_token}")
        raise HTTPException(status_code=403, detail="Verification failed")

# =====================================
# WEBHOOK HANDLER UTAMA
# =====================================
@app.post("/webhook/")
async def handle_webhook(request: Request):
    """Handle incoming WhatsApp messages with typing indicator"""
    try:
        body = await request.json()
        logger.info(f"📨 Webhook received")
        
        entry = body.get("entry", [])
        for ent in entry:
            changes = ent.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                
                # =====================================
                # FILTER: HANYA PROSES MESSAGES, ABIAIKAN STATUSES
                # =====================================
                messages = value.get("messages", [])
                
                if not messages:
                    # Ini adalah status update (sent, delivered, read), abaikan
                    statuses = value.get("statuses", [])
                    if statuses:
                        logger.info(f"📊 Status update received (ignored): {[s.get('status') for s in statuses]}")
                    continue
                
                # =====================================
                # PROSES PESAN DARI USER
                # =====================================
                for message in messages:
                    sender = message.get("from")
                    message_type = message.get("type")
                    message_id = message.get("id")
                    
                    # Cegah pesan duplikat berdasarkan ID
                    if message_id in processed_messages:
                        logger.info(f"⚠️ Duplicate message {message_id} from {sender}, ignored")
                        continue
                    
                    # Tambahkan ke set untuk cegah duplikat
                    processed_messages.add(message_id)
                    logger.info(f"📝 New message ID: {message_id}")
                    
                    # Hanya proses pesan teks
                    if message_type == "text":
                        text_content = message.get("text", {}).get("body", "").strip()
                        
                        if not text_content:
                            logger.info(f"⚠️ Empty message from {sender}, ignored")
                            continue
                        
                        logger.info(f"📨 Message from {sender}: {text_content}")
                        
                        # =====================================
                        # KIRIM TYPING INDICATOR
                        # Menampilkan "sedang mengetik..." di WhatsApp user
                        # =====================================
                        send_typing_indicator(sender, message_id)
                        
                        # Tambahkan pesan user ke history
                        add_to_history(sender, "user", text_content)
                        
                        # Ambil history user (5 pesan terakhir)
                        history = get_user_history(sender, max_length=5)
                        
                        # Dapatkan respons dari AI (hybrid: Ollama/OpenAI)
                        try:
                            ai_response = openai_client.complete(
                                message=text_content,
                                history=history
                            )
                            
                            # Validasi respons tidak kosong
                            if not ai_response or ai_response.strip() == "":
                                ai_response = "Maaf, saya tidak dapat memproses permintaan Anda. Silakan coba lagi."
                            
                            # Tambahkan respons AI ke history
                            add_to_history(sender, "assistant", ai_response)
                            
                            # Kirim balasan ke WhatsApp
                            send_whatsapp_message(sender, ai_response)
                            logger.info(f"✅ Response sent to {sender}")
                            
                        except Exception as e:
                            logger.error(f"❌ AI processing error: {e}")
                            error_msg = "Maaf, terjadi gangguan pada sistem. Silakan coba lagi nanti."
                            send_whatsapp_message(sender, error_msg)
                    
                    else:
                        # Pesan non-teks (gambar, audio, dll)
                        logger.info(f"📎 Non-text message from {sender} (type: {message_type}), sending info response")
                        info_msg = "Maaf, saat ini SIMBAH hanya dapat membaca pesan teks. Silakan kirim pesan dalam bentuk teks."
                        send_whatsapp_message(sender, info_msg)
        
        return JSONResponse({"status": "ok"}, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# =====================================
# ENDPOINT UNTUK TEST (OPSIONAL)
# =====================================
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "bot": "SIMBAH",
        "ollama_available": openai_client.ollama_available,
        "openai_available": openai_client.openai_client is not None,
        "active_sessions": len(user_histories),
        "cached_message_ids": len(processed_messages)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)