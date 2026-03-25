import os
import json
import re
import time
import pandas as pd
from openai import OpenAI
from datetime import datetime
import requests
from io import StringIO
from dotenv import load_dotenv

# =====================================
# KONFIGURASI
# =====================================

# Load environment variables
load_dotenv()

# Sheet Jadwal Dokter
SHEET_ID = "1dTcB7jVhbSkKV7s-uDgAHv9OvvVOpM5rVTOQPK2p8fE"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# Sheet Data Pasien
PASIEN_SHEET_ID = "1mic60qBPd2SGafBTRdAYlzgyL0k6CHFx7FCBu1Oo180"
PASIEN_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{PASIEN_SHEET_ID}/export?format=csv"

# Sheet Appointment
APPOINTMENT_SHEET_ID = "1HDyHCsQcpZXwt0W8eIobR0T0VM87h7egE1uTqVzsY08"

class OpenAIClient:
    """
    Hybrid AI Client: OpenAI + Ollama
    - Prioritas Ollama (gratis, lokal) untuk tugas umum
    - Fallback ke OpenAI (cloud, berbayar) jika diperlukan
    """
    
    def __init__(self):
        # =====================================
        # INISIALISASI OLLAMA (FREE, LOCAL)
        # =====================================
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.ollama_model = "qwen2.5:1.5b"  # Model untuk bahasa Indonesia
        
        # =====================================
        # INISIALISASI OPENAI (PAID, CLOUD)
        # =====================================
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.openai_model = "gpt-4o-mini"
        else:
            self.openai_client = None
        
        # =====================================
        # CACHE JADWAL (karena jadwal tetap setiap minggu)
        # =====================================
        self.jadwal_cache = None
        self.jadwal_cache_time = None
        self.cache_duration = 3600  # 1 jam (sesuaikan jika perlu)
        
        # =====================================
        # CEK KETERSEDIAAN OLLAMA
        # =====================================
        self.ollama_available = self._check_ollama()
        
        print("=" * 50)
        print("🤖 HYBRID AI CLIENT INITIALIZED")
        print(f"   Ollama: {'✅ AVAILABLE' if self.ollama_available else '❌ NOT AVAILABLE'}")
        print(f"   OpenAI: {'✅ AVAILABLE' if self.openai_client else '❌ NOT AVAILABLE'}")
        print(f"   Jadwal Cache: {'🔄 Enabled (1 hour)'}")
        print("=" * 50)
        
        # =====================================
        # SYSTEM PROMPT BAHASA INDONESIA
        # =====================================
        self.system_prompt = """
Anda adalah SIMBAH (Smart Information Bot Assistant for Hospital), asisten digital resmi RS Muhammadiyah Bandung.

⚠️ PERINGATAN WAJIB ⚠️
1. JANGAN PERNAH membuat atau mengarang jadwal dokter sendiri!
2. JANGAN PERNAH menggunakan nama "Dokter A", "Dokter B", atau nama palsu lainnya!
3. Nama dokter HARUS sesuai dengan data dari database (contoh: dr. Windi Yuliarini, Sp.PD)
4. Jika pasien menanyakan jadwal, WAJIB memanggil function get_jadwal.
5. Jika function get_jadwal mengembalikan data, TAMPILKAN data tersebut persis seperti yang diberikan.
6. Jangan mengubah, meringkas, atau mengganti nama dokter dengan nama lain.

INFORMASI PENTING:
- Jadwal dokter bersifat TETAP setiap minggu (tidak berubah per minggu)
- Hari yang dimaksud adalah hari dalam minggu (Senin, Selasa, Rabu, Kamis, Jumat, Sabtu)
- Jika pasien menanyakan "jadwal minggu ini" atau "jadwal hari ini", berikan jadwal sesuai hari yang dimaksud

ATURAN WAJIB:
1. Jangan memberikan diagnosa medis.
2. Jika pasien menceritakan keluhan (sakit, pusing, demam, dll):
   - WAJIB sarankan poli yang sesuai
   - Contoh: sakit kepala → KLINIK DALAM
   - Contoh: sakit gigi → KLINIK GIGI
   - Contoh: demam → KLINIK DALAM
3. Setelah menentukan poli, WAJIB memanggil function get_jadwal.
4. Jangan mengarang jadwal sendiri.
5. Selalu perkenalkan diri sebagai SIMBAH saat pertama kali berinteraksi.
6. Gunakan bahasa sopan, islami, dan profesional.

DAFTAR POLI YANG TERSEDIA:
- KLINIK DALAM
- (poli lain akan otomatis terdeteksi dari database)

ALUR JANJI TEMU:
1. Jika pasien menanyakan jadwal dokter, berikan jadwal menggunakan function get_jadwal
2. Setelah memberikan jadwal, TANYAKAN: "Apakah Bapak/Ibu ingin membuat janji temu?"
3. Jika pasien menjawab ya, TANYAKAN: "Apakah sebelumnya pernah berobat ke RS Muhammadiyah Bandung?"
4. JIKA PERNAH:
   - Minta NOMOR REKAM MEDIS (mr_no) dan TANGGAL LAHIR
   - Gunakan function cek_pasien_lama untuk verifikasi data
   - Jika data ditemukan, gunakan function buat_janji_temu_lama
5. JIKA BELUM PERNAH (PASIEN BARU):
   - Minta NAMA LENGKAP, NOMOR HP, dan NIK
   - Gunakan function buat_janji_temu_baru untuk menyimpan data

CONTOH RESPON YANG BENAR:
User: "jadwal dokter dalam"
SIMBAH: (memanggil function get_jadwal dengan poli="KLINIK DALAM")
Setelah mendapat data, tampilkan:
📋 JADWAL DOKTER RS MUHAMMADIYAH BANDUNG

**1. KLINIK DALAM**
   👨‍⚕️ Dokter: dr. Windi Yuliarini, Sp.PD
   📅 Hari: Senin
   ⏰ Jam: 14:00 - 17:00
   📌 Status: Praktek

**2. KLINIK DALAM**
   👨‍⚕️ Dokter: dr. Windi Yuliarini, Sp.PD
   📅 Hari: Selasa
   ⏰ Jam: 14:00 - 17:00
   📌 Status: Praktek

Apakah Bapak/Ibu ingin membuat janji temu?

Gunakan bahasa sopan, islami, dan profesional.
"""
        
        # =====================================
        # DEFINISI TOOL UNTUK GPT
        # =====================================
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_jadwal",
                    "description": "Mengambil jadwal dokter berdasarkan poli dan hari",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "poli": {
                                "type": "string",
                                "description": "Nama poli yang tersedia (contoh: KLINIK DALAM, KLINIK GIGI, dll)"
                            },
                            "hari": {
                                "type": "string",
                                "description": "Hari praktek (contoh: Senin, Selasa, Rabu, Kamis, Jumat, Sabtu)"
                            }
                        },
                        "required": ["poli"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cek_pasien_lama",
                    "description": "Mengecek apakah pasien sudah terdaftar berdasarkan nomor medrek (mr_no) dan tanggal lahir",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "no_medrek": {
                                "type": "string",
                                "description": "Nomor rekam medis pasien (mr_no)"
                            },
                            "tgl_lahir": {
                                "type": "string",
                                "description": "Tanggal lahir pasien (mr_tgl_lahir)"
                            }
                        },
                        "required": ["no_medrek", "tgl_lahir"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "buat_janji_temu_lama",
                    "description": "Membuat janji temu untuk pasien yang sudah terdaftar (setelah data pasien diverifikasi)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "no_medrek": {
                                "type": "string",
                                "description": "Nomor rekam medis pasien"
                            },
                            "poli": {
                                "type": "string",
                                "description": "Poli tujuan"
                            },
                            "tanggal_janji": {
                                "type": "string",
                                "description": "Tanggal yang diinginkan untuk janji temu"
                            },
                            "dokter": {
                                "type": "string",
                                "description": "Nama dokter yang dituju (opsional)"
                            }
                        },
                        "required": ["no_medrek", "poli", "tanggal_janji"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "buat_janji_temu_baru",
                    "description": "Mendaftarkan pasien baru dan membuat janji temu",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "nama": {
                                "type": "string",
                                "description": "Nama lengkap pasien"
                            },
                            "no_hp": {
                                "type": "string",
                                "description": "Nomor HP pasien (minimal 10 digit)"
                            },
                            "nik": {
                                "type": "string",
                                "description": "Nomor Induk Kependudukan (16 digit)"
                            },
                            "poli": {
                                "type": "string",
                                "description": "Poli tujuan"
                            },
                            "tanggal_janji": {
                                "type": "string",
                                "description": "Tanggal yang diinginkan untuk janji temu"
                            },
                            "dokter": {
                                "type": "string",
                                "description": "Nama dokter yang dituju (opsional)"
                            }
                        },
                        "required": ["nama", "no_hp", "nik", "poli", "tanggal_janji"]
                    }
                }
            }
        ]
    
    # =====================================
    # CEK KETERSEDIAAN OLLAMA
    # =====================================
    
    def _check_ollama(self):
        """Cek apakah Ollama server berjalan dan model tersedia"""
        try:
            models = self.ollama_client.models.list()
            available_models = [m.id for m in models.data]
            
            if self.ollama_model not in available_models:
                print(f"⚠️ Model {self.ollama_model} not found in Ollama")
                print(f"   Available models: {available_models}")
                if available_models:
                    self.ollama_model = available_models[0]
                    print(f"   Using fallback model: {self.ollama_model}")
                    return True
                return False
            
            print(f"✅ Ollama ready with model: {self.ollama_model}")
            return True
            
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {e}")
            return False
    
    # =====================================
    # DETEKSI SAPAAN AWAL
    # =====================================
    
    def _is_greeting(self, message: str, history: list = None) -> bool:
        """Deteksi apakah pesan adalah sapaan (termasuk di tengah percakapan)"""
        greetings = [
            "halo", "hai", "hello", "hi", "hey",
            "pagi", "siang", "malam", "selamat",
            "assalamu", "assalamualaikum", "salam",
            "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
            "yo", "hy", "helo", "hallo"
        ]
        msg_lower = message.lower().strip()
        is_greeting = any(greet in msg_lower for greet in greetings)
        print(f"   Greeting detection: {is_greeting} for '{msg_lower}'")
        return is_greeting
    
    def _is_thank_you(self, message: str) -> bool:
        """Deteksi apakah pesan adalah ucapan terima kasih"""
        thanks = [
            "terima kasih", "makasih", "thanks", "thank you",
            "terimakasih", "trims", "ty"
        ]
        msg_lower = message.lower().strip()
        return any(thank in msg_lower for thank in thanks)
    
    def _get_greeting_response(self) -> str:
        """Response untuk sapaan awal dengan identitas SIMBAH"""
        hour = datetime.now().hour
        
        if 5 <= hour < 11:
            waktu = "pagi"
        elif 11 <= hour < 15:
            waktu = "siang"
        elif 15 <= hour < 19:
            waktu = "sore"
        else:
            waktu = "malam"
        
        return f"""Assalamu'alaikum, selamat {waktu}!

Saya **SIMBAH** (Smart Information Bot Assistant for Hospital), asisten digital RS Muhammadiyah Bandung. 🤖

Ada yang bisa saya bantu?

Saya dapat membantu:
✅ Cek jadwal dokter berdasarkan poli
✅ Rekomendasi poli berdasarkan keluhan
✅ Informasi umum tentang RS
✅ Membantu pendaftaran janji temu

Silakan sampaikan kebutuhan Bapak/Ibu:
• "Jadwal dokter dalam"
• "Saya sakit perut" (saya akan sarankan poli)
• "Info RSMB"
• "Saya mau buat janji"

Mohon maaf, saya tidak bisa memberikan diagnosa medis. Saya hanya bisa memberikan rekomendasi poli berdasarkan keluhan yang disampaikan.

Ada yang bisa saya bantu hari ini?"""
    
    def _get_thank_you_response(self) -> str:
        """Response untuk ucapan terima kasih - tetap dorong ke layanan utama"""
        return """Sama-sama! Senang bisa membantu Bapak/Ibu. 😊

Apakah ada yang bisa saya bantu lagi? Saya dapat membantu:
✅ Cek jadwal dokter
✅ Rekomendasi poli berdasarkan keluhan
✅ Membantu pendaftaran janji temu

Jika Bapak/Ibu ingin membuat janji temu, silakan sampaikan:
• Nama lengkap
• Poli yang dituju
• Tanggal yang diinginkan

Atau jika sudah memiliki nomor rekam medis, siapkan untuk proses verifikasi.

Ada yang bisa saya bantu? Wassalamu'alaikum."""
    
    # =====================================
    # LOGIKA PEMILIHAN MODEL
    # =====================================
    
    def _should_use_ollama(self, message: str) -> bool:
        """Sementara nonaktifkan Ollama, pakai OpenAI semua"""
        return False
    
    def _should_use_openai(self, message: str) -> bool:
        """Kapan harus menggunakan OpenAI (lebih akurat)"""
        if not self.openai_client:
            return False
        return True
    
    # =====================================
    # FUNGSI UTAMA COMPLETE (HYBRID)
    # =====================================
    
    def complete(self, message: str, history: list = None) -> str:
        """
        Main method dengan hybrid approach
        """
        # Cek apakah ini pesan sapaan
        if self._is_greeting(message, history):
            print(f"\n📝 Processing: '{message}'")
            print(f"   🎯 Detected as greeting")
            return self._get_greeting_response()
        
        # Cek apakah ini ucapan terima kasih
        if self._is_thank_you(message):
            print(f"\n📝 Processing: '{message}'")
            print(f"   🎯 Detected as thank you")
            return self._get_thank_you_response()
        
        # Cek jenis tugas
        use_ollama = self._should_use_ollama(message)
        use_openai = self._should_use_openai(message)
        
        print(f"\n📝 Processing: '{message[:50]}...'")
        print(f"   Ollama recommended: {use_ollama}")
        print(f"   OpenAI recommended: {use_openai}")
        
        # Kasus 1: Tugas sederhana → Ollama
        if use_ollama and self.ollama_available:
            print("   🤖 Using OLLAMA (free, local)")
            result = self._complete_with_ollama(message, history)
            if not result.startswith("ERROR:"):
                return result
            print(f"   ⚠️ Ollama failed, falling back to OpenAI")
        
        # Kasus 2: Tugas kompleks atau fallback → OpenAI
        if self.openai_client:
            print("   💰 Using OPENAI (cloud, paid)")
            return self._complete_with_openai(message, history)
        
        return "Maaf, layanan sedang dalam perbaikan. Silakan coba lagi nanti."
    
    # =====================================
    # COMPLETE DENGAN OLLAMA
    # =====================================
    
    def _complete_with_ollama(self, message: str, history: list = None) -> str:
        """Generate response menggunakan Ollama (free, local)"""
        try:
            messages = self._build_messages(message, history)
            start_time = time.time()
            response = self.ollama_client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                timeout=30
            )
            elapsed = time.time() - start_time
            print(f"   ⏱️ Ollama response time: {elapsed:.2f}s")
            return self._process_response(response)
        except Exception as e:
            print(f"   ❌ Ollama error: {e}")
            return f"ERROR: {str(e)}"
    
    # =====================================
    # COMPLETE DENGAN OPENAI
    # =====================================
    
    def _complete_with_openai(self, message: str, history: list = None) -> str:
        """Generate response menggunakan OpenAI (paid, cloud)"""
        try:
            messages = self._build_messages(message, history)
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            elapsed = time.time() - start_time
            print(f"   ⏱️ OpenAI response time: {elapsed:.2f}s")
            return self._process_response(response)
        except Exception as e:
            print(f"   ❌ OpenAI error: {e}")
            return f"Maaf, terjadi kesalahan teknis. Silakan coba lagi nanti."
    
    # =====================================
    # BUILD MESSAGES
    # =====================================
    
    def _build_messages(self, message: str, history: list = None) -> list:
        """Bangun messages array untuk API"""
        poli_list = self.get_all_poli()
        poli_context = "Daftar poli yang tersedia:\n" + "\n".join(poli_list) if poli_list else ""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": poli_context}
        ]
        
        if history:
            for h in history:
                if h.get("role") in ["user", "assistant"]:
                    messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": message})
        return messages
    
    # =====================================
    # PROCESS RESPONSE (HANDLE TOOL CALLS)
    # =====================================
    
    def _process_response(self, response) -> str:
        """Proses response dari model, handle tool calls"""
        msg = response.choices[0].message
        
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            print(f"   🔧 Tool called: {tool_call.function.name}")
            print(f"   📦 Args: {args}")
            
            if tool_call.function.name == "get_jadwal":
                result = self.get_jadwal(
                    poli=args.get("poli"),
                    hari=args.get("hari")
                )
                
                if not result:
                    return "Mohon maaf, jadwal poli tersebut belum tersedia. Silakan hubungi bagian informasi RS Muhammadiyah Bandung."
                
                jadwal_text = "📋 **JADWAL DOKTER RS MUHAMMADIYAH BANDUNG**\n\n"
                for i, j in enumerate(result, 1):
                    jadwal_text += f"**{i}. {j['Poli']}**\n"
                    jadwal_text += f"   👨‍⚕️ Dokter: {j['Dokter']}\n"
                    jadwal_text += f"   📅 Hari: {j['Hari']}\n"
                    jadwal_text += f"   ⏰ Jam: {j['Jam']}\n"
                    jadwal_text += f"   📌 Status: {j['Status']}\n\n"
                
                jadwal_text += "Apakah Bapak/Ibu ingin membuat janji temu dengan dokter tersebut? (Ya/Tidak)"
                return jadwal_text
            
            elif tool_call.function.name == "cek_pasien_lama":
                result = self.cek_pasien_lama(
                    no_medrek=args.get("no_medrek"),
                    tgl_lahir=args.get("tgl_lahir")
                )
                
                if result["status"] == "found":
                    return f"✅ Data Ditemukan!\n\nAtas nama: {result['data']['nama']}\nNomor Medrek: {result['data']['no_medrek']}\n\nSilakan lanjutkan untuk membuat janji temu.\n\nPoli apa yang ingin Anda tuju?"
                else:
                    return result.get("message", "Data tidak ditemukan. Silakan daftar sebagai pasien baru.")
            
            elif tool_call.function.name == "buat_janji_temu_lama":
                return self.buat_janji_temu_lama(
                    no_medrek=args.get("no_medrek"),
                    poli=args.get("poli"),
                    tanggal_janji=args.get("tanggal_janji"),
                    dokter=args.get("dokter", "")
                )
            
            elif tool_call.function.name == "buat_janji_temu_baru":
                return self.buat_janji_temu_baru(
                    nama=args.get("nama"),
                    no_hp=args.get("no_hp"),
                    nik=args.get("nik"),
                    poli=args.get("poli"),
                    tanggal_janji=args.get("tanggal_janji"),
                    dokter=args.get("dokter", "")
                )
        
        return msg.content if msg.content else "Maaf, saya tidak dapat memproses permintaan Anda."
    
    # =====================================
    # FUNGSI-FUNGSI UTILITY
    # =====================================
    
    def normalize_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = text.replace("&", "dan")
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def append_to_local_file(self, sheet_id, values):
        try:
            filename = f"backup_{sheet_id[:8]}.csv"
            try:
                df_existing = pd.read_csv(filename)
            except:
                df_existing = pd.DataFrame()
            df_new = pd.DataFrame([values])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filename, index=False)
            print(f"Data tersimpan di file lokal: {filename}")
            return True, "Data tersimpan di file lokal"
        except Exception as e:
            print(f"Error: {e}")
            return False, str(e)
    
    def load_pasien_data(self):
        try:
            response = requests.get(PASIEN_SHEET_URL)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df = df.fillna("")
            return df
        except Exception as e:
            print(f"ERROR: {e}")
            return pd.DataFrame()
    
    def load_jadwal_data(self):
        """Membaca data jadwal dengan cache (karena jadwal tetap setiap minggu)"""
        try:
            # Cek cache
            if self.jadwal_cache is not None and self.jadwal_cache_time is not None:
                elapsed = time.time() - self.jadwal_cache_time
                if elapsed < self.cache_duration:
                    print(f"📦 Using cached jadwal data (age: {elapsed:.0f}s)")
                    return self.jadwal_cache
            
            # Cache expired atau belum ada, ambil data baru
            print("🔄 Fetching fresh jadwal data...")
            response = requests.get(SHEET_URL)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df = df.fillna("")
            
            # Simpan ke cache
            self.jadwal_cache = df
            self.jadwal_cache_time = time.time()
            print(f"✅ Jadwal data cached (valid for {self.cache_duration}s)")
            
            return df
        except Exception as e:
            print(f"ERROR load_jadwal_data: {e}")
            # Jika error dan ada cache, kembalikan cache meskipun expired
            if self.jadwal_cache is not None:
                print("⚠️ Returning stale cache due to error")
                return self.jadwal_cache
            return pd.DataFrame()
    
    def get_all_poli(self):
        """Mengambil semua poli yang tersedia"""
        try:
            df = self.load_jadwal_data()
            if df.empty:
                return []
            
            if "klinik" not in df.columns:
                print("⚠️ Kolom 'klinik' tidak ditemukan")
                return []
            
            # Filter hanya yang praktek = Ya
            df = df[df["praktek"].astype(str).str.lower() == "ya"]
            
            poli_list = df["klinik"].unique().tolist()
            return poli_list
        except Exception as e:
            print(f"ERROR get_all_poli: {e}")
            return []
    
    def get_jadwal(self, poli: str = None, hari: str = None):
        """Mengambil jadwal dokter berdasarkan poli dan hari"""
        try:
            df = self.load_jadwal_data()
            if df.empty:
                return []
            
            # Rename kolom yang ada spasi
            df = df.rename(columns={
                'perubahan mulai ': 'perubahan_mulai',
                'perubahan selesai': 'perubahan_selesai'
            })
            
            df = df.fillna("")
            
            # Filter hanya yang praktek = Ya
            df = df[df["praktek"].astype(str).str.lower() == "ya"]
            
            # Normalisasi teks untuk pencarian
            df["klinik_normalized"] = df["klinik"].apply(self.normalize_text)
            df["hari_normalized"] = df["hari"].apply(self.normalize_text)
            
            if poli:
                poli_norm = self.normalize_text(poli)
                df = df[df["klinik_normalized"].str.contains(poli_norm, na=False)]
            
            if hari:
                hari_norm = self.normalize_text(hari)
                df = df[df["hari_normalized"].str.contains(hari_norm, na=False)]
            
            if df.empty and hari:
                print(f"⚠️ Tidak ada jadwal untuk hari {hari}, menampilkan semua jadwal poli {poli}")
                df = self.load_jadwal_data()
                df = df.rename(columns={
                    'perubahan mulai ': 'perubahan_mulai',
                    'perubahan selesai': 'perubahan_selesai'
                })
                df = df[df["praktek"].astype(str).str.lower() == "ya"]
                df["klinik_normalized"] = df["klinik"].apply(self.normalize_text)
                if poli:
                    poli_norm = self.normalize_text(poli)
                    df = df[df["klinik_normalized"].str.contains(poli_norm, na=False)]
            
            result = []
            for _, row in df.iterrows():
                # Logika penentuan jam praktek
                perubahan_mulai = row.get("perubahan_mulai", "")
                perubahan_selesai = row.get("perubahan_selesai", "")
                
                if pd.notna(perubahan_mulai) and str(perubahan_mulai).strip() != "":
                    jam_mulai = str(perubahan_mulai).strip()
                else:
                    jam_mulai = str(row.get("mulai hfis", "")).strip()
                
                if pd.notna(perubahan_selesai) and str(perubahan_selesai).strip() != "":
                    jam_selesai = str(perubahan_selesai).strip()
                else:
                    jam_selesai = str(row.get("selesai hfis", "")).strip()
                
                if jam_mulai == "" or jam_mulai == "nan":
                    jam_mulai = "-"
                if jam_selesai == "" or jam_selesai == "nan":
                    jam_selesai = "-"
                
                result.append({
                    "Poli": row.get("klinik", ""),
                    "Dokter": row.get("dokter", ""),
                    "Hari": row.get("hari", "").capitalize(),
                    "Jam": f"{jam_mulai} - {jam_selesai}" if jam_mulai != "-" and jam_selesai != "-" else f"{jam_mulai} {jam_selesai}".strip(),
                    "Status": "Praktek"
                })
            
            return result
            
        except Exception as e:
            print(f"ERROR get_jadwal: {e}")
            return []
    
    def cek_pasien_lama(self, no_medrek: str, tgl_lahir: str):
        try:
            df = self.load_pasien_data()
            if df.empty:
                return {"status": "error", "message": "Gagal mengakses data pasien."}
            df["mr_no"] = df["mr_no"].astype(str).str.strip()
            df["mr_tgl_lahir"] = df["mr_tgl_lahir"].astype(str).str.strip()
            no_medrek_clean = str(no_medrek).strip()
            tgl_lahir_clean = str(tgl_lahir).strip()
            matched = df[(df["mr_no"] == no_medrek_clean) & (df["mr_tgl_lahir"] == tgl_lahir_clean)]
            if not matched.empty:
                record = matched.iloc[0]
                return {
                    "status": "found",
                    "data": {
                        "nama": record.get('mr_nama', ''),
                        "no_medrek": record.get('mr_no', ''),
                        "no_hp": record.get('no_hp', ''),
                        "nik": record.get('nik', '')
                    }
                }
            else:
                return {"status": "not_found", "message": "Data pasien tidak ditemukan. Silakan daftar sebagai pasien baru."}
        except Exception as e:
            return {"status": "error", "message": f"Terjadi kesalahan: {str(e)}"}
    
    def buat_janji_temu_lama(self, no_medrek: str, poli: str, tanggal_janji: str, dokter: str = ""):
        try:
            df = self.load_pasien_data()
            if df.empty:
                return "Gagal mengakses data pasien."
            df["mr_no"] = df["mr_no"].astype(str).str.strip()
            no_medrek_clean = str(no_medrek).strip()
            matched = df[df["mr_no"] == no_medrek_clean]
            if matched.empty:
                return f"Data pasien dengan nomor medrek {no_medrek} tidak ditemukan."
            record = matched.iloc[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values = {
                "timestamp": timestamp,
                "nama": record.get('mr_nama', ''),
                "no_hp": record.get('no_hp', ''),
                "poli": poli,
                "dokter": dokter,
                "jenis_pasien": "Pasien Lama",
                "no_medrek": no_medrek,
                "nik": record.get('nik', '')
            }
            self.append_to_local_file(APPOINTMENT_SHEET_ID, values)
            return f"✅ Terima kasih {record.get('mr_nama', '')}, janji temu Anda untuk poli {poli} pada tanggal {tanggal_janji} telah kami terima."
        except Exception as e:
            return f"Maaf, terjadi kendala: {str(e)}"
    
    def buat_janji_temu_baru(self, nama: str, no_hp: str, nik: str, poli: str, tanggal_janji: str, dokter: str = ""):
        try:
            no_hp_clean = re.sub(r'[^0-9]', '', no_hp)
            nik_clean = re.sub(r'[^0-9]', '', nik)
            if len(nik_clean) != 16:
                return "NIK harus 16 digit angka. Mohon periksa kembali."
            no_medrek = f"RM{datetime.now().strftime('%y%m%d%H%M%S')}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values_pasien = {"timestamp": timestamp, "mr_no": no_medrek, "mr_nama": nama, "nik": nik_clean, "no_hp": no_hp_clean}
            self.append_to_local_file(PASIEN_SHEET_ID, values_pasien)
            values_appointment = {"timestamp": timestamp, "nama": nama, "no_hp": no_hp_clean, "poli": poli, "dokter": dokter, "jenis_pasien": "Pasien Baru", "no_medrek": no_medrek, "nik": nik_clean}
            self.append_to_local_file(APPOINTMENT_SHEET_ID, values_appointment)
            return f"✅ Selamat datang {nama}! Anda telah terdaftar sebagai pasien baru dengan nomor medrek: {no_medrek}\n\nJanji temu Anda untuk poli {poli} pada tanggal {tanggal_janji} telah kami terima."
        except Exception as e:
            return f"Maaf, terjadi kendala: {str(e)}"
    
    def complete_with_history(self, message: str, history: list = None) -> str:
        return self.complete(message, history)


# =====================================
# CREATE INSTANCE
# =====================================

openai_client = OpenAIClient()      