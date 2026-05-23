import json
import time
import fitz
import pdfplumber
import hashlib
from pathlib import Path
from datetime import datetime
from collections import deque
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# ══════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════
OPENROUTER_API_KEY = ""
PINECONE_API_KEY   = "YOUR_PINECONE_API_KEY_HERE"
PINECONE_INDEX     = "college-chatbot"
PINECONE_REGION    = "us-east-1"
EMBEDDING_MODEL    = "openai/text-embedding-3-small"
EMBEDDING_DIM      = 1536
CHAT_MODEL         = "minimax/minimax-m2.5"
COLLEGE_NAME       = "VLB Janakiammal College of Arts and Science"

MONGO_URI  = "mongodb://localhost:27017/"
MONGO_DB   = "college_chatbot_db"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
TOP_K         = 5
# ══════════════════════════════════════════════

DOCS_DIR = Path("uploaded_docs")
DOCS_DIR.mkdir(exist_ok=True)

# ── CLIENTS ───────────────────────────────────
openrouter = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ── MONGODB ───────────────────────────────────
mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[MONGO_DB]
users_col    = db["users"]
history_col  = db["chat_history"]
docs_col     = db["documents"]

users_col.create_index("email",    unique=True)
users_col.create_index("user_id",  unique=True)
history_col.create_index("user_id", unique=True)
docs_col.create_index("filename",   unique=True)

# ── PINECONE ──────────────────────────────────
pc         = Pinecone(api_key=PINECONE_API_KEY)
_index_obj = None

def get_index():
    global _index_obj
    if _index_obj is not None:
        return _index_obj
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name      = PINECONE_INDEX,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
    _index_obj = pc.Index(PINECONE_INDEX)
    return _index_obj

# ── FASTAPI APP ───────────────────────────────
app = FastAPI(title="Student Support Chatbot (RAG + MongoDB)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── TIMING MIDDLEWARE ─────────────────────────
request_log: deque = deque(maxlen=50)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - t0) * 1000

    response.headers["X-Response-Time"] = f"{elapsed:.0f}ms"

    emoji = (
        "🟢" if elapsed < 1000 else
        "🟡" if elapsed < 2000 else
        "🟠" if elapsed < 4000 else
        "🔴"
    )
    print(f"{emoji} {request.method} {request.url.path} → {elapsed:.0f}ms")

    request_log.append({
        "time":   datetime.now().strftime("%H:%M:%S"),
        "method": request.method,
        "path":   request.url.path,
        "ms":     round(elapsed, 1),
        "status": response.status_code,
        "rating": (
            "excellent"  if elapsed < 1000 else
            "good"       if elapsed < 2000 else
            "acceptable" if elapsed < 4000 else
            "slow"
        ),
    })
    return response

# ── MODELS ────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    user_id: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# ══════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════
@app.post("/auth/register")
def register(req: RegisterRequest):
    email = req.email.strip().lower()
    if not req.name.strip():
        raise HTTPException(400, "Name cannot be empty.")
    if len(req.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters.")
    user_id  = email.replace("@", "_at_").replace(".", "_")
    user_doc = {
        "name":     req.name.strip(),
        "email":    email,
        "password": req.password,
        "user_id":  user_id,
        "created":  datetime.now().isoformat(),
    }
    try:
        users_col.insert_one(user_doc)
    except DuplicateKeyError:
        raise HTTPException(400, "Email already registered.")
    return {"message": "Registered successfully!", "user_id": user_id, "name": req.name.strip()}

@app.post("/auth/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    user  = users_col.find_one({"email": email}, {"_id": 0})
    if not user:
        raise HTTPException(401, "Email not found.")
    if user["password"] != req.password:
        raise HTTPException(401, "Incorrect password.")
    return {
        "message": "Login successful!",
        "user_id": user["user_id"],
        "name":    user["name"],
        "email":   user["email"],
    }

# ══════════════════════════════════════════════
#  DOCUMENT PROCESSING
# ══════════════════════════════════════════════
def read_pdf(file_path: Path) -> str:
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
            if text.strip():
                return text.strip()
    except Exception:
        pass
    try:
        doc  = fitz.open(str(file_path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return ""

def read_txt(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""

def chunk_text(text: str, filename: str) -> list[dict]:
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunk_id = hashlib.md5(f"{filename}_{idx}".encode()).hexdigest()
            chunks.append({"id": chunk_id, "text": chunk, "filename": filename, "chunk_no": idx})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ══════════════════════════════════════════════
#  EMBEDDINGS + PINECONE
# ══════════════════════════════════════════════
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openrouter.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]

def upsert_chunks(chunks: list[dict]):
    if not chunks:
        return
    index = get_index()
    vecs  = get_embeddings([c["text"] for c in chunks])
    items = [
        (c["id"], v, {"text": c["text"], "filename": c["filename"], "chunk_no": c["chunk_no"]})
        for c, v in zip(chunks, vecs)
    ]
    for i in range(0, len(items), 100):
        index.upsert(vectors=items[i:i + 100])

def delete_file_chunks(filename: str):
    file_path = DOCS_DIR / filename
    if not file_path.exists():
        return
    ext  = file_path.suffix.lower()
    text = read_pdf(file_path) if ext == ".pdf" else read_txt(file_path)
    if not text:
        return
    ids = [c["id"] for c in chunk_text(text, filename)]
    if ids:
        get_index().delete(ids=ids)

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    q_vec  = get_embeddings([query])[0]
    result = get_index().query(vector=q_vec, top_k=top_k, include_metadata=True)
    return [
        {
            "text":     m.metadata.get("text", ""),
            "filename": m.metadata.get("filename", ""),
            "score":    round(m.score, 3),
        }
        for m in result.matches
    ]

# ══════════════════════════════════════════════
#  LANGUAGE HELPERS
# ══════════════════════════════════════════════
def detect_language(message: str) -> str:
    char_sets = {
        "tamil":     set("அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன"),
        "malayalam": set("അആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹ"),
        "hindi":     set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"),
        "telugu":    set("అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహ"),
        "kannada":   set("ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹ"),
    }
    counts = {lang: sum(1 for c in message if c in chars) for lang, chars in char_sets.items()}
    best   = max(counts, key=counts.get)
    return best if counts[best] > 0 else "english"

def get_language_instruction(lang: str) -> str:
    return {
        "tamil":     "Reply in Tamil only. ",
        "malayalam": "Reply in Malayalam only. ",
        "hindi":     "Reply in Hindi only. ",
        "telugu":    "Reply in Telugu only. ",
        "kannada":   "Reply in Kannada only. ",
        "english":   "",
    }.get(lang, "")

GENERAL_KEYWORDS = [
    "full form","fullform","what is","what are","define","definition",
    "meaning of","abbreviation","stands for","expand","short form",
    "who is","who was","when was","where is","how does","explain",
    "tell me about","describe","example of","difference between",
    "என்ன","யார்","எப்போது","எங்கே","எப்படி","விளக்கு","முழு வடிவம்",
    "എന്താണ്","ആരാണ്","എप्पोൾ","എവിടെ","എങ്ങനെ",
    "क्या है","कौन है","कब","कहाँ","कैसे","बताओ","समझाओ",
    "ఏమిటి","ఎవరు","ఎప్పుడు","ఎక్కడ","ఎలా",
    "ಏನು","ಯಾರು","ಯಾವಾಗ","ಎಲ್ಲಿ","ಹೇಗೆ",
]

def is_general_question(message: str) -> bool:
    return any(kw in message.lower() for kw in GENERAL_KEYWORDS)

# ══════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════
def load_history(user_id: str) -> list:
    record = history_col.find_one({"user_id": user_id}, {"_id": 0})
    if record:
        return [
            m for m in record.get("messages", [])
            if isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
        ]
    return []

def save_history(user_id: str, history: list):
    history_col.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "messages": history, "last_updated": datetime.now().isoformat()}},
        upsert=True,
    )

FORMATTING_RULES = (
    "Use emojis (📌✅📋) instead of markdown. "
    "Numbered lists for steps. CAPITALS for section labels. "
    "Be brief and direct. Never ask clarifying questions.\n"
)

# ══════════════════════════════════════════════
#  ADMIN ROUTES
# ══════════════════════════════════════════════
@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".md"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Only {allowed} files allowed.")
    file_path = DOCS_DIR / file.filename
    file_path.write_bytes(await file.read())
    text = read_pdf(file_path) if ext == ".pdf" else read_txt(file_path)
    if not text:
        raise HTTPException(422, "Could not extract text from file.")
    chunks = chunk_text(text, file.filename)
    upsert_chunks(chunks)
    docs_col.update_one(
        {"filename": file.filename},
        {"$set": {
            "filename":    file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "char_count":  len(text),
        }},
        upsert=True,
    )
    return {"message": f"'{file.filename}' uploaded and indexed successfully.", "chunks_added": len(chunks)}

@app.get("/admin/documents")
def list_documents():
    return {"documents": list(docs_col.find({}, {"_id": 0}))}

@app.delete("/admin/documents/{filename}")
def delete_document(filename: str):
    if not docs_col.find_one({"filename": filename}):
        raise HTTPException(404, "File not found.")
    delete_file_chunks(filename)
    (DOCS_DIR / filename).unlink(missing_ok=True)
    docs_col.delete_one({"filename": filename})
    return {"message": f"'{filename}' deleted from storage, vector index, and database."}

@app.get("/admin/users")
def list_users():
    users = list(users_col.find({}, {"_id": 0, "password": 0}))
    for u in users:
        record = history_col.find_one({"user_id": u["user_id"]}, {"_id": 0})
        u["message_count"] = len(record.get("messages", [])) if record else 0
    return {"users": users}

# ══════════════════════════════════════════════
#  CHAT  (RAG core)
# ══════════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    lang             = detect_language(request.message)
    lang_instruction = get_language_instruction(lang)
    history          = load_history(request.user_id)

    if is_general_question(request.message):
        system_prompt = (
            f"{lang_instruction}"
            f"You are a friendly student assistant at {COLLEGE_NAME}. "
            "Answer directly and clearly. Be short and helpful.\n"
            f"{FORMATTING_RULES}"
        )
    else:
        matches = retrieve_chunks(request.message, top_k=TOP_K)
        if matches:
            context_block = "\n\n".join(
                f"[{m['filename']} | score {m['score']}]\n{m['text']}" for m in matches
            )
            system_prompt = (
                f"{lang_instruction}"
                f"You are a student support assistant at {COLLEGE_NAME}. "
                "Answer ONLY using the CONTEXT below. Be direct and concise. "
                "If the answer is not in the context, say: 'Please contact the college office.'\n"
                f"{FORMATTING_RULES}"
                f"\nCONTEXT:\n{context_block}"
            )
        else:
            system_prompt = (
                f"{lang_instruction}"
                f"You are a student support assistant at {COLLEGE_NAME}. "
                "No specific documents were found. Answer clearly and concisely.\n"
                f"{FORMATTING_RULES}"
            )

    recent   = history[-4:] if len(history) >= 4 else history
    messages = [{"role": "system", "content": system_prompt}]
    for m in recent:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": request.message})

    total_chars = sum(len(m["content"]) for m in messages)
    print(f"DEBUG — messages: {len(messages)}, chars: {total_chars}, ~tokens: {total_chars // 4}")

    try:
        response = openrouter.chat.completions.create(
            model=CHAT_MODEL, messages=messages, max_tokens=400, temperature=0.3,
        )
        reply = response.choices[0].message.content
        reply = reply.strip() if reply and reply.strip() else "Sorry, I could not generate a response."
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)}")

    history.append({"role": "user",      "content": request.message, "timestamp": datetime.now().isoformat()})
    history.append({"role": "assistant", "content": reply,           "timestamp": datetime.now().isoformat()})
    save_history(request.user_id, history)

    return ChatResponse(reply=reply, user_id=request.user_id)

# ══════════════════════════════════════════════
#  HISTORY ROUTES
# ══════════════════════════════════════════════
@app.get("/chat/history/{user_id}")
def get_chat_history(user_id: str):
    history = load_history(user_id)
    return {"user_id": user_id, "count": len(history), "history": history}

@app.delete("/chat/history/{user_id}")
def clear_chat_history(user_id: str):
    history_col.delete_one({"user_id": user_id})
    return {"message": f"History cleared for '{user_id}'."}

# ══════════════════════════════════════════════
#  DEBUG ROUTES
# ══════════════════════════════════════════════
@app.get("/debug/rag")
def debug_rag(query: str = "college fees"):
    matches = retrieve_chunks(query, top_k=TOP_K)
    return {"query": query, "top_k": TOP_K, "results": matches}

@app.get("/debug/db")
def debug_db():
    return {
        "users":          users_col.count_documents({}),
        "chat_histories": history_col.count_documents({}),
        "documents":      docs_col.count_documents({}),
        "mongo_status":   "connected",
    }

@app.get("/debug/speed")
def debug_speed():
    logs = list(request_log)
    if not logs:
        return {"message": "No requests logged yet. Make some API calls first then check again."}
    times = [r["ms"] for r in logs]
    return {
        "last_requests": logs,
        "summary": {
            "average_ms":   round(sum(times) / len(times), 1),
            "fastest_ms":   round(min(times), 1),
            "slowest_ms":   round(max(times), 1),
            "total_logged": len(logs),
        },
    }

# ══════════════════════════════════════════════
#  ROOT
# ══════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status":   "running",
        "project":  "Student Support Chatbot (RAG + MongoDB)",
        "college":  COLLEGE_NAME,
        "rag":      f"Pinecone ({PINECONE_INDEX}) + {EMBEDDING_MODEL}",
        "database": f"MongoDB → {MONGO_DB}",
        "routes": [
            "POST /auth/register", "POST /auth/login",
            "POST /chat", "GET /chat/history/{user_id}", "DELETE /chat/history/{user_id}",
            "POST /admin/upload", "GET /admin/documents", "DELETE /admin/documents/{filename}",
            "GET /admin/users",
            "GET /debug/speed", "GET /debug/db", "GET /debug/rag",
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)