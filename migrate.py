
import json
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# ── CONFIG (must match main.py) ───────────────
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB  = "college_chatbot_db"

# ── OLD FILE PATHS ────────────────────────────
USERS_FILE  = Path("users.json")
HISTORY_DIR = Path("chat_history")
DOCS_DIR    = Path("uploaded_docs")

# ── CONNECT ───────────────────────────────────
client      = MongoClient(MONGO_URI)
db          = client[MONGO_DB]
users_col   = db["users"]
history_col = db["chat_history"]
docs_col    = db["documents"]

# Indexes
users_col.create_index("email",    unique=True)
users_col.create_index("user_id",  unique=True)
history_col.create_index("user_id", unique=True)
docs_col.create_index("filename",   unique=True)

print("=" * 50)
print("  MongoDB Migration Tool")
print("=" * 50)

# ══════════════════════════════════════════════
#  1. MIGRATE USERS
# ══════════════════════════════════════════════
migrated_users = 0
skipped_users  = 0

if USERS_FILE.exists():
    try:
        users_data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        for email, user in users_data.items():
            try:
                users_col.insert_one(user)
                migrated_users += 1
            except DuplicateKeyError:
                skipped_users += 1
                print(f"  ⚠  User already exists, skipped: {email}")
    except Exception as e:
        print(f"  ✗  Failed to read users.json: {e}")
else:
    print("  ⚠  users.json not found, skipping users migration.")

print(f"\n✅ Users migrated : {migrated_users}")
print(f"   Users skipped  : {skipped_users}")

# ══════════════════════════════════════════════
#  2. MIGRATE CHAT HISTORIES
# ══════════════════════════════════════════════
migrated_hist = 0
skipped_hist  = 0

if HISTORY_DIR.exists():
    for f in HISTORY_DIR.glob("*.json"):
        user_id = f.stem
        try:
            messages = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                print(f"  ⚠  Unexpected format in {f.name}, skipping.")
                continue
            history_col.update_one(
                {"user_id": user_id},
                {"$set": {
                    "user_id":      user_id,
                    "messages":     messages,
                    "last_updated": datetime.now().isoformat(),
                }},
                upsert=True,
            )
            migrated_hist += 1
        except Exception as e:
            skipped_hist += 1
            print(f"  ✗  Failed {f.name}: {e}")
else:
    print("  ⚠  chat_history/ folder not found, skipping history migration.")

print(f"\n✅ Chat histories migrated : {migrated_hist}")
print(f"   Chat histories skipped  : {skipped_hist}")

# ══════════════════════════════════════════════
#  3. REGISTER EXISTING DOCUMENTS (metadata only)
# ══════════════════════════════════════════════
migrated_docs = 0

if DOCS_DIR.exists():
    for f in DOCS_DIR.glob("*"):
        if f.suffix.lower() in {".pdf", ".txt", ".md"}:
            try:
                docs_col.update_one(
                    {"filename": f.name},
                    {"$set": {
                        "filename":    f.name,
                        "uploaded_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        "chunk_count": 0,   # will be updated on next re-upload
                        "char_count":  f.stat().st_size,
                    }},
                    upsert=True,
                )
                migrated_docs += 1
            except Exception as e:
                print(f"  ✗  Failed to register {f.name}: {e}")
else:
    print("  ⚠  uploaded_docs/ folder not found.")

print(f"\n✅ Documents registered: {migrated_docs}")

# ══════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("  Migration Summary")
print("=" * 50)
print(f"  Database     : {MONGO_DB}")
print(f"  Users        : {users_col.count_documents({})}")
print(f"  Histories    : {history_col.count_documents({})}")
print(f"  Documents    : {docs_col.count_documents({})}")
print("=" * 50)
print("\n🎉 Migration complete! You can now run the new main.py")
print("   Tip: Keep your old JSON files as backup for now.\n")