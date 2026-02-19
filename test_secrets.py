import os
from dotenv import load_dotenv
load_dotenv()

# Check all required env vars
required = {
    "DB_HOST":          os.getenv("DB_HOST"),
    "DB_PORT":          os.getenv("DB_PORT"),
    "DB_NAME":          os.getenv("DB_NAME"),
    "DB_USER":          os.getenv("DB_USER"),
    "DB_PASS":          os.getenv("DB_PASS"),
    "MIN_NEW_ROWS":     os.getenv("MIN_NEW_ROWS"),
    "DATA_FETCH_HOURS": os.getenv("DATA_FETCH_HOURS"),
    "MODEL_DIR":        os.getenv("MODEL_DIR"),
    "LLM_BACKEND":      os.getenv("LLM_BACKEND"),
    "GROQ_MODEL":       os.getenv("GROQ_MODEL"),
}

print("=" * 50)
print("ENV VAR CHECK")
print("=" * 50)
all_ok = True
for key, value in required.items():
    if value:
        # mask sensitive values
        if key in ("DB_PASS", "DB_USER", "DB_HOST", "DB_NAME"):
            display = value[:4] + "***"
        else:
            display = value
        print(f"  OK      {key} = {display}")
    else:
        print(f"  MISSING {key}")
        all_ok = False

print("=" * 50)
if all_ok:
    print("All env vars present — safe to run workflow")
else:
    print("Fix missing vars in .env AND in GitHub Secrets")
print("=" * 50)