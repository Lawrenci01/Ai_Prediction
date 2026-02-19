from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME')}"

print(f"Connecting to: {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")

try:
    engine = create_engine(url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM sensor_data"))
        print(f"Row count: {result.scalar()}")
        print("DB connection OK")
except Exception as e:
    print(f"DB connection FAILED: {e}")