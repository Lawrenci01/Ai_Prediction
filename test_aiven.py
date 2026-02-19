import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
name = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
pwd  = os.getenv("DB_PASS")

print(f"Host : {host}")
print(f"Port : {port}")
print(f"Name : {name}")
print(f"User : {user}")
print(f"Pass : {pwd[:4]}***")

url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}"
print(f"\nConnecting...")

try:
    engine = create_engine(url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM sensor_data"))
        print(f"OK - Row count: {result.scalar()}")
except Exception as e:
    print(f"FAILED: {e}")