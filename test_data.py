from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME')}"
engine = create_engine(url)

with engine.connect() as conn:
    # date range
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total_rows,
            MIN(recorded_at) as oldest,
            MAX(recorded_at) as newest
        FROM sensor_data
    """))
    row = result.fetchone()
    print(f"Total rows : {row[0]}")
    print(f"Oldest     : {row[1]}")
    print(f"Newest     : {row[2]}")

    # check nulls
    result = conn.execute(text("""
        SELECT
            SUM(CASE WHEN co2_density IS NULL THEN 1 ELSE 0 END) as null_co2,
            SUM(CASE WHEN temperature_c IS NULL THEN 1 ELSE 0 END) as null_temp,
            SUM(CASE WHEN humidity IS NULL THEN 1 ELSE 0 END) as null_humidity
        FROM sensor_data
    """))
    row = result.fetchone()
    print(f"Null co2   : {row[0]}")
    print(f"Null temp  : {row[1]}")
    print(f"Null humid : {row[2]}")