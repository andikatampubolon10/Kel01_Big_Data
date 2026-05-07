import os

# ---------- PostgreSQL (Supabase / Local Postgres) ----------
# Contoh env var:
#   PG_HOST=xxxx.supabase.co
#   PG_PORT=5432
#   PG_DB=postgres
#   PG_USER=postgres
#   PG_PASSWORD=...
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "postgres")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

PG_JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
PG_JDBC_PROPS = {
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",
}

# ---------- MongoDB Atlas ----------
# Contoh:
#   MONGO_URI="mongodb+srv://USER:PASS@CLUSTER.mongodb.net"
#   MONGO_DB="retail_dw"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "retail_dw")