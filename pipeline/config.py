import os
from dotenv import load_dotenv

load_dotenv()

# Database
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

# NHL API
NHL_API_BASE_URL = "https://api-web.nhle.com"
API_RATE_LIMIT_SECONDS = 1.0
API_MAX_RETRIES = 3
API_TIMEOUT_SECONDS = 10

# Pipeline
BACKFILL_SEASONS = [
    "20192020",
    "20202021",
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    # "20252026"
]

VALID_SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot"}

VALID_PIPELINE_STATUSES = {
    "success",
    "failed",
    "skipped_in_progress",
    "skipped_duplicate",
}
