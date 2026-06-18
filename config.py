import os

from dotenv import load_dotenv

load_dotenv(override=True)

APIFOOTBALL_KEY: str = os.getenv("APIFOOTBALL_KEY", "")
