from serpapi import GoogleSearch

from dotenv import load_dotenv
import os

load_dotenv()

params = {
    "engine": "google",
    "q": "LinkedIn login page",
    "num": 5,
    "api_key": os.getenv("SERPAPI_KEY"),
}

results = GoogleSearch(params).get_dict()

for i, item in enumerate(results.get("organic_results", []), start=1):
    print(i, item.get("title"))
    print("   ", item.get("link"))