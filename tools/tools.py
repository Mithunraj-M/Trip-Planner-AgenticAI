import os
import re
import requests
import googlemaps
import openai
from serpapi import GoogleSearch
from dotenv import load_dotenv


from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
EXCHANGE_RATE_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")


openai.api_key = OPENROUTER_KEY
openai.api_base = "https://openrouter.ai/api/v1"


gmaps_places = googlemaps.Client(key=GOOGLE_PLACES_KEY)
gmaps_maps = googlemaps.Client(key=GOOGLE_MAPS_KEY)


def serpapi_search(query: str) -> str:
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        response = GoogleSearch(params).get_dict()
        results = response.get("organic_results", [])[:5]
        return '\n'.join([f"{r['title']}: {r.get('snippet', '')}" for r in results])
    except Exception as e:
        return f"[SerpAPI Error] {e}"


def google_places(query: str) -> str:
    try:
        result = gmaps_places.find_place(
            input=query,
            input_type="textquery",
            fields=["name", "formatted_address", "geometry", "rating"]
        )
        return str(result)
    except Exception as e:
        return f"[Google Places Error] {e}"


def map_distance(task: str) -> str:
    try:
        prompt = f"""
You are given a user task: "{task}"
Extract the source and destination cities for calculating distance.
Respond only in this format: Source=CityA, Destination=CityB
"""
        resp = openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "You extract cities from travel tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        reply = resp["choices"][0]["message"]["content"].strip()

        if "Source=" not in reply or "Destination=" not in reply:
            return f"[Distance Error] Unexpected format in LLM response: {reply}"

        source = reply.split("Source=")[1].split(",")[0].strip()
        dest = reply.split("Destination=")[1].split("Destination=")[-1].strip()

        dist_response = gmaps_maps.distance_matrix(source, dest)
        dist_data = dist_response["rows"][0]["elements"][0]

        if dist_data.get("status") != "OK":
            return f"[Distance Error] Could not find route from {source} to {dest}"

        return f"{source} → {dest} is {dist_data['distance']['text']} (ETA: {dist_data['duration']['text']})"

    except Exception as e:
        return f"[Distance Error] {e}"



def smart_currency_conversion(task: str) -> str:
    try:

        amount_match = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)", task)
        if not amount_match:
            return "Couldn't extract amount."
        amount = float(amount_match.group(1).replace(",", ""))

        
        prompt = f"""
You are given a user task: "{task}"
Identify the base currency (e.g., USD, EUR) and the destination country or city.
Respond only in this format: Currency=XXX, Destination=YYY
"""
        resp = openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        reply = resp["choices"][0]["message"]["content"]
        currency_code = reply.split("Currency=")[1].split(",")[0].strip().upper()
        destination = reply.split("Destination=")[1].strip()

        
        url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_KEY}/pair/{currency_code}/INR/{amount}"
        response = requests.get(url).json()

        if response.get("result") != "success":
            return f"Currency conversion failed: {response.get('error-type', 'Unknown error')}"

        converted = response["conversion_result"]
        return (f"Destination: {destination}\n"
                f"Currency: {currency_code}\n"
                f"{amount} {currency_code} ≈ ₹{converted:.2f}")
    except Exception as e:
        return f"[Currency Conversion Error] {e}"
