import os
import uuid
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
OPENROUTE_KEY   = os.getenv("OPENROUTE_KEY")

app = FastAPI(title="AI Travel Planner - Stage 4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}


# ─────────────────────────────────────────
# OPENROUTESERVICE — TRAVEL TIME
# ─────────────────────────────────────────

def geocode(place: str) -> tuple:
    """Convert place name to lat/lon using free Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "TravelAgentAI/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        if data:
            return float(data[0]["lon"]), float(data[0]["lat"])
    except Exception as e:
        print(f"Geocode error: {e}")
    return None, None


def get_travel_time(origin: str, destination: str) -> dict:
    """Get real driving time using OpenRouteService (free, no card)."""
    if not OPENROUTE_KEY or not origin or origin == "Not specified":
        return {"duration": "N/A", "distance": "N/A", "status": "no_key"}
    try:
        # Geocode both places
        orig_lon, orig_lat = geocode(origin)
        dest_lon, dest_lat = geocode(destination)

        if not orig_lon or not dest_lon:
            return {"duration": "Unable to geocode", "distance": "N/A", "status": "geocode_error"}

        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {
            "Authorization": OPENROUTE_KEY,
            "Content-Type": "application/json"
        }
        body = {
            "coordinates": [[orig_lon, orig_lat], [dest_lon, dest_lat]]
        }
        r = requests.post(url, json=body, headers=headers, timeout=8)
        data = r.json()

        if "routes" in data and data["routes"]:
            seg = data["routes"][0]["summary"]
            duration_mins = round(seg["duration"] / 60)
            distance_km   = round(seg["distance"] / 1000, 1)

            if duration_mins >= 60:
                hrs  = duration_mins // 60
                mins = duration_mins % 60
                duration_text = f"{hrs}h {mins}min" if mins else f"{hrs} hours"
            else:
                duration_text = f"{duration_mins} mins"

            return {
                "duration": duration_text,
                "distance": f"{distance_km} km",
                "duration_mins": duration_mins,
                "status": "ok"
            }
    except Exception as e:
        print(f"ORS error: {e}")
    return {"duration": "4-5 hours (estimate)", "distance": "~150 km", "status": "fallback"}


# ─────────────────────────────────────────
# OPENWEATHERMAP — WEATHER
# ─────────────────────────────────────────

def get_weather(city: str) -> dict:
    """Get current weather + 3-day forecast (free tier)."""
    if not OPENWEATHER_KEY:
        return {"status": "no_key"}
    try:
        # Current weather
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        if data.get("cod") == 200:
            weather = {
                "city":       data["name"],
                "temp_c":     round(data["main"]["temp"]),
                "feels_like": round(data["main"]["feels_like"]),
                "condition":  data["weather"][0]["description"].title(),
                "humidity":   data["main"]["humidity"],
                "wind_kph":   round(data["wind"]["speed"] * 3.6),
                "status":     "ok"
            }

            # 3-day forecast
            furl = "https://api.openweathermap.org/data/2.5/forecast"
            fr   = requests.get(furl, params=params, timeout=5)
            fd   = fr.json()

            if fd.get("cod") == "200":
                days = {}
                for item in fd["list"]:
                    date = item["dt_txt"].split(" ")[0]
                    if date not in days and len(days) < 3:
                        days[date] = {
                            "date":     date,
                            "temp_max": round(item["main"]["temp_max"]),
                            "temp_min": round(item["main"]["temp_min"]),
                            "condition": item["weather"][0]["description"].title(),
                        }
                weather["forecast"] = list(days.values())

            return weather

    except Exception as e:
        print(f"Weather error: {e}")
    return {"status": "error"}


# ─────────────────────────────────────────
# BUDGET ALLOCATOR
# ─────────────────────────────────────────

WEIGHTS = {
    "budget":  {"transport":0.30,"hotel":0.25,"food":0.22,"activities":0.15,"buffer":0.08},
    "mid":     {"transport":0.28,"hotel":0.28,"food":0.20,"activities":0.16,"buffer":0.08},
    "luxury":  {"transport":0.25,"hotel":0.35,"food":0.18,"activities":0.15,"buffer":0.07},
}

def allocate_budget(total: float, style: str, has_flight: bool = True) -> dict:
    sk = "luxury" if "lux" in style.lower() else "budget" if "budget" in style.lower() else "mid"
    w  = dict(WEIGHTS[sk])
    if not has_flight:
        extra = w["transport"] - 0.08
        w["transport"] = 0.08
        w["hotel"] += extra * 0.6
        w["food"]  += extra * 0.4
    return {k: round(total * v, 2) for k, v in w.items()}


# ─────────────────────────────────────────
# SLOT EXTRACTOR
# ─────────────────────────────────────────

EXTRACTOR_PROMPT = """You are a travel slot extractor.
Return ONLY a JSON object — no markdown, no explanation.

Fields:
- origin: departure city (string or null)
- destination: destination city (string or null)
- days: number of days (integer or null)
- budget: budget in USD (integer or null)
- style: budget | mid | luxury (string or null)
- has_flight: boolean (default true)
- missing: list of missing required fields

Required: destination, days, budget
Defaults if missing: origin="Not specified", style="mid"

Example: {"origin":"Karachi","destination":"Dubai","days":3,"budget":2500,"style":"luxury","has_flight":true,"missing":[]}
"""

MISSING_Q = {
    "destination": "Where would you like to travel to?",
    "days":        "How many days is your trip?",
    "budget":      "What is your total budget in USD?",
}

def extract_slots(msg: str, existing: dict) -> dict:
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300, temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":EXTRACTOR_PROMPT},
                {"role":"user","content":f"Existing:{json.dumps(existing)}\nMessage:{msg}"}
            ]
        )
        raw = res.choices[0].message.content.strip().replace("```json","").replace("```","")
        merged = {**existing}
        for k,v in json.loads(raw).items():
            if v is not None:
                merged[k] = v
        return merged
    except Exception:
        return existing

def missing_fields(slots: dict) -> list:
    return [f for f in ["destination","days","budget"] if not slots.get(f)]

def clarification(missing: list) -> str:
    qs = [MISSING_Q[f] for f in missing if f in MISSING_Q]
    if len(qs) == 1: return qs[0]
    return "I need a few more details:\n" + "\n".join(f"• {q}" for q in qs)


# ─────────────────────────────────────────
# PLANNER PROMPT
# ─────────────────────────────────────────

def build_prompt(slots, budget, travel, weather) -> str:
    # Travel block
    if travel.get("status") == "ok":
        t_block = f"✅ Real driving time: {travel['duration']} | Distance: {travel['distance']}"
    else:
        t_block = "Estimated travel time: 4-5 hours"

    # Weather block
    if weather.get("status") == "ok":
        w_block = (
            f"✅ Live weather in {weather.get('city', slots['destination'])}: "
            f"{weather['temp_c']}°C, {weather['condition']}, "
            f"Humidity {weather['humidity']}%, Wind {weather['wind_kph']} km/h"
        )
        if weather.get("forecast"):
            w_block += "\nForecast: " + " | ".join(
                f"{d['date'][5:]}: {d['temp_min']}–{d['temp_max']}°C {d['condition']}"
                for d in weather["forecast"]
            )
    else:
        w_block = "Weather data unavailable"

    days = int(slots['days'])
    hotel_per_night_mid = round(budget['hotel'] * 0.9 / days)

    return f"""You are an expert AI travel planning assistant.

TRIP DETAILS:
- Origin: {slots.get('origin','Not specified')}
- Destination: {slots['destination']}
- Duration: {days} days
- Budget: ${slots['budget']} USD
- Style: {slots.get('style','mid')}

REAL-WORLD DATA (use these exact figures in your response):
{t_block}
{w_block}

BUDGET (use exact amounts):
- Transport: ${budget['transport']}
- Hotel: ${budget['hotel']} total (≈${hotel_per_night_mid}/night mid-range)
- Food: ${budget['food']}
- Activities: ${budget['activities']}
- Buffer: ${budget['buffer']}

OUTPUT FORMAT — follow exactly:

✈️ TRIP OVERVIEW
- Departure: {slots.get('origin','Not specified')}
- Destination: {slots['destination']}
- Duration: {days} days
- Budget: ${slots['budget']}
- Travel Style: {slots.get('style','mid-range')}
- Weather: {weather.get('temp_c','?')}°C, {weather.get('condition','check forecast')}
- Estimated Total Cost: ${slots['budget']}

🚆 TRANSPORT PLAN
Use the real travel time above. Include cost breakdown, options, timing.

🏨 HOTEL OPTIONS
Option 1 - Budget: [name] ~${round(budget['hotel']*0.6/days)}/night
Option 2 - Mid-range: [name] ~${hotel_per_night_mid}/night
Option 3 - Luxury: [name] ~${round(budget['hotel']*1.3/days)}/night

🗓️ DAY-BY-DAY ITINERARY
{"".join(f"""
Day {i}:
Morning (9:00 AM):
- Activity
- Breakfast spot (~$X)
Afternoon (1:00 PM):
- Activity
- Lunch spot (~$X)
Evening (6:00 PM):
- Activity
- Dinner spot (~$X)
""" for i in range(1, days+1))}

🍽️ FOOD HIGHLIGHTS
- Must-try dish 1
- Must-try dish 2
- Best local restaurant: [name] (~$X/person)
- Street food spot: [name] (~$X/person)
- Price range: $X–$Y per meal

💰 BUDGET BREAKDOWN
- Transport: ${budget['transport']}
- Hotel ({days} nights): ${budget['hotel']}
- Food: ${budget['food']}
- Activities: ${budget['activities']}
- Emergency Buffer: ${budget['buffer']}
- TOTAL: ${slots['budget']}

💡 TIPS
- Money saving tip 1
- Money saving tip 2
- Luxury upgrade option 1
- Luxury upgrade option 2

RULES:
- Mention the real weather and suggest appropriate clothing
- Use the exact travel time from real data
- Max 3 attractions per day
- Keep timings realistic with travel gaps
"""


# ─────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    session_id: str = ""


@app.post("/session/new")
def new_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"history":[],"slots":{},"plan_generated":False,"travel_info":{},"weather":{}}
    return {"session_id": sid}


@app.post("/plan")
def plan_trip(req: ChatMessage):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = {"history":[],"slots":{},"plan_generated":False,"travel_info":{},"weather":{}}

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    s = sessions[sid]
    s["history"].append({"role":"user","content":req.message.strip()})

    # Extract slots
    s["slots"] = extract_slots(req.message.strip(), s["slots"])
    mf = missing_fields(s["slots"])

    if mf:
        reply = clarification(mf)
        s["history"].append({"role":"assistant","content":reply})
        return {"reply":reply,"session_id":sid,"slots":s["slots"],"status":"collecting_info","missing":mf}

    slots = s["slots"]
    if not slots.get("origin"): slots["origin"] = "Not specified"
    if not slots.get("style"):  slots["style"]  = "mid"

    # Fetch real data only once
    if not s["plan_generated"]:
        print(f"→ Travel time: {slots.get('origin')} → {slots['destination']}")
        s["travel_info"] = get_travel_time(slots.get("origin",""), slots["destination"])
        print(f"→ Weather: {slots['destination']}")
        s["weather"] = get_weather(slots["destination"])
        print(f"  Travel: {s['travel_info']}")
        print(f"  Weather: {s['weather']}")

    budget = allocate_budget(float(slots["budget"]), slots["style"], slots.get("has_flight",True))
    prompt = build_prompt(slots, budget, s["travel_info"], s["weather"])

    if s["plan_generated"]:
        messages = [{"role":"system","content":prompt}] + s["history"]
    else:
        messages = [
            {"role":"system","content":prompt},
            {"role":"user","content":f"Generate the complete {slots['days']}-day plan for {slots['destination']}."}
        ]

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=4000, temperature=0.7,
            messages=messages
        )
        reply = res.choices[0].message.content
    except Exception as e:
        s["history"].pop()
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    s["history"].append({"role":"assistant","content":reply})
    s["plan_generated"] = True

    return {
        "reply":          reply,
        "session_id":     sid,
        "slots":          slots,
        "budget_breakdown": budget,
        "travel_info":    s["travel_info"],
        "weather":        s["weather"],
        "status":         "plan_ready"
    }


@app.get("/weather/{city}")
def weather_check(city: str):
    return get_weather(city)

@app.get("/travel-time")
def travel_check(origin: str, destination: str):
    return get_travel_time(origin, destination)

@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": sessions[session_id]["history"]}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id] = {"history":[],"slots":{},"plan_generated":False,"travel_info":{},"weather":{}}
    return {"status":"cleared"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage": "4 — OpenRouteService + OpenWeatherMap",
        "openweather":  "connected" if OPENWEATHER_KEY else "missing key",
        "openroute":    "connected" if OPENROUTE_KEY else "missing key",
        "active_sessions": len(sessions)
    }

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))