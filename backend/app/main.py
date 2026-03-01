#backend/main.py
from fastapi import FastAPI, HTTPException  #for FastAPI
import httpx    # better alternative of `requests` that I used with BS4, Sel

# AI Integration:
from src.ai_integration import *
# Other Services:
from config import OPENWEATHER, GNEWS, PlanRequestClass

# Making Fast API Object/Instance:
app = FastAPI(title="DayMate API")

## Homepage route (Default):---
# from fastapi.responses import RedirectResponse
# @app.get("/")
# async def docs_redirect():
#       #"from root to root/docs auto redirects"
#     return RedirectResponse(url="/docs")

@app.get("/")
async def root():
    return {
        "message": "Welcome to DayMate API!",
        "swagger_ui": "https://daymate-bitmascot-backend.onrender.com/docs",
        "docs_url": "/docs",
        "status": "running"
    }


# My Main APIs:--->

# @app.get("/health")
# async def health():
#     return {"status": "ok"}


@app.get("/weather")
async def get_weather(lat: float, lon: float):
    # GET, 'http://127.0.0.1:8000/weather?lat=23.7104&lon=90.40744' #my_backend_api
    if not OPENWEATHER['api_key'] or not OPENWEATHER['base_url']:
        raise HTTPException(status_code=500, detail="OPENWEATHER_KEY or Open Weather URL not configured")
    params = {  #payload
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER['api_key'],
        "units": "metric"
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(OPENWEATHER['base_url'], params=params)    #"https://api.openweathermap.org/data/2.5/weather" with payload

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Weather API error")

    return r.json()

@app.get("/news")
async def get_news(country: str = "bd", q: str | None = None):
    if not GNEWS['api_key'] or not GNEWS['base_url']:
        # It's better practice to use the actual variable name in the error message
        raise HTTPException(status_code=500, detail="GNEWS_API_KEY or URL not configured")

    params = {  #payload
        "apikey": GNEWS['api_key'],      # GNews API KEY
        "category": "general",       # category
        "lang": "en",                # language
        "max": 5,                    # max results
        "country": country           # Dynamic country (defaults to 'bd')
    }

    if q:
        params["q"] = q

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(GNEWS['base_url'], params=params)

    if r.status_code != 200:
        # Check the response text for specific GNews error messages
        raise HTTPException(status_code=502, detail=f"GNews API error: {r.text}")

    # news = r.json()
    # print([a.get("title") for a in news.get("articles", [])[:5]])
    return r.json()

# Reasoning with LLM:-
@app.post("/plan")
async def generate_plan(req: PlanRequestClass):
    # fetching weather (my backend api call)
    weather = await get_weather(req.lat, req.lon)
    # fetching news [Default Country: BD] (my backend api call)
    news = await get_news(country="bd")
    headlines = [a.get("title") for a in news.get("articles", [])[:5]]  # Safe extraction of the dict.get() value with default value []




    # RAG system (Ai-model) -> load_model[NAME].invoke(Message with Prompt).content:
    prompt = (
        f"User is at {req.location_name or f'{req.lat},{req.lon}'}. "
        f"Weather: {weather.get('weather')[0].get('description')}, temp {weather.get('main').get('temp')}°C. "
        f"Top headlines: {headlines}. "
        "Generate a concise daily plan (3-6 items) and practical recommendations (carry items, suggest reschedule if needed)."
    )
    message = prompt_for_model(prompt=prompt, system_instruction="You are DayMate, a helpful daily planner.")
    # print("Prompt is ===>\n",prompt)
    # Calling AI Model:--
    if load_llm_objects['groq_model']:
        print("\nLLM key is Found. Prompting with LLM...\n")
        response_obj = load_llm_objects['groq_model'].invoke(message)  # made the message with the prompt above!
        response_text = response_obj.content
        return {"planning": response_text, "prompt": prompt}




    else:   # //fallback logic: manual//
        print("LLM key not Found. manual reasoning...")
        plan = []
        desc = weather.get('weather')[0].get('main', '')
        if 'rain' in desc.lower():
            plan.append("Carry an umbrella / waterproof jacket.")
            plan.append("Avoid scheduling long outdoor meetings; consider indoor alternatives.")
        elif 'clear' in desc.lower() or 'sun' in desc.lower():
            plan.append("Good day for outdoor activities: short walk or exercise.")
        else:
            plan.append("Check local conditions before leaving; bring a light jacket.")
        # Add a headline-driven advisory if serious news present (basic heuristic: look for 'alert', 'storm', 'strike')
        critical = [h for h in headlines if
                    h and any(k in h.lower() for k in ['alert', 'storm', 'strike', 'emergency', 'flood'])]
        if critical:
            plan.append(f"Important news: {critical[0]} — consider rescheduling sensitive plans.")
        plan.append("Suggested schedule: morning focus work, afternoon errands with buffer time.")
        return {"planning": '\n'.join(plan), "prompt": prompt}

if __name__ == "__main__":
    # Test Payload:
    instance_payload = PlanRequestClass(  #payload values type checked with pydentic Python-library
        location_name="bd",
        lat=23.7104,
        lon=90.40744
    )

    # Async runner to call the async type functions
    import asyncio
    result = asyncio.run(generate_plan(req=instance_payload))
    print("Prompt is ===>\n",result.get("prompt","No Prompt Pushed!"))
    print("\nLLM Result ===>\n",result.get("planning","No Result from LLM!"))



#Notes:----
# uvicorn command for Render
    # Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

# During Dev:--
    # uvicorn filename:fastapiObj --reload
    # uvicorn main:app --reload

    # pyenv shell 3.11.14
    # uvicorn main:app --reload --port $PORT