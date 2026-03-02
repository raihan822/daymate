#backend/main.py
#todo: fix bug, now response from News.!

from fastapi import FastAPI  #for FastAPI

# AI Integration:
from .src.ai_integration import *
# Other Services:
from .src.services import *
from .config import PlanRequestClass

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
    return await fetch_weather(lat=lat, lon=lon)    #Just the service func() call

@app.get("/news")
async def get_news(country: str = "bd", q: str | None = None):
    return await fetch_news(country=country, q=q)

# Reasoning with LLM:-
@app.post("/plan")
async def generate_plan(req: PlanRequestClass):
    # 1. Fetch data from services
    weather = await get_weather(req.lat, req.lon)   # fetching weather (my backend api call)

    news = await get_news(country="bd") # fetching news [Default Country: BD] (my backend api call)
    headlines = [a.get("title") for a in news.get("articles", [])[:5]]  # Safe extraction of the dict.get() value with default value []

    # 2. AI Logic
    # RAG system (Ai-model) -> load_model[NAME].invoke(Message with Prompt).content:
    prompt = (
        f"User is at {req.location_name or f'{req.lat},{req.lon}'}. "
        f"Weather: {weather.get('weather')[0].get('description')}, temp {weather.get('main').get('temp')}°C. "
        f"Top headlines: {headlines}. "
        "Generate a concise daily plan (3-6 items) and practical recommendations (carry items, suggest reschedule if needed)."
    )
    message = prompt_for_model(prompt=prompt, system_instruction="You are DayMate, a helpful daily planner.")
    # print("Prompt is ===>\n",prompt)

    # 3. Calling AI Model:--
    llm = load_llm_objects['groq_model']
    if llm:
        print("\nLLM key is Found. Prompting with LLM...\n")
        response_obj = llm.invoke(message)  # made the message with the prompt above!
        response_text = response_obj.content
        return {"planning": response_text, "prompt": prompt}




    else:   # //fallback logic: manual//
        print("LLM Model not Available. manual reasoning...")
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
    # Local Testing:-
    # Payload:
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