"""
backend/main.py
"""
# Loading .env file for local testing:
from dotenv import load_dotenv
load_dotenv()   #loads the api keys stored in the .env file and then acts like os.getenv() function as like os env ver

# Other Library imports:
from fastapi import FastAPI, HTTPException  #for FastAPI
import httpx    # better alternative to requests that I used with BS4, Sel
import os

"""Best Practice. Always Do type checking with pydentic for API call with PAYLOAD(s), 
You can also combine the API_KEYs+PROVIDER_NAME+BASE_URL etc 
later API_KEY raw value can be caught with you_masked_apikey.get_secret_value()

"""
from pydantic import BaseModel, SecretStr  #for explicit type checking, Generally used with fastAPI, and other libraries requiring strict type formalities of the variables.

# All API Keys for the Project:---
class AiModelClass(BaseModel):
    model_name: str
    api_key: SecretStr
    provider_name: str
    base_url: str

instance_groq_model = AiModelClass(
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),  # Pydentic will auto convert this str into SecretStr! You can do --> `raw_key = instance_groq_model.api_key.get_secret_value()` to read/get actual api_key value.
    provider_name="openai",
    base_url="https://api.groq.com/openai/v1"
)
#instance_gemini_model = AiModelClass().. etc

OPENWEATHER = {
    'api_key' : os.getenv("OPENWEATHER_KEY"),
    'base_url': "https://api.openweathermap.org/data/2.5/weather"     #GET https://api.openweathermap.org/data/2.5/weather ?lat={lat}&lon={lon}&appid={API key}
}

GNEWS = {
    'api_key' : os.getenv("GNEWS_API_KEY"),
    'base_url' : "https://gnews.io/api/v4/top-headlines"
}




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

# LLM Integration Function:
    # Generalised Ai-Model Calling for the 'OpenAI-compatible' API providers
    # This will apply for all that follow OpenAI’s message schema
''' There are different types of LLM API Calling Style
Layer 1 → Raw provider SDK (different different company's own style of calling)
Layer 2 → API abstraction (LiteLLM)
Layer 3 → Framework abstraction (LangChain)
    > I am using LangChain's only the OpenAi format schema generalization 'langchain_openai'.
'''
from langchain_core.language_models import BaseChatModel #[Optional] for return type check.

from typing import Optional, Union
def load_llm(
        # Required Params:-
        model_provider: str,
        model_name :str,
        model_api_key :Union[str, SecretStr],

        # [Optional] BASE_URL (Required for OpenAI-compatible APIs Providers like GROQ,ChatGPT, etc.)
        base_url: Optional[str] = None,

        # General Configuration Settings:-
        model_top_p: float = 1.0,
        model_temperature: float = 0.7,    #0.7 is langchain default
        model_max_tokens: Optional[int]= None,
        # Other Configs:
        model_timeout: int=60,
        model_max_retries:int = 2
) -> BaseChatModel:
    # #Can be used to convert a str to secretstr for langchain params:-
    # from langchain_core.utils import convert_to_secret_str    #then use convert_to_secret_str() function

    if not model_api_key:
        raise ValueError(f"API Key not found. {model_api_key}")
    model_provider = model_provider.lower()

    # 1. OpenAI-Compatible APIs (OpenAI / Groq / Compatible):-
    if model_provider == 'openai':
        try:
            #pip install langchain-openai
            from langchain_openai import ChatOpenAI # ChatOpenAI is a LangChain wrapper from LangChain.
        except ImportError:
            raise ImportError("Could not import langchain-openai. "
                              "Please install it with: pip install langchain-openai")

        return ChatOpenAI(
            #General Params:
            model = model_name,
            api_key = model_api_key,
            base_url = base_url,

            #Other Configs:
            temperature = model_temperature,
            top_p = model_top_p,
            max_tokens = model_max_tokens,
            timeout = model_timeout,
            max_retries = model_max_retries
        )
    # 2. Anthropic (Claude):-
    elif model_provider == 'anthropic':
        try:
            #pip install langchain-anthropic
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Could not import langchain-anthropic. "
                              "Please install it with: pip install langchain-anthropic")
        return ChatAnthropic(
            model=model_name,
            api_key=model_api_key,
            anthropic_api_url = base_url,

            temperature = model_temperature,
            max_tokens = model_max_tokens,
            timeout = model_timeout,
            max_retries = model_max_retries
        )
    # 3. Google Gemini:-
    elif model_provider == 'google':
        try:
            # pip install langchain-google-genai
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Could not import langchain-google-genai. "
                              "Please install it with: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=model_api_key,

            temperature=model_temperature,
            top_p=model_top_p,
            max_output_tokens=model_max_tokens,
        )
    # 4. Mistral:-
    elif model_provider == 'mistral':
        try:
            #pip install langchain-mistralai
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise ImportError("Could not import langchain-mistralai. "
                              "Please install it with: pip install langchain-mistralai")
        return ChatMistralAI(
            model=model_name,
            api_key=model_api_key,
            endpoint=base_url,

            temperature=model_temperature,
            top_p=model_top_p,
            max_tokens=model_max_tokens,
            timeout=model_timeout,
            max_retries=model_max_retries
        )
    # 5. Cohere:-
    elif model_provider == "cohere":
        try:
            #pip install langchain-cohere
            from langchain_cohere import ChatCohere
        except ImportError:
            raise ImportError("Could not import langchain-cohere. "
                              "Please install it with: pip install langchain-cohere")
        return ChatCohere(
            model=model_name,
            cohere_api_key=model_api_key,
            base_url = base_url,

            temperature=model_temperature,
            max_tokens=model_max_tokens,
        )

    else:
        raise ValueError(f"Unsupported provider: {model_provider}. May needs setup!")



# LLM Final Reasoning: --->
from langchain_core.messages import HumanMessage, SystemMessage
class PlanRequestClass(BaseModel):  # Payload for POST. Strict Type checked with Pydentic!
    lat: float  # BD lat == 23.7104
    lon: float  # BD lon == 90.40744
    location_name: str | None = None

@app.post("/plan")
async def generate_plan(req: PlanRequestClass):
    # fetching weather (my backend api call)
    weather = await get_weather(req.lat, req.lon)

    # fetching news [Default Country: BD] (my backend api call)
    news = await get_news(country="bd")
    headlines = [a.get("title") for a in news.get("articles", [])[:5]]  # Safe extraction of the dict.get() value with default value []

    # Prompt & Message for the RAG system (Ai-model):
    prompt = (
        f"User is at {req.location_name or f'{req.lat},{req.lon}'}. "
        f"Weather: {weather.get('weather')[0].get('description')}, temp {weather.get('main').get('temp')}°C. "
        f"Top headlines: {headlines}. "
        "Generate a concise daily plan (3-6 items) and practical recommendations (carry items, suggest reschedule if needed)."
    )
    # message = [
    #     {"role": "system",
    #      "content": "You are DayMate, a helpful daily planner."},
    #     {"role": "user",
    #      "content": prompt}
    # ]
    message = [
        SystemMessage(content="You are DayMate, a helpful daily planner."),
        HumanMessage(content=prompt)
    ]
    # print("Prompt is ===>\n",prompt)

    # Calling AI Model:--
    if instance_groq_model.api_key:
        print("\nLLM key is Found. Prompting with LLM...\n")
        llm = load_llm( #instance of the object Class OpenChatAi returned from load_llm() function
            model_provider=instance_groq_model.provider_name,
            model_name=instance_groq_model.model_name,
            model_api_key=instance_groq_model.api_key,
            base_url=instance_groq_model.base_url
        )
        # llm2 = load_llm(    #another instance for loading a smaller model for inference.
        #     model_name="llama-3.1-8b-instant",
        #     base_url = "https://api.groq.com/openai/v1",
        #     model_api_key = os.getenv("GROQ_API_KEY")
        # )
        response = llm.invoke(message)  # made the message with the prompt above!
        response_text = response.content
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
    result = asyncio.run(generate_plan(instance_payload))
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