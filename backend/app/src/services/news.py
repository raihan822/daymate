import httpx    # better alternative of `requests` that I used with BS4, Sel
from fastapi import HTTPException  #for FastAPI

from backend.app.config import GNEWS
async def fetch_news(country: str = "bd", q: str | None = None):
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

if __name__ == "__main__":
    pass