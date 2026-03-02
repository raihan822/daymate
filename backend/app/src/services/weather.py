#httpx, request etc. are used for API level call
#asyncio is used for async type function
import httpx    # better alternative of `requests` that I used with BS4, Sel
from fastapi import HTTPException  #for FastAPI

from ...config import OPENWEATHER
async def fetch_weather(lat: float, lon: float):
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


if __name__ == "__main__":
    # Local unit testing:
    import asyncio  #used to call the async type of functions
    lat, lon = 23.7104, 90.40744
    response = asyncio.run(
        fetch_weather(lat=lat, lon=lon)
    )
    print(f"Location set to: \n"
          f"lat={lat}, lon={lon}\n"
          f"temperature: {response.get('main').get('temp')}°C."
          f"Weather Description:\n"
          f"{response.get('weather')[0].get('description')}\n")