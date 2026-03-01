import os
# Loading .env file for local testing:
from dotenv import load_dotenv
load_dotenv()   #loads the api keys stored in the .env file and then acts like os.getenv() function as like os env ver

# Feature APIs: (took Free Subscription)
OPENWEATHER = {
    'api_key' : os.getenv("OPENWEATHER_KEY"),
    'base_url': "https://api.openweathermap.org/data/2.5/weather"     #GET https://api.openweathermap.org/data/2.5/weather ?lat={lat}&lon={lon}&appid={API key}
}
GNEWS = {
    'api_key' : os.getenv("GNEWS_API_KEY"),
    'base_url' : "https://gnews.io/api/v4/top-headlines"
}


"""Best Practice. Always Do type checking with pydentic(BaseModel) for API call with PAYLOAD(s),
You can also combine the API_KEYs+PROVIDER_NAME+BASE_URL into pydentic(BaseModel) and SecrectStr etc
later API_KEY raw value can be caught with you_masked_apikey.get_secret_value()
"""
# Data Modeling with Pydentic
from pydantic import BaseModel  #for explicit type checking, Generally used with fastAPI, and other libraries requiring strict type formalities of the variables.
class PlanRequestClass(BaseModel):  # Payload for POST. Strict Type checked with Pydentic!
    lat: float  # BD lat == 23.7104
    lon: float  # BD lon == 90.40744
    location_name: str | None = None
