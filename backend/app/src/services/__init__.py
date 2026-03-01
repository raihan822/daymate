"""
Without an __init__.py, your imports look like this (in main.py):
from src.services.weather import fetch_weather
from src.services.news import fetch_news

With a well-configured __init__.py, you can make them much shorter:
from src.services import *
And this will import directly all that I list here
"""
from .weather import fetch_weather
from .news import fetch_news
__all__ = ['fetch_weather', 'fetch_news']