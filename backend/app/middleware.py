# backend/app/middleware.py
# to connect and accept frontend api requrests.
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
    # FastAPI will allows request from the below `origins` list:
    origins = [
        "http://localhost:5173",      # React default local port 3000
        "http://127.0.0.1:5173",
        "https://your-app.vercel.app" # Deployed React frontend URL of Vercel
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )