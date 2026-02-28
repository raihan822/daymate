import os
from dotenv import load_dotenv
load_dotenv()
#
# AI_Model_API_KEY = os.getenv("GROQ_API_KEY")    # I got the GROQ Subscription Key!
# print(type(AI_Model_API_KEY))

from pydantic import BaseModel, SecretStr
class AiModel(BaseModel):
    api_key: SecretStr
    provider_name: str
    base_url: str
groq_model_instance = AiModel(
    api_key=os.getenv("GROQ_API_KEY"),  # I got the GROQ Subscription Key!
    provider_name="openai",
    base_url="https://api.groq.com/openai/v1"
)
print(f"key: {groq_model_instance.api_key}\n"
      f"name: {groq_model_instance.provider_name}\n"
      f"URL: {groq_model_instance.base_url}")