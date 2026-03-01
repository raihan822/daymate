import os
from dotenv import load_dotenv  # Loads .env file for local testing:
load_dotenv()   #loads the api keys stored in the .env file and then acts like os.getenv() function as like os env ver

# ENTRY YOUR AI MODEL SUBSCRIPTIONS HERE... ==>
from .llm_schema import AiModelClass
MODELS_TO_LOAD: dict[str, AiModelClass]= {
    'groq_model' : AiModelClass(
        model_name="llama-3.3-70b-versatile",
        model_api_key=os.getenv("GROQ_API_KEY"),  # Pydentic will auto convert this str into SecretStr! You can do --> `raw_key = instance_groq_model.api_key.get_secret_value()` to read/get actual api_key value.
        model_provider="openai",
        base_url="https://api.groq.com/openai/v1"
    )

}


# GET MODEL OBJECTS ready-to-infer with!:
from .llm_model import get_model_obj
load_llm_objects = get_model_obj(MODELS_TO_LOAD)    # Will receive a dict with ModelNames AS Key and value AS LLM_Object to .invoke

""" ----------  RAG SYSTEM PROMPTING Msg ----------  """
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
def prompt_for_model(prompt: str, system_instruction: str)-> list[BaseMessage]:
    # OldStyle:
    # message = [
    #     {"role": "system",
    #      "content": "You are DayMate, a helpful daily planner."},
    #     {"role": "user",
    #      "content": prompt}
    # ]

    #NewStyle:
    message = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=prompt)
    ]
    return message