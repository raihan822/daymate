""" -----   LOADING A MODEL (as Object) WITH RECEIVED CONFIGURATIONS -----   """

# LLM Integration Function: Generalised Ai-Model Calling
''' There are different types of LLM API Calling Style
Layer 1 → Raw provider SDK (different company has different native style of calling)
Layer 2 → API abstraction (LiteLLM)
Layer 3 → Framework abstraction (LangChain)
    > I am using LangChain's only the OpenAi format schema generalization 'langchain_openai'.
'''
from langchain_core.language_models import BaseChatModel #[Optional] for return type check.
from typing import Optional, Union
from pydantic import SecretStr
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
    # from langchain_core.utils import convert_to_secret_str    #now you can use `convert_to_secret_str()` function to get raw value of api_key

    if not model_api_key.get_secret_value():
        raise ValueError(f"API Key not found. {model_api_key}")
    raw_api_key = model_api_key.get_secret_value() if isinstance(model_api_key, SecretStr) else model_api_key

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
            api_key = raw_api_key,
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
            api_key=raw_api_key,
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
            google_api_key=raw_api_key,

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
            api_key=raw_api_key,
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
            cohere_api_key=raw_api_key,
            base_url = base_url,

            temperature=model_temperature,
            max_tokens=model_max_tokens,
        )

    else:
        raise ValueError(f"Unsupported provider: {model_provider}. May needs setup!")




# GET Model OBJECTS LOADED with load_llm() to get ready-to-infer MODEL OBJECTS!:
# from langchain_core.language_models import BaseChatModel #[Optional] for return type check.
# from .llm_model import load_llm
from .llm_schema import AiModelClass
def get_model_obj(models_to_load: dict[str, AiModelClass])-> BaseChatModel:
    load_llm_objects = {}
    for key_model_name, val_config_instance in models_to_load.items():
        if val_config_instance.model_api_key:
            try:
                load_llm_objects[key_model_name] = load_llm(
                    # instance of the object Class OpenChatAi returned from load_llm() function
                    model_provider=val_config_instance.model_provider,
                    model_name=val_config_instance.model_name,
                    model_api_key=val_config_instance.model_api_key,
                    base_url=val_config_instance.base_url,

                    # OTHER General Configuration Settings:-
                    model_top_p=val_config_instance.model_top_p,
                    model_temperature=val_config_instance.model_temperature,
                    model_max_tokens=val_config_instance.model_max_tokens,
                    # Other Configs:
                    model_timeout=val_config_instance.model_timeout,
                    model_max_retries=val_config_instance.model_max_retries
                )
            except Exception as e:
                print(f"Failed to load {key_model_name}: {e}")
                load_llm_objects[key_model_name] = None
        else:
            print(f"API KEY not provided for {key_model_name}")
            load_llm_objects[key_model_name] = None
    return load_llm_objects