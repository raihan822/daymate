""" -----   WHAT FEATURES A MODEL WILL HAVE WHEN MAKING INSTANCE -----   """

# Data Modeling with Pydentic
# What is Pydentic? :-
# Pydentic is for explicit type checking, Generally used with fastAPI, and other libraries requiring strict type formalities of the variables.
"""Best Practice. Always Do type checking with pydentic(BaseModel) for 
    > API call with PAYLOAD(s),
    > Combine the API_KEYs+PROVIDER_NAME+BASE_URL into pydentic(BaseModel) and SecrectStr etc. Like I did below.
    later API_KEY raw value can be caught with `apikey_variable.get_secret_value()`
"""
from typing import Optional
from pydantic import BaseModel, SecretStr
class AiModelClass(BaseModel):                  #Dictionary also doable. But pydantic Class making is better choice with `SecretStr`
    """Make an instance/multiple instance of this all-in-one class with all the models you want to use.
        - the model you want to use
        - Provider's name
        - base_url
        - Api_Key
    Each different model will have different Class Instance!
    """
    # REQUIRED INFO NEEDED:-
    model_name: str
    model_api_key: SecretStr
    model_provider: str
    #OPTIONAL URL:
    base_url: str | None = None

    # OTHER General Configuration Settings:-
    model_top_p: float = 1.0
    model_temperature: float = 0.7  # 0.7 is langchain default
    model_max_tokens: Optional[int] = None
    # Other Configs:
    model_timeout: int = 60
    model_max_retries: int = 2
