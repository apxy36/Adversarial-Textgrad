from huggingface_hub import login
import os
# from kaggle_secrets import UserSecretsClient
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
import transformers
login(HF_TOKEN)
from openai import OpenAI
client = OpenAI(
    # base_url = "https://api.lambdalabs.com/v1",
    # api_key=user_secrets.get_secret("LAMBDA_KEY"),
  # base_url= "https://api.hyperbolic.xyz/v1",
  # api_key=user_secrets.get_secret("HYPER_KEY3"),
    api_key =  OPENAI_KEY,
)