from huggingface_hub import login
import os
import sys
from dotenv import load_dotenv

# Put this at the VERY TOP of your python script (before importing transformers)
# import os
import torch

load_dotenv()
# from kaggle_secrets import UserSecretsClient
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")
# import transformers
login(token = HF_TOKEN) 
from openai import OpenAI
client = OpenAI(
    # base_url = "https://api.lambdalabs.com/v1",
    # api_key=user_secrets.get_secret("LAMBDA_KEY"),
  # base_url= "https://api.hyperbolic.xyz/v1",
  # api_key=user_secrets.get_secret("HYPER_KEY3"),
    api_key =  OPENAI_KEY,
)