from huggingface_hub import login
import os
import sys, random
from dotenv import load_dotenv
import numpy as np

# Put this at the VERY TOP of your python script (before importing transformers)
# import os
import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1234)
random.seed(1234)

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