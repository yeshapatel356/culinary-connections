import os
import requests
#from dotenv import load_dotenv
import json
import kagglehub



# Download latest version
path = kagglehub.dataset_download("snehallokesh31096/recipe")

print("Path to dataset files:", path)