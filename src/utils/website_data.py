import random
import requests
import openai
import os
from dotenv import load_dotenv
from src.utils.assets import prompt_example_options, prompt_phrasing_options, init_script
# File -> Settings -> Interpreter (Invalid?)

load_env()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

class DownloaderEnv:
    """
    Initialize a playwright browser and download website data into format suitable for later BrowserGym use
    """

def random_query_list(n_queries: int = 100) -> str:
    """
    gpt generates list of random internet queries using random prompt and example
    """
    client = openai.OpenAI()
    prompt_phrase = random.choice(prompt_phrasing_options)
    prompt_example = random.sample(prompt_example_options, 3)
    random_letter = random.choice('abcdefghijklmnopqrstuvwxyz')
    prompt = f"""
    {prompt_phrase}. Please provide a numbered list of {n_queries} examples. Example:
    1. {prompt_example[0]}
    2. {prompt_example[1]}
    3. {prompt_example[2]}
    Please avoid using the letter {random_letter} in your examples and be specific.
    """
    print(prompt)
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.99,
    )
    return response.choices[0].message.content

def fetch_sites_from_google(query: str) -> str:
    """
    Fetch random site url from Bing search results for a given query
    """
    api_key = os.getenv("BING_API_KEY")
    cx = os.getenv("BING_CX")
    url = f'url for bing search api'
    response = requests.get(url)
    return requests.choice([item['link'] for item in response.json().get('items', [])])

def get_web_data(delay: int = 0) -> None:
    """
    Generate dataset of JSON files with observation dictionaries
    """