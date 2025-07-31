import configparser
import os
from dotenv import load_dotenv

def get_api_key():
    """Get API key from either .env file or config.ini"""
    # First try environment variable
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # If not found in .env, try config.ini
    if not api_key:
        try:
            config = configparser.ConfigParser()
            config.read('config.ini')
            api_key = config.get('API', 'apikey')
        except:
            raise ValueError("API key not found in .env or config.ini")
    
    if not api_key:
        raise ValueError("API key not found")
        
    return api_key

def is_valid_api_key(api_key):
    """Check if the API key format is valid"""
    if not api_key:
        return False
    return api_key.startswith('sk-') or api_key.startswith('sk-proj-')
