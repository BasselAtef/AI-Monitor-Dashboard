"""
Example client showing how to integrate the monitoring dashboard with your AI applications
"""

import requests
import time
import os
from dotenv import load_dotenv


from groq import Groq  # Example with Groq API

# Load api key from .env file

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


# Dashboard endpoint
MONITOR_URL = "http://localhost:5000/api/log"

def log_to_monitor(provider, model, prompt_tokens, completion_tokens, latency_ms, status="success", error=None):
    """Send metrics to the monitoring dashboard"""
    try:
        payload = {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "error_message": error
        }
        requests.post(MONITOR_URL, json=payload)
    except Exception as e:
        print(f"Failed to log to monitor: {e}")

# Example: Monitored Groq API call
def monitored_groq_call(prompt, api_key):
    """Make a Groq API call with monitoring"""
    client = Groq(api_key=api_key)
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log success
        log_to_monitor(
            provider="groq",
            model="llama-3.1-70b",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            status="success"
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log error
        log_to_monitor(
            provider="groq",
            model="llama-3.1-70b",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            status="error",
            error=str(e)
        )
        
        raise e

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = api_key
    
    try:
        result = monitored_groq_call("Explain quantum computing in simple terms", API_KEY)
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")
