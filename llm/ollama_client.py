"""
Ollama Client - Local LLM Integration

What is Ollama?
- Ollama runs large language models (LLMs) locally on your computer
- No internet needed, no API costs, your data stays private
- We use llama3.2 model which runs entirely on your machine
- Ollama provides a simple HTTP API similar to OpenAI, but free and local

This module connects to Ollama running at http://localhost:11434
"""

import requests
from typing import Optional
import json


class OllamaClient:
    """
    Client for interacting with Ollama's local HTTP API.
    
    Ollama must be running on your machine. Start it with: ollama serve
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API URL (default: localhost)
            model: Model name (default: llama3.2)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api"
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text response from Ollama.
        
        This is used in RAG: we send retrieved chunks + user question,
        and Ollama generates an answer based on that context.
        
        Args:
            prompt: Full prompt including context and question
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Generated text response
        """
        # Check connection first
        if not self.check_connection():
            raise ConnectionError(
                "Ollama is not running or not accessible.\n\n"
                "To fix this:\n"
                "1. Open a new terminal/PowerShell window\n"
                "2. Run: ollama serve\n"
                "3. Wait for 'Ollama is running' message\n"
                "4. Refresh this page and try again"
            )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (400, 500, etc.)
            error_msg = f"Ollama server error: {e.response.status_code}"
            if e.response.status_code == 500:
                error_msg += "\n\nInternal server error. This usually means:\n"
                error_msg += "1. Ollama server needs to be restarted\n"
                error_msg += "2. The model might need to be reloaded\n"
                error_msg += "3. Try: Stop Ollama (Ctrl+C) and run 'ollama serve' again"
            elif e.response.status_code == 404:
                error_msg += f"\n\nModel '{self.model}' not found. Run: ollama pull {self.model}"
            try:
                error_detail = e.response.json().get('error', '')
                if error_detail:
                    error_msg += f"\n\nDetails: {error_detail}"
            except:
                pass
            raise ConnectionError(error_msg)
        except requests.exceptions.Timeout:
            raise ConnectionError(
                "Ollama request timed out. The model might be taking too long to respond.\n\n"
                "Try:\n"
                "1. Check if Ollama is still running\n"
                "2. Try a smaller prompt or simpler question\n"
                "3. Restart Ollama: ollama serve"
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server.\n\n"
                "To fix this:\n"
                "1. Open a new terminal/PowerShell window\n"
                "2. Run: ollama serve\n"
                "3. Wait for 'Ollama is running' message\n"
                "4. Refresh this page and try again"
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama: {e}\n\n"
                "Make sure Ollama is running: ollama serve"
            )
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and model is available.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                return any(self.model in name for name in model_names)
            return False
        except Exception:
            return False


if __name__ == "__main__":
    # Test connection
    client = OllamaClient()
    if client.check_connection():
        print(f"[OK] Ollama is running with model: {client.model}")
        test_response = client.generate("Say 'Hello, RAG!' in one sentence.")
        print(f"Test response: {test_response}")
    else:
        print("❌ Ollama is not running or model not found.")
        print("Start Ollama with: ollama serve")

