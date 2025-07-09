#!/usr/bin/env python3
"""
Ollama Client for Local LLM Integration
Handles communication with Ollama API for answer generation
"""
import logging
import requests
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.1:8b",
                 timeout: int = 30):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL
            model: Model to use for generation
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.available = None
        
        logger.info(f"Initialized Ollama client: {base_url}, model: {model}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is running and available
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        if self.available is not None:
            return self.available
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if our model is available
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check if our target model exists
                if any(self.model in name for name in model_names):
                    self.available = True
                    logger.info(f"Ollama is available with model {self.model}")
                else:
                    logger.warning(f"Ollama is running but model {self.model} not found. Available models: {model_names}")
                    self.available = False
            else:
                self.available = False
                logger.warning(f"Ollama health check failed: {response.status_code}")
        except Exception as e:
            self.available = False
            logger.warning(f"Ollama not available: {e}")
        
        return self.available
    
    def list_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List[str]: List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model.get('name', '') for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    def pull_model(self, model: Optional[str] = None) -> bool:
        """
        Pull/download a model
        
        Args:
            model: Model name to pull (uses self.model if None)
        
        Returns:
            bool: True if successful, False otherwise
        """
        model_name = model or self.model
        
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {model_name}")
                self.available = None  # Reset availability check
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def generate_answer(self, 
                       query: str, 
                       context: str, 
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> Optional[str]:
        """
        Generate an answer using Ollama
        
        Args:
            query: User's question
            context: Relevant document context
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
        
        Returns:
            Optional[str]: Generated answer or None if failed
        """
        if not self.is_available():
            logger.warning("Ollama not available for answer generation")
            return None
        
        # Prepare the prompt
        prompt = self._create_rag_prompt(query, context)
        
        try:
            logger.info(f"Generating answer for query: '{query[:50]}...'")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nQuestion:"]
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                if answer:
                    logger.info(f"Generated answer ({len(answer)} chars)")
                    return answer
                else:
                    logger.warning("Empty response from Ollama")
                    return None
            else:
                logger.error(f"Ollama generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG-optimized prompt
        
        Args:
            query: User's question
            context: Document context
        
        Returns:
            str: Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant that answers questions based only on the provided context. Follow these rules:

1. Only use information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
3. Be concise and direct
4. Include specific details from the context when relevant
5. Do not make up or assume information not in the context

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> Optional[str]:
        """
        Generate completion using chat format
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        
        Returns:
            Optional[str]: Generated response or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            # Convert messages to prompt format
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"Human: {content}\n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
            
            prompt += "Assistant: "
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
        
        return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current model
        
        Returns:
            Optional[Dict]: Model information or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
        
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check
        
        Returns:
            Dict: Health status information
        """
        status = {
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
            "models": [],
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                status["models"] = model_names
                status["available"] = any(self.model in name for name in model_names)
                
                if not status["available"]:
                    status["error"] = f"Model {self.model} not found. Available: {model_names}"
            else:
                status["error"] = f"Ollama API returned {response.status_code}"
                
        except requests.ConnectionError:
            status["error"] = "Cannot connect to Ollama (is it running?)"
        except requests.Timeout:
            status["error"] = "Ollama request timed out"
        except Exception as e:
            status["error"] = f"Unexpected error: {str(e)}"
        
        return status

# Global instance
ollama_client = OllamaClient()

def get_ollama_client() -> OllamaClient:
    """Get the global Ollama client instance"""
    return ollama_client

def test_ollama_connection():
    """Test Ollama connection and print status"""
    client = get_ollama_client()
    health = client.health_check()
    
    print("Ollama Health Check:")
    print(f"Available: {health['available']}")
    print(f"Model: {health['model']}")
    print(f"Base URL: {health['base_url']}")
    print(f"Available Models: {health['models']}")
    
    if health['error']:
        print(f"Error: {health['error']}")
    
    return health['available']

if __name__ == "__main__":
    # Test the Ollama client
    test_ollama_connection()