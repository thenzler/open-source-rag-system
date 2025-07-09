#!/usr/bin/env python3
"""
Ollama Client for Local LLM Integration
Handles communication with Ollama API for answer generation
"""
import logging
import requests
import json
import time
from typing import Optional, Dict, List, Any, Generator
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = None,
                 timeout: int = 30):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL
            model: Model to use for generation (auto-detect if None)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.available = None
        
        # Auto-detect available model if not specified
        if model is None:
            self.model = self._auto_detect_model()
        else:
            self.model = model
        
        logger.info(f"Initialized Ollama client: {base_url}, model: {self.model}")
    
    def _auto_detect_model(self) -> str:
        """
        Auto-detect the best available model
        
        Returns:
            str: Best available model name
        """
        try:
            # Preferred models in order of preference
            preferred_models = [
                "llama3.1:8b",
                "llama3:8b", 
                "llama2:13b",
                "llama2:7b",
                "mistral:latest",
                "mistral:7b",
                "phi3:latest",
                "phi3-mini:latest",
                "mannix/phi3-mini-4k:latest"
            ]
            
            # Get available models
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                # Find the first preferred model that's available
                for preferred in preferred_models:
                    if preferred in available_models:
                        logger.info(f"Auto-detected model: {preferred}")
                        return preferred
                
                # If no preferred model found, use the first available
                if available_models:
                    model = available_models[0]
                    logger.info(f"Using first available model: {model}")
                    return model
                else:
                    logger.warning("No models available")
                    return "llama3.1:8b"  # fallback
            else:
                logger.warning(f"Failed to get models list: {response.status_code}")
                return "llama3.1:8b"  # fallback
                
        except Exception as e:
            logger.warning(f"Error auto-detecting model: {e}")
            return "llama3.1:8b"  # fallback
    
    def is_available(self) -> bool:
        """
        Check if Ollama is running and available with improved error handling
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        # Reset cached availability periodically (every 30 seconds)
        import time
        current_time = time.time()
        if hasattr(self, '_last_check_time') and self._last_check_time:
            if current_time - self._last_check_time < 30 and self.available is not None:
                return self.available
        
        self._last_check_time = current_time
        
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
                    # Try to find similar models
                    similar_models = [name for name in model_names if self.model.split(':')[0] in name]
                    if similar_models:
                        logger.warning(f"Exact model {self.model} not found, but similar models available: {similar_models}")
                    else:
                        logger.warning(f"Ollama is running but model {self.model} not found. Available models: {model_names}")
                    self.available = False
            else:
                self.available = False
                logger.warning(f"Ollama health check failed with status {response.status_code}: {response.text}")
        except requests.ConnectionError:
            self.available = False
            logger.warning("Cannot connect to Ollama - is it running?")
        except requests.Timeout:
            self.available = False
            logger.warning("Ollama connection timed out")
        except Exception as e:
            self.available = False
            logger.warning(f"Ollama availability check failed: {e}")
        
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
                       temperature: float = 0.7,
                       max_retries: int = 3) -> Optional[str]:
        """
        Generate an answer using Ollama with improved error handling and retry logic
        
        Args:
            query: User's question
            context: Relevant document context
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
            max_retries: Maximum number of retry attempts
        
        Returns:
            Optional[str]: Generated answer or None if failed
        """
        if not self.is_available():
            logger.warning("Ollama not available for answer generation")
            return None
        
        # Prepare the prompt
        prompt = self._create_rag_prompt(query, context)
        
        # Validate inputs
        if not prompt or not prompt.strip():
            logger.error("Invalid prompt for answer generation")
            return None
        
        if len(prompt) > 32000:  # Reasonable limit for prompt length
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating")
            prompt = prompt[:32000] + "..."
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating answer for query: '{query[:50]}...' (attempt {attempt + 1}/{max_retries})")
                
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
                        logger.info(f"Generated answer ({len(answer)} chars) on attempt {attempt + 1}")
                        return answer
                    else:
                        logger.warning(f"Empty response from Ollama on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)  # Brief pause before retry
                            continue
                        return None
                        
                elif response.status_code == 404:
                    logger.error(f"Model {self.model} not found on Ollama server")
                    self.available = False  # Reset availability
                    return None
                    
                elif response.status_code == 503:
                    logger.warning(f"Ollama server busy (503) on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)  # Longer pause for server busy
                        continue
                    return None
                    
                else:
                    logger.error(f"Ollama generation failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
                        continue
                    return None
                    
            except requests.ConnectionError:
                logger.error(f"Connection error on attempt {attempt + 1}")
                self.available = False  # Reset availability
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                return None
                
            except requests.Timeout:
                logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Error generating answer on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                return None
        
        logger.error(f"Failed to generate answer after {max_retries} attempts")
        return None
    
    def generate_answer_stream(self, 
                              query: str, 
                              context: str, 
                              max_tokens: int = 2048,
                              temperature: float = 0.7):
        """
        Generate an answer using Ollama with streaming response
        
        Args:
            query: User's question
            context: Relevant document context
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
        
        Yields:
            str: Streaming response chunks
        """
        if not self.is_available():
            logger.warning("Ollama not available for answer generation")
            return
        
        # Prepare the prompt
        prompt = self._create_rag_prompt(query, context)
        
        # Validate inputs
        if not prompt or not prompt.strip():
            logger.error("Invalid prompt for answer generation")
            return
        
        if len(prompt) > 32000:  # Reasonable limit for prompt length
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating")
            prompt = prompt[:32000] + "..."
        
        try:
            logger.info(f"Generating streaming answer for query: '{query[:50]}...'")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nQuestion:"]
                },
                "stream": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Streaming request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
    
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
        Comprehensive health check with detailed diagnostics
        
        Returns:
            Dict: Health status information
        """
        status = {
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
            "models": [],
            "error": None,
            "timestamp": datetime.now().isoformat(),
            "diagnostics": {
                "connection_test": False,
                "model_exists": False,
                "generation_test": False,
                "response_time": None
            }
        }
        
        start_time = time.time()
        
        try:
            # Test 1: Basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            status["diagnostics"]["connection_test"] = True
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                status["models"] = model_names
                status["diagnostics"]["response_time"] = time.time() - start_time
                
                # Test 2: Model exists
                model_available = any(self.model in name for name in model_names)
                status["diagnostics"]["model_exists"] = model_available
                
                if model_available:
                    # Test 3: Basic generation test
                    try:
                        test_response = requests.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model,
                                "prompt": "Hello, respond with just 'OK'",
                                "options": {"num_predict": 5, "temperature": 0.1},
                                "stream": False
                            },
                            timeout=10
                        )
                        
                        if test_response.status_code == 200:
                            test_result = test_response.json()
                            if test_result.get('response', '').strip():
                                status["diagnostics"]["generation_test"] = True
                                status["available"] = True
                            else:
                                status["error"] = "Model available but generation test failed"
                        else:
                            status["error"] = f"Generation test failed with status {test_response.status_code}"
                    
                    except requests.Timeout:
                        status["error"] = "Generation test timed out"
                    except Exception as gen_e:
                        status["error"] = f"Generation test error: {str(gen_e)}"
                else:
                    # Find similar models
                    similar_models = [name for name in model_names if self.model.split(':')[0] in name]
                    if similar_models:
                        status["error"] = f"Model {self.model} not found. Similar models: {similar_models}"
                    else:
                        status["error"] = f"Model {self.model} not found. Available: {model_names}"
            else:
                status["error"] = f"Ollama API returned {response.status_code}: {response.text}"
                
        except requests.ConnectionError:
            status["error"] = "Cannot connect to Ollama (is it running?)"
        except requests.Timeout:
            status["error"] = "Ollama request timed out"
        except Exception as e:
            status["error"] = f"Unexpected error: {str(e)}"
        
        # Final availability check
        if not status["available"] and not status["error"]:
            status["error"] = "Health check failed for unknown reasons"
        
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