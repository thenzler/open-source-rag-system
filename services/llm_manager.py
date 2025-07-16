#!/usr/bin/env python3
"""
LLM Manager Service
Handles dynamic model switching and configuration
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages LLM configurations and model switching
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Look for config in multiple locations
            possible_paths = [
                Path("config/llm_config.yaml"),
                Path("llm_config.yaml"),
                Path(__file__).parent.parent / "config" / "llm_config.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                logger.warning("No LLM config file found, using defaults")
                self.config = self._get_default_config()
                return
        
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model = self.config.get('default_model', 'mistral:latest')
        
        logger.info(f"LLM Manager initialized with model: {self.current_model}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded LLM config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found"""
        return {
            'default_model': 'mistral:latest',
            'models': {
                'mistral': {
                    'name': 'mistral:latest',
                    'description': 'Default model',
                    'context_length': 32768,
                    'temperature': 0.4,
                    'max_tokens': 2048,
                    'prompt_template': 'default'
                }
            },
            'prompt_templates': {
                'default': """Beantworte die Frage basierend AUSSCHLIESSLICH auf den bereitgestellten Dokumenten.

FRAGE: {query}

DOKUMENTE:
{context}

ANTWORT:"""
            }
        }
    
    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.current_model
    
    def set_model(self, model_key: str) -> bool:
        """
        Switch to a different model
        
        Args:
            model_key: Key from the models configuration
            
        Returns:
            bool: True if successful
        """
        if model_key in self.config.get('models', {}):
            model_config = self.config['models'][model_key]
            self.current_model = model_config['name']
            logger.info(f"Switched to model: {self.current_model}")
            return True
        else:
            logger.error(f"Model key '{model_key}' not found in configuration")
            return False
    
    def get_model_config(self, model_key: str = None) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if model_key is None:
            # Find current model in config
            for key, config in self.config.get('models', {}).items():
                if config['name'] == self.current_model:
                    return config
            # Fallback
            return self.config['models'].get('mistral', {})
        
        return self.config.get('models', {}).get(model_key, {})
    
    def get_prompt_template(self, query: str, context: str) -> str:
        """
        Get formatted prompt for current model
        
        Args:
            query: User query
            context: Document context
            
        Returns:
            str: Formatted prompt
        """
        # Get current model config
        model_config = self.get_model_config()
        template_name = model_config.get('prompt_template', 'default')
        
        # Get template
        templates = self.config.get('prompt_templates', {})
        template = templates.get(template_name, templates.get('default', ''))
        
        # Format template
        return template.format(query=query, context=context)
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get generation parameters for current model"""
        config = self.get_model_config()
        return {
            'temperature': config.get('temperature', 0.4),
            'max_tokens': config.get('max_tokens', 2048),
            'context_length': config.get('context_length', 8192)
        }
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available models with descriptions"""
        models = {}
        for key, config in self.config.get('models', {}).items():
            models[key] = {
                'name': config['name'],
                'description': config.get('description', 'No description'),
                'current': config['name'] == self.current_model
            }
        return models
    
    def reload_config(self):
        """Reload configuration from file"""
        if hasattr(self, 'config_path') and self.config_path:
            self.config = self._load_config()
            logger.info("Reloaded LLM configuration")
    
    def get_context_limit(self) -> int:
        """Get context length limit for current model"""
        config = self.get_model_config()
        return config.get('context_length', 4096)

# Global instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get or create LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager