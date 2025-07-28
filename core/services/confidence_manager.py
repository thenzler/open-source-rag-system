"""
Configurable Confidence Manager
Handles intelligent multi-tier confidence system with full customization
"""
import logging
import re
import yaml
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfidenceManager:
    """Manages configurable confidence system for RAG queries"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize confidence manager with configuration
        
        Args:
            config_path: Path to confidence config YAML file
        """
        self.config_path = config_path or "config/confidence_config.yaml"
        self.config = self._load_config()
        self.active_profile = None
        
        logger.info(f"Confidence Manager initialized (enabled: {self.config.get('enabled', True)})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded confidence configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading confidence config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is unavailable"""
        return {
            "enabled": True,
            "simple_threshold": 0.4,
            "base_thresholds": {
                "high_threshold": 0.65,
                "medium_threshold": 0.40,
                "low_threshold": 0.25,
                "refusal_threshold": 0.20
            },
            "query_analysis": {"enabled": True},
            "external_knowledge": {"enabled": True},
            "response_validation": {"enabled": True}
        }
    
    def set_profile(self, profile_name: str) -> bool:
        """
        Set active configuration profile
        
        Args:
            profile_name: Name of profile (strict, permissive, simple)
            
        Returns:
            bool: True if profile was set successfully
        """
        try:
            profiles = self.config.get("profiles", {})
            if profile_name not in profiles:
                logger.error(f"Profile '{profile_name}' not found in configuration")
                return False
            
            profile_config = profiles[profile_name]
            
            # Apply profile settings to main config
            for key, value in profile_config.items():
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            
            self.active_profile = profile_name
            logger.info(f"Applied configuration profile: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting profile {profile_name}: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if intelligent confidence system is enabled"""
        return self.config.get("enabled", True)
    
    def get_simple_threshold(self) -> float:
        """Get simple threshold for when intelligent system is disabled"""
        return self.config.get("simple_threshold", 0.4)
    
    def determine_confidence_tiers(self, query: str, use_llm: bool = True) -> Dict[str, float]:
        """
        Determine adaptive confidence thresholds based on query characteristics
        
        Args:
            query: User query to analyze
            use_llm: Whether LLM will be used
            
        Returns:
            Dict with confidence tier thresholds
        """
        try:
            if not self.is_enabled():
                # Simple mode - single threshold
                threshold = self.get_simple_threshold()
                return {
                    "high_threshold": threshold,
                    "medium_threshold": threshold,
                    "low_threshold": threshold,
                    "refusal_threshold": threshold * 0.5
                }
            
            # Get base thresholds
            base_thresholds = self.config.get("base_thresholds", {})
            tiers = {
                "high_threshold": base_thresholds.get("high_threshold", 0.65),
                "medium_threshold": base_thresholds.get("medium_threshold", 0.40),
                "low_threshold": base_thresholds.get("low_threshold", 0.25),
                "refusal_threshold": base_thresholds.get("refusal_threshold", 0.20)
            }
            
            # Apply intelligent adjustments if enabled
            if self.config.get("query_analysis", {}).get("enabled", True):
                adjustment = self._analyze_query_for_adjustments(query)
                
                # Apply adjustment to all tiers
                for tier in tiers:
                    tiers[tier] = max(0.05, tiers[tier] + adjustment)  # Minimum 0.05
            
            return tiers
            
        except Exception as e:
            logger.error(f"Error determining confidence tiers: {e}")
            # Fallback to safe defaults
            return {
                "high_threshold": 0.6,
                "medium_threshold": 0.4,
                "low_threshold": 0.25,
                "refusal_threshold": 0.15
            }
    
    def _analyze_query_for_adjustments(self, query: str) -> float:
        """
        Analyze query characteristics to determine threshold adjustments
        
        Args:
            query: User query text
            
        Returns:
            float: Adjustment value (positive = stricter, negative = more permissive)
        """
        try:
            query_lower = query.lower()
            total_adjustment = 0.0
            applied_categories = []
            
            query_analysis = self.config.get("query_analysis", {})
            
            # Check irrelevant topics first (highest priority)
            irrelevant_topics = query_analysis.get("irrelevant_topics", {})
            for category, settings in irrelevant_topics.items():
                if not settings.get("enabled", True):
                    continue
                    
                terms = settings.get("terms", [])
                if any(term in query_lower for term in terms):
                    adjustment = settings.get("adjustment", 0.1)
                    total_adjustment += adjustment
                    applied_categories.append(f"irrelevant_{category}")
                    logger.debug(f"Applied irrelevant topic adjustment: {category} (+{adjustment})")
                    break  # Only apply one irrelevant category
            
            # Check domain-specific terms (high priority)
            if not applied_categories:  # Only if no irrelevant topics found
                domain_terms = query_analysis.get("domain_terms", {})
                best_match = None
                best_priority = 0
                
                for domain, settings in domain_terms.items():
                    if not settings.get("enabled", True):
                        continue
                        
                    terms = settings.get("terms", [])
                    priority = settings.get("priority", 1)
                    
                    if any(term in query_lower for term in terms):
                        if priority > best_priority:
                            best_match = domain
                            best_priority = priority
                
                if best_match:
                    settings = domain_terms[best_match]
                    adjustment = settings.get("adjustment", 0.0)
                    total_adjustment += adjustment
                    applied_categories.append(f"domain_{best_match}")
                    logger.debug(f"Applied domain adjustment: {best_match} ({adjustment})")
            
            # Check factual indicators (medium priority)
            if not applied_categories:  # Only if no domain match
                factual = query_analysis.get("factual_indicators", {})
                if factual.get("enabled", True):
                    terms = factual.get("terms", [])
                    if any(term in query_lower for term in terms):
                        adjustment = factual.get("adjustment", -0.03)
                        total_adjustment += adjustment
                        applied_categories.append("factual")
                        logger.debug(f"Applied factual query adjustment: {adjustment}")
            
            # Check vague queries (always check, additive)
            vague = query_analysis.get("vague_indicators", {})
            if vague.get("enabled", True):
                terms = vague.get("terms", [])
                if any(term in query_lower for term in terms):
                    adjustment = vague.get("adjustment", 0.05)
                    total_adjustment += adjustment
                    applied_categories.append("vague")
                    logger.debug(f"Applied vague query adjustment: +{adjustment}")
            
            if applied_categories and self.config.get("logging", {}).get("log_threshold_adjustments", True):
                logger.info(f"Query analysis applied: {', '.join(applied_categories)} (total adjustment: {total_adjustment:+.3f})")
            
            return total_adjustment
            
        except Exception as e:
            logger.error(f"Error analyzing query for adjustments: {e}")
            return 0.0
    
    def detect_external_knowledge(self, query: str) -> Tuple[bool, str]:
        """
        Detect if query requires external knowledge
        
        Args:
            query: User query to analyze
            
        Returns:
            Tuple[bool, str]: (requires_external_knowledge, reason)
        """
        try:
            external_config = self.config.get("external_knowledge", {})
            if not external_config.get("enabled", True):
                return False, ""
            
            query_lower = query.lower()
            
            # Check each category of external knowledge patterns
            for category, settings in external_config.items():
                if category == "enabled" or not isinstance(settings, dict):
                    continue
                    
                if not settings.get("enabled", True):
                    continue
                
                patterns = settings.get("patterns", [])
                reason = settings.get("reason", "externes Wissen")
                category_name = settings.get("category", category)
                
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        logger.info(f"External knowledge detected: {reason} in category {category_name}")
                        return True, f"{reason} ({category_name})"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error detecting external knowledge: {e}")
            return False, ""
    
    def should_validate_responses(self) -> bool:
        """Check if response validation is enabled"""
        return self.config.get("response_validation", {}).get("enabled", True)
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get response validation settings"""
        return self.config.get("response_validation", {
            "content_similarity_threshold": 0.3,
            "require_source_citations": True,
            "max_response_length": 2000
        })
    
    def enable_domain(self, domain_name: str) -> bool:
        """
        Enable a specific domain in the configuration
        
        Args:
            domain_name: Name of domain to enable
            
        Returns:
            bool: True if domain was enabled successfully
        """
        try:
            domain_terms = self.config.get("query_analysis", {}).get("domain_terms", {})
            if domain_name in domain_terms:
                domain_terms[domain_name]["enabled"] = True
                logger.info(f"Enabled domain: {domain_name}")
                return True
            else:
                logger.warning(f"Domain '{domain_name}' not found in configuration")
                return False
                
        except Exception as e:
            logger.error(f"Error enabling domain {domain_name}: {e}")
            return False
    
    def disable_domain(self, domain_name: str) -> bool:
        """
        Disable a specific domain in the configuration
        
        Args:
            domain_name: Name of domain to disable
            
        Returns:
            bool: True if domain was disabled successfully
        """
        try:
            domain_terms = self.config.get("query_analysis", {}).get("domain_terms", {})
            if domain_name in domain_terms:
                domain_terms[domain_name]["enabled"] = False
                logger.info(f"Disabled domain: {domain_name}")
                return True
            else:
                logger.warning(f"Domain '{domain_name}' not found in configuration")
                return False
                
        except Exception as e:
            logger.error(f"Error disabling domain {domain_name}: {e}")
            return False
    
    def add_custom_domain(self, domain_name: str, terms: List[str], adjustment: float = -0.05, priority: int = 3) -> bool:
        """
        Add a custom domain to the configuration
        
        Args:
            domain_name: Name of the new domain
            terms: List of terms that identify this domain
            adjustment: Threshold adjustment for this domain
            priority: Priority level (higher = more important)
            
        Returns:
            bool: True if domain was added successfully
        """
        try:
            if "query_analysis" not in self.config:
                self.config["query_analysis"] = {}
            if "domain_terms" not in self.config["query_analysis"]:
                self.config["query_analysis"]["domain_terms"] = {}
            
            self.config["query_analysis"]["domain_terms"][domain_name] = {
                "terms": terms,
                "adjustment": adjustment,
                "priority": priority,
                "enabled": True
            }
            
            logger.info(f"Added custom domain: {domain_name} with {len(terms)} terms")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom domain {domain_name}: {e}")
            return False
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of current confidence manager status"""
        try:
            enabled_domains = []
            disabled_domains = []
            
            domain_terms = self.config.get("query_analysis", {}).get("domain_terms", {})
            for domain, settings in domain_terms.items():
                if settings.get("enabled", True):
                    enabled_domains.append(domain)
                else:
                    disabled_domains.append(domain)
            
            enabled_external = []
            disabled_external = []
            
            external_config = self.config.get("external_knowledge", {})
            for category, settings in external_config.items():
                if category == "enabled" or not isinstance(settings, dict):
                    continue
                if settings.get("enabled", True):
                    enabled_external.append(category)
                else:
                    disabled_external.append(category)
            
            return {
                "intelligent_system_enabled": self.is_enabled(),
                "active_profile": self.active_profile,
                "simple_threshold": self.get_simple_threshold(),
                "base_thresholds": self.config.get("base_thresholds", {}),
                "query_analysis_enabled": self.config.get("query_analysis", {}).get("enabled", True),
                "enabled_domains": enabled_domains,
                "disabled_domains": disabled_domains,
                "external_knowledge_enabled": self.config.get("external_knowledge", {}).get("enabled", True),
                "enabled_external_categories": enabled_external,
                "disabled_external_categories": disabled_external,
                "response_validation_enabled": self.should_validate_responses()
            }
            
        except Exception as e:
            logger.error(f"Error getting status summary: {e}")
            return {"error": str(e)}