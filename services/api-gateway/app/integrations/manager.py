"""
Integration utilities for connecting RAG system with external platforms.
Includes API clients, webhooks, and data synchronization tools.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import httpx
from fastapi import HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field
import hmac
import hashlib
import base64
from urllib.parse import urljoin

from app.core.config import get_settings
from app.core.database import get_database
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

logger = logging.getLogger(__name__)
settings = get_settings()


class IntegrationType(Enum):
    """Types of integrations supported."""
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    SHAREPOINT = "sharepoint"
    GOOGLE_DRIVE = "google_drive"
    DROPBOX = "dropbox"
    JIRA = "jira"
    GITHUB = "github"
    GITLAB = "gitlab"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class IntegrationConfig:
    """Configuration for external integration."""
    integration_type: IntegrationType
    name: str
    enabled: bool
    credentials: Dict[str, str]
    settings: Dict[str, Any]
    webhook_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    auth_method: str = "bearer"
    rate_limit: int = 100
    timeout: int = 30
    retry_count: int = 3
    last_sync: Optional[datetime] = None


class WebhookEvent(BaseModel):
    """Webhook event model."""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None


class SlackIntegration:
    """Slack integration for notifications and queries."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.bot_token = config.credentials.get("bot_token")
        self.signing_secret = config.credentials.get("signing_secret")
        self.channel = config.settings.get("default_channel", "#general")
        
    async def send_message(self, message: str, channel: Optional[str] = None) -> bool:
        """Send message to Slack channel."""
        try:
            target_channel = channel or self.channel
            
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": target_channel,
                "text": message,
                "as_user": True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                
            if response.status_code == 200:
                result = response.json()
                return result.get("ok", False)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def send_query_result(self, query: str, results: List[Dict], channel: Optional[str] = None) -> bool:
        """Send query results to Slack as formatted message."""
        try:
            # Format results for Slack
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Query:* {query}\n*Results:* {len(results)} documents found"
                    }
                },
                {"type": "divider"}
            ]
            
            # Add top results
            for i, result in enumerate(results[:5]):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{i+1}. {result.get('title', 'Document')}*\n{result.get('content', '')[:200]}..."
                    },
                    "accessory": {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Full"
                        },
                        "action_id": f"view_document_{result.get('id')}"
                    }
                })
            
            target_channel = channel or self.channel
            
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": target_channel,
                "blocks": blocks,
                "as_user": True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                
            if response.status_code == 200:
                result = response.json()
                return result.get("ok", False)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Slack query result: {e}")
            return False
    
    def verify_webhook_signature(self, body: str, signature: str) -> bool:
        """Verify Slack webhook signature."""
        try:
            timestamp = signature.split("=")[1]
            basestring = f"v0:{timestamp}:{body}"
            
            expected_signature = hmac.new(
                self.signing_secret.encode(),
                basestring.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(f"v0={expected_signature}", signature)
            
        except Exception as e:
            logger.error(f"Failed to verify Slack signature: {e}")
            return False


class NotionIntegration:
    """Notion integration for document synchronization."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.token = config.credentials.get("integration_token")
        self.database_id = config.settings.get("database_id")
        self.base_url = "https://api.notion.com/v1"
        
    async def sync_documents(self) -> Dict[str, Any]:
        """Sync documents from Notion database."""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28"
            }
            
            url = f"{self.base_url}/databases/{self.database_id}/query"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers)
                
            if response.status_code == 200:
                data = response.json()
                pages = data.get("results", [])
                
                synced_docs = []
                for page in pages:
                    doc_data = await self._process_notion_page(page)
                    if doc_data:
                        synced_docs.append(doc_data)
                
                return {
                    "success": True,
                    "synced_documents": len(synced_docs),
                    "documents": synced_docs
                }
            
            return {"success": False, "error": f"API error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Failed to sync Notion documents: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_notion_page(self, page: Dict) -> Optional[Dict]:
        """Process a Notion page into document format."""
        try:
            properties = page.get("properties", {})
            
            # Extract title
            title_prop = properties.get("title") or properties.get("Title") or properties.get("Name")
            title = ""
            if title_prop and title_prop.get("title"):
                title = "".join([t.get("plain_text", "") for t in title_prop["title"]])
            
            # Extract content
            content = await self._get_page_content(page["id"])
            
            # Extract metadata
            metadata = {
                "notion_id": page["id"],
                "url": page["url"],
                "created_time": page["created_time"],
                "last_edited_time": page["last_edited_time"]
            }
            
            return {
                "title": title,
                "content": content,
                "metadata": metadata,
                "source": "notion"
            }
            
        except Exception as e:
            logger.error(f"Failed to process Notion page: {e}")
            return None
    
    async def _get_page_content(self, page_id: str) -> str:
        """Get content blocks from Notion page."""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28"
            }
            
            url = f"{self.base_url}/blocks/{page_id}/children"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                
            if response.status_code == 200:
                data = response.json()
                blocks = data.get("results", [])
                
                content_parts = []
                for block in blocks:
                    text = self._extract_text_from_block(block)
                    if text:
                        content_parts.append(text)
                
                return "\n".join(content_parts)
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to get Notion page content: {e}")
            return ""
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from Notion block."""
        try:
            block_type = block.get("type", "")
            
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
                rich_text = block.get(block_type, {}).get("rich_text", [])
                return "".join([t.get("plain_text", "") for t in rich_text])
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to extract text from block: {e}")
            return ""


class GitHubIntegration:
    """GitHub integration for code documentation."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.token = config.credentials.get("access_token")
        self.repo_owner = config.settings.get("repo_owner")
        self.repo_name = config.settings.get("repo_name")
        self.base_url = "https://api.github.com"
        
    async def sync_repository_docs(self) -> Dict[str, Any]:
        """Sync documentation from GitHub repository."""
        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get repository contents
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                
            if response.status_code == 200:
                contents = response.json()
                
                docs = []
                for item in contents:
                    if item["type"] == "file" and item["name"].endswith((".md", ".txt", ".rst")):
                        doc_data = await self._process_github_file(item, headers)
                        if doc_data:
                            docs.append(doc_data)
                
                return {
                    "success": True,
                    "synced_documents": len(docs),
                    "documents": docs
                }
            
            return {"success": False, "error": f"API error: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Failed to sync GitHub docs: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_github_file(self, file_item: Dict, headers: Dict) -> Optional[Dict]:
        """Process a GitHub file into document format."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(file_item["download_url"], headers=headers)
                
            if response.status_code == 200:
                content = response.text
                
                return {
                    "title": file_item["name"],
                    "content": content,
                    "metadata": {
                        "github_path": file_item["path"],
                        "github_url": file_item["html_url"],
                        "size": file_item["size"],
                        "sha": file_item["sha"]
                    },
                    "source": "github"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to process GitHub file: {e}")
            return None


class WebhookHandler:
    """Generic webhook handler for various integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self.document_service = DocumentService()
        self.query_service = QueryService()
        
    def register_integration(self, name: str, integration: Any):
        """Register an integration handler."""
        self.integrations[name] = integration
        
    async def handle_webhook(self, integration_name: str, event: WebhookEvent) -> Dict[str, Any]:
        """Handle incoming webhook event."""
        try:
            integration = self.integrations.get(integration_name)
            if not integration:
                raise HTTPException(status_code=404, detail="Integration not found")
            
            # Verify signature if present
            if hasattr(integration, 'verify_webhook_signature') and event.signature:
                if not integration.verify_webhook_signature(json.dumps(event.data), event.signature):
                    raise HTTPException(status_code=401, detail="Invalid signature")
            
            # Process event based on type
            if event.event_type == "message":
                return await self._handle_message_event(integration, event)
            elif event.event_type == "query":
                return await self._handle_query_event(integration, event)
            elif event.event_type == "document_sync":
                return await self._handle_document_sync_event(integration, event)
            else:
                return {"success": False, "error": "Unknown event type"}
            
        except Exception as e:
            logger.error(f"Failed to handle webhook: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_message_event(self, integration: Any, event: WebhookEvent) -> Dict[str, Any]:
        """Handle message event from integration."""
        try:
            message = event.data.get("message", "")
            
            # Check if message is a query
            if message.startswith("/query "):
                query = message[7:]  # Remove "/query " prefix
                
                # Execute query
                results = await self.query_service.query_documents(query=query, top_k=5)
                
                # Send results back
                if hasattr(integration, 'send_query_result'):
                    await integration.send_query_result(query, results.get("results", []))
                
                return {"success": True, "action": "query_executed"}
            
            return {"success": True, "action": "message_received"}
            
        except Exception as e:
            logger.error(f"Failed to handle message event: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_query_event(self, integration: Any, event: WebhookEvent) -> Dict[str, Any]:
        """Handle query event from integration."""
        try:
            query = event.data.get("query", "")
            
            if not query:
                return {"success": False, "error": "No query provided"}
            
            # Execute query
            results = await self.query_service.query_documents(query=query, top_k=10)
            
            # Format results for response
            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", "")[:500],
                    "score": result.get("score", 0),
                    "source": result.get("source", "")
                })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to handle query event: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_document_sync_event(self, integration: Any, event: WebhookEvent) -> Dict[str, Any]:
        """Handle document synchronization event."""
        try:
            if hasattr(integration, 'sync_documents'):
                sync_result = await integration.sync_documents()
                
                if sync_result.get("success"):
                    # Process synced documents
                    documents = sync_result.get("documents", [])
                    
                    processed_count = 0
                    for doc in documents:
                        try:
                            # Create document in system
                            await self.document_service.create_document_from_data(doc)
                            processed_count += 1
                        except Exception as e:
                            logger.error(f"Failed to process document: {e}")
                    
                    return {
                        "success": True,
                        "synced_documents": len(documents),
                        "processed_documents": processed_count
                    }
                else:
                    return sync_result
            
            return {"success": False, "error": "Integration does not support document sync"}
            
        except Exception as e:
            logger.error(f"Failed to handle document sync event: {e}")
            return {"success": False, "error": str(e)}


class IntegrationManager:
    """Manages all external integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self.webhook_handler = WebhookHandler()
        
    def register_integration(self, name: str, config: IntegrationConfig):
        """Register a new integration."""
        try:
            if config.integration_type == IntegrationType.SLACK:
                integration = SlackIntegration(config)
            elif config.integration_type == IntegrationType.NOTION:
                integration = NotionIntegration(config)
            elif config.integration_type == IntegrationType.GITHUB:
                integration = GitHubIntegration(config)
            else:
                raise ValueError(f"Unsupported integration type: {config.integration_type}")
            
            self.integrations[name] = integration
            self.webhook_handler.register_integration(name, integration)
            
            logger.info(f"Registered integration: {name}")
            
        except Exception as e:
            logger.error(f"Failed to register integration {name}: {e}")
            raise
    
    async def sync_all_integrations(self) -> Dict[str, Any]:
        """Sync all registered integrations."""
        results = {}
        
        for name, integration in self.integrations.items():
            try:
                if hasattr(integration, 'sync_documents'):
                    result = await integration.sync_documents()
                    results[name] = result
                else:
                    results[name] = {"success": True, "message": "No sync method available"}
                    
            except Exception as e:
                logger.error(f"Failed to sync integration {name}: {e}")
                results[name] = {"success": False, "error": str(e)}
        
        return results
    
    async def send_notification(self, message: str, integration_names: Optional[List[str]] = None):
        """Send notification to specified integrations."""
        targets = integration_names or list(self.integrations.keys())
        
        for name in targets:
            integration = self.integrations.get(name)
            if integration and hasattr(integration, 'send_message'):
                try:
                    await integration.send_message(message)
                except Exception as e:
                    logger.error(f"Failed to send notification to {name}: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        status = {}
        
        for name, integration in self.integrations.items():
            try:
                last_sync = getattr(integration.config, 'last_sync', None)
                status[name] = {
                    "type": integration.config.integration_type.value,
                    "enabled": integration.config.enabled,
                    "last_sync": last_sync.isoformat() if last_sync else None,
                    "health": "healthy"  # Could add health check method
                }
            except Exception as e:
                status[name] = {
                    "type": "unknown",
                    "enabled": False,
                    "last_sync": None,
                    "health": "unhealthy",
                    "error": str(e)
                }
        
        return status


# Global integration manager instance
integration_manager = IntegrationManager()


async def initialize_integrations():
    """Initialize all configured integrations."""
    try:
        # Load integration configurations from settings
        integration_configs = getattr(settings, 'integrations', {})
        
        for name, config_dict in integration_configs.items():
            config = IntegrationConfig(**config_dict)
            if config.enabled:
                integration_manager.register_integration(name, config)
        
        logger.info("All integrations initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize integrations: {e}")
        raise


async def cleanup_integrations():
    """Cleanup integration resources."""
    logger.info("Integration cleanup completed")


# API endpoints for integrations
from fastapi import APIRouter, Depends, BackgroundTasks

router = APIRouter(prefix="/api/v1/integrations", tags=["Integrations"])


@router.get("/")
async def list_integrations():
    """List all registered integrations."""
    return integration_manager.get_integration_status()


@router.post("/sync")
async def sync_integrations(background_tasks: BackgroundTasks):
    """Sync all integrations."""
    background_tasks.add_task(integration_manager.sync_all_integrations)
    return {"message": "Sync started"}


@router.post("/notify")
async def send_notification(
    message: str,
    integrations: Optional[List[str]] = None
):
    """Send notification to integrations."""
    await integration_manager.send_notification(message, integrations)
    return {"message": "Notification sent"}


@router.post("/webhook/{integration_name}")
async def handle_webhook(
    integration_name: str,
    event: WebhookEvent,
    background_tasks: BackgroundTasks
):
    """Handle incoming webhook."""
    result = await integration_manager.webhook_handler.handle_webhook(integration_name, event)
    return result
