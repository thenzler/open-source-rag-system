# Open Source RAG System - Makefile
# Comprehensive build and management automation

.PHONY: help install dev prod test clean setup-env build deploy stop logs backup restore

# Variables
DOCKER_COMPOSE_FILE = docker-compose.yml
DOCKER_COMPOSE_DEV = docker-compose.dev.yml
DOCKER_COMPOSE_PROD = docker-compose.prod.yml
PROJECT_NAME = open-source-rag-system
PYTHON_VERSION = 3.11

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BLUE)Open Source RAG System - Management Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  1. make setup-env     # Set up environment"
	@echo "  2. make dev           # Start development environment"
	@echo "  3. make test          # Run tests"
	@echo ""

# Environment Setup
setup-env: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(RED)Please edit .env file with your settings before proceeding$(NC)"; \
	else \
		echo "$(GREEN).env file already exists$(NC)"; \
	fi
	@echo "$(GREEN)Checking Docker installation...$(NC)"
	@docker --version || (echo "$(RED)Docker not found. Please install Docker$(NC)" && exit 1)
	@docker-compose --version || (echo "$(RED)Docker Compose not found. Please install Docker Compose$(NC)" && exit 1)
	@echo "$(GREEN)Environment setup complete!$(NC)"

check-env: ## Check if .env file exists
	@if [ ! -f .env ]; then \
		echo "$(RED).env file not found. Run 'make setup-env' first$(NC)"; \
		exit 1; \
	fi

# Development Commands
dev: check-env ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(BLUE)Services available at:$(NC)"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Web Interface: http://localhost:3000"
	@echo "  Grafana: http://localhost:3001"
	@echo "  Prometheus: http://localhost:9090"

dev-build: check-env ## Build and start development environment
	@echo "$(GREEN)Building and starting development environment...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) up -d --build

dev-logs: ## Show development logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

dev-shell: ## Open shell in API gateway container
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway /bin/bash

# Production Commands
prod: check-env ## Start production environment
	@echo "$(GREEN)Starting production environment...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) -f $(DOCKER_COMPOSE_PROD) up -d
	@echo "$(GREEN)Production environment started!$(NC)"

prod-build: check-env ## Build and start production environment
	@echo "$(GREEN)Building and starting production environment...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) -f $(DOCKER_COMPOSE_PROD) up -d --build

prod-deploy: check-env ## Deploy to production with health checks
	@echo "$(GREEN)Deploying to production...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) -f $(DOCKER_COMPOSE_PROD) up -d --build
	@echo "$(YELLOW)Waiting for services to be healthy...$(NC)"
	@sleep 30
	@make health-check
	@echo "$(GREEN)Production deployment complete!$(NC)"

# Service Management
stop: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) down
	@echo "$(GREEN)All services stopped$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) restart
	@echo "$(GREEN)All services restarted$(NC)"

ps: ## Show running containers
	@docker-compose -f $(DOCKER_COMPOSE_FILE) ps

logs: ## Show logs for all services
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

logs-api: ## Show API Gateway logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f api-gateway

logs-processor: ## Show Document Processor logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f document-processor

logs-vector: ## Show Vector Engine logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f vector-engine

# Database Management
db-init: ## Initialize database with schema
	@echo "$(GREEN)Initializing database...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway python -m alembic upgrade head
	@echo "$(GREEN)Database initialized$(NC)"

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway python -m alembic upgrade head
	@echo "$(GREEN)Database migrations complete$(NC)"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(YELLOW)Resetting database...$(NC)"; \
		docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres psql -U raguser -d ragdb -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		make db-migrate; \
		echo "$(GREEN)Database reset complete$(NC)"; \
	else \
		echo "$(GREEN)Database reset cancelled$(NC)"; \
	fi

db-backup: ## Backup database
	@echo "$(GREEN)Creating database backup...$(NC)"
	@mkdir -p backups
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres pg_dump -U raguser ragdb > backups/ragdb_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created in backups/$(NC)"

db-restore: ## Restore database from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Please specify BACKUP_FILE: make db-restore BACKUP_FILE=backup.sql$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Restoring database from $(BACKUP_FILE)...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec -T postgres psql -U raguser ragdb < $(BACKUP_FILE)
	@echo "$(GREEN)Database restore complete$(NC)"

# Vector Database Management
vector-backup: ## Backup vector database
	@echo "$(GREEN)Creating vector database backup...$(NC)"
	@mkdir -p backups/qdrant
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec qdrant curl -X POST "http://localhost:6333/snapshots"
	@docker cp $(shell docker-compose -f $(DOCKER_COMPOSE_FILE) ps -q qdrant):/qdrant/snapshots backups/qdrant/snapshots_$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)Vector database backup created$(NC)"

vector-status: ## Check vector database status
	@echo "$(GREEN)Checking vector database status...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec qdrant curl -s "http://localhost:6333/health" | python -m json.tool

# Testing
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway pytest tests/ -v
	@echo "$(GREEN)Tests complete$(NC)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway pytest tests/integration/ -v

test-api: ## Run API tests only
	@echo "$(GREEN)Running API tests...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway pytest tests/api/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-load: ## Run load tests (requires locust)
	@echo "$(GREEN)Starting load tests...$(NC)"
	@pip install locust || echo "$(YELLOW)Installing locust...$(NC)" && pip install locust
	@locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Code Quality
lint: ## Run code linting
	@echo "$(GREEN)Running code linting...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway flake8 app/
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway black --check app/
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway isort --check-only app/

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway black app/
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway isort app/

type-check: ## Run type checking
	@echo "$(GREEN)Running type checking...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway mypy app/

security-scan: ## Run security scan
	@echo "$(GREEN)Running security scan...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway bandit -r app/
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway safety check

# Health Checks
health-check: ## Check system health
	@echo "$(GREEN)Checking system health...$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "$(RED)API Gateway health check failed$(NC)"
	@curl -s http://localhost:6333/health | python -m json.tool || echo "$(RED)Vector DB health check failed$(NC)"

status: ## Show system status
	@echo "$(GREEN)System Status:$(NC)"
	@echo "$(BLUE)Services:$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) ps
	@echo ""
	@echo "$(BLUE)Health Checks:$(NC)"
	@make health-check

# Data Management
upload-sample: ## Upload sample documents
	@echo "$(GREEN)Uploading sample documents...$(NC)"
	@curl -X POST "http://localhost:8000/api/v1/documents" \
		-H "Content-Type: multipart/form-data" \
		-F "file=@tests/fixtures/documents/sample.pdf" \
		-H "Authorization: Bearer $(shell make get-token)" || \
		echo "$(YELLOW)Please ensure you have a valid token and sample files$(NC)"

get-token: ## Get authentication token (for testing)
	@curl -s -X POST "http://localhost:8000/api/v1/auth/login" \
		-H "Content-Type: application/json" \
		-d '{"username": "testuser", "password": "testpass"}' | \
		python -c "import sys, json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null || \
		echo "demo_token"

query-test: ## Test query functionality
	@echo "$(GREEN)Testing query functionality...$(NC)"
	@curl -X POST "http://localhost:8000/api/v1/query" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $(shell make get-token)" \
		-d '{"query": "What is artificial intelligence?", "top_k": 3}' | \
		python -m json.tool || echo "$(YELLOW)Query test failed$(NC)"

# Monitoring
monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Opening monitoring dashboard...$(NC)"
	@echo "Grafana: http://localhost:3001 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

metrics: ## Show system metrics
	@echo "$(GREEN)Current system metrics:$(NC)"
	@curl -s http://localhost:8001/metrics | grep -E "(http_requests_total|response_time)" | head -10

# Clean up
clean: ## Clean up Docker resources
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) down -v --remove-orphans
	@docker system prune -f
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-data: ## Clean up data volumes (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(YELLOW)Cleaning up data volumes...$(NC)"; \
		docker-compose -f $(DOCKER_COMPOSE_FILE) down -v; \
		docker volume prune -f; \
		echo "$(GREEN)Data cleanup complete$(NC)"; \
	else \
		echo "$(GREEN)Data cleanup cancelled$(NC)"; \
	fi

clean-all: ## Clean everything (images, containers, volumes, networks)
	@echo "$(RED)WARNING: This will remove all Docker resources!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(YELLOW)Cleaning all Docker resources...$(NC)"; \
		docker-compose -f $(DOCKER_COMPOSE_FILE) down -v --remove-orphans; \
		docker system prune -a -f; \
		docker volume prune -f; \
		echo "$(GREEN)Complete cleanup finished$(NC)"; \
	else \
		echo "$(GREEN)Complete cleanup cancelled$(NC)"; \
	fi

# Documentation
docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway python -m mkdocs build
	@echo "$(GREEN)Documentation generated in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8080$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway python -m mkdocs serve -a 0.0.0.0:8080

# Development Utilities
shell-api: ## Open shell in API Gateway container
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec api-gateway /bin/bash

shell-db: ## Open PostgreSQL shell
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres psql -U raguser ragdb

shell-redis: ## Open Redis CLI
	@docker-compose -f $(DOCKER_COMPOSE_FILE) exec redis redis-cli

# Build and Release
build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) build
	@echo "$(GREEN)Build complete$(NC)"

build-no-cache: ## Build all Docker images without cache
	@echo "$(GREEN)Building Docker images without cache...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) build --no-cache
	@echo "$(GREEN)Build complete$(NC)"

push: ## Push Docker images to registry
	@echo "$(GREEN)Pushing Docker images...$(NC)"
	@docker-compose -f $(DOCKER_COMPOSE_FILE) push
	@echo "$(GREEN)Push complete$(NC)"

version: ## Show version information
	@echo "$(GREEN)Version Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(PYTHON_VERSION)"
	@docker --version
	@docker-compose --version

# Installation helpers
install-dev: ## Install development dependencies locally
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@pip install -r services/api-gateway/requirements.txt
	@pip install -r services/api-gateway/requirements-dev.txt
	@echo "$(GREEN)Development dependencies installed$(NC)"

pre-commit: ## Run pre-commit checks
	@echo "$(GREEN)Running pre-commit checks...$(NC)"
	@make lint
	@make type-check
	@make test-unit
	@echo "$(GREEN)Pre-commit checks complete$(NC)"

# Quick commands for common workflows
quick-start: setup-env dev db-init ## Quick start for new users
	@echo "$(GREEN)Quick start complete!$(NC)"
	@echo "$(BLUE)Your RAG system is ready at http://localhost:8000$(NC)"

reset-dev: stop clean dev db-init ## Reset development environment
	@echo "$(GREEN)Development environment reset complete$(NC)"

update: ## Update to latest version
	@echo "$(GREEN)Updating to latest version...$(NC)"
	@git pull origin main
	@make build
	@make db-migrate
	@echo "$(GREEN)Update complete$(NC)"

# CI/CD helpers
ci-test: ## Run CI test suite
	@echo "$(GREEN)Running CI test suite...$(NC)"
	@make build
	@make test
	@make security-scan
	@echo "$(GREEN)CI tests complete$(NC)"

ci-deploy: ## Deploy in CI environment
	@echo "$(GREEN)Deploying in CI environment...$(NC)"
	@make prod-build
	@make health-check
	@echo "$(GREEN)CI deployment complete$(NC)"
