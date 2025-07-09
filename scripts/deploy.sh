#!/bin/bash

# Advanced Deployment Automation Script for RAG System
# Handles multi-environment deployment, health checks, and rollback

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENTS=("development" "staging" "production")
SERVICES=("api-gateway" "document-processor" "vector-engine" "web-interface")
BACKUP_DIR="$PROJECT_ROOT/backups"
LOGS_DIR="$PROJECT_ROOT/logs"
DEPLOYMENT_LOG="$LOGS_DIR/deployment.log"

# Default values
ENVIRONMENT="development"
SERVICE="all"
SKIP_TESTS=false
SKIP_BACKUP=false
FORCE_DEPLOY=false
ROLLBACK_VERSION=""
DRY_RUN=false
PARALLEL_DEPLOY=true
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_TIMEOUT=120

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" >&2
            echo "[$timestamp] [ERROR] $message" >> "$DEPLOYMENT_LOG"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            echo "[$timestamp] [WARN] $message" >> "$DEPLOYMENT_LOG"
            ;;
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message"
            echo "[$timestamp] [INFO] $message" >> "$DEPLOYMENT_LOG"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} $message"
            echo "[$timestamp] [DEBUG] $message" >> "$DEPLOYMENT_LOG"
            ;;
    esac
}

# Help function
show_help() {
    cat << EOF
Advanced RAG System Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV     Target environment (development|staging|production)
    -s, --service SERVICE     Service to deploy (all|api-gateway|document-processor|vector-engine|web-interface)
    -t, --skip-tests          Skip running tests before deployment
    -b, --skip-backup         Skip creating backup before deployment
    -f, --force               Force deployment even if health checks fail
    -r, --rollback VERSION    Rollback to specific version
    -d, --dry-run            Show what would be deployed without actually deploying
    -p, --no-parallel        Disable parallel deployment
    -h, --help               Show this help message

EXAMPLES:
    $0 -e production -s api-gateway
    $0 -e staging --skip-tests
    $0 --rollback v1.2.3
    $0 --dry-run -e production

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY          Docker registry URL (default: local)
    DEPLOYMENT_TIMEOUT       Deployment timeout in seconds (default: 300)
    BACKUP_RETENTION_DAYS    Backup retention period (default: 30)
    SLACK_WEBHOOK_URL        Slack webhook for notifications
    HEALTH_CHECK_RETRIES     Number of health check retries (default: 5)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -p|--no-parallel)
                PARALLEL_DEPLOY=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ ! " ${ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
        log "ERROR" "Invalid environment: $ENVIRONMENT"
        log "INFO" "Valid environments: ${ENVIRONMENTS[*]}"
        exit 1
    fi
    
    if [[ "$SERVICE" != "all" ]] && [[ ! " ${SERVICES[@]} " =~ " ${SERVICE} " ]]; then
        log "ERROR" "Invalid service: $SERVICE"
        log "INFO" "Valid services: all ${SERVICES[*]}"
        exit 1
    fi
}

# Setup logging
setup_logging() {
    mkdir -p "$LOGS_DIR"
    touch "$DEPLOYMENT_LOG"
    
    log "INFO" "Starting deployment script"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Service: $SERVICE"
    log "INFO" "Dry run: $DRY_RUN"
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "curl" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log "ERROR" "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    local env_file="$PROJECT_ROOT/.env.${ENVIRONMENT}"
    if [[ ! -f "$env_file" ]]; then
        log "ERROR" "Environment file not found: $env_file"
        exit 1
    fi
    
    log "INFO" "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log "INFO" "Skipping tests as requested"
        return 0
    fi
    
    log "INFO" "Running tests..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would run tests here"
        return 0
    fi
    
    # Run unit tests
    if ! make test-unit; then
        log "ERROR" "Unit tests failed"
        exit 1
    fi
    
    # Run integration tests
    if ! make test-integration; then
        log "ERROR" "Integration tests failed"
        exit 1
    fi
    
    # Run security tests
    if ! make security-scan; then
        log "ERROR" "Security scan failed"
        exit 1
    fi
    
    log "INFO" "All tests passed"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log "INFO" "Skipping backup as requested"
        return 0
    fi
    
    log "INFO" "Creating backup..."
    
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="$BACKUP_DIR/${ENVIRONMENT}_${backup_timestamp}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would create backup at $backup_path"
        return 0
    fi
    
    mkdir -p "$backup_path"
    
    # Backup database
    log "INFO" "Backing up database..."
    if ! make db-backup BACKUP_FILE="$backup_path/database.sql"; then
        log "ERROR" "Database backup failed"
        exit 1
    fi
    
    # Backup vector database
    log "INFO" "Backing up vector database..."
    if ! make vector-backup; then
        log "ERROR" "Vector database backup failed"
        exit 1
    fi
    
    # Backup configuration
    log "INFO" "Backing up configuration..."
    cp "$PROJECT_ROOT/.env.${ENVIRONMENT}" "$backup_path/"
    cp "$PROJECT_ROOT/docker-compose.yml" "$backup_path/"
    
    # Create backup manifest
    cat > "$backup_path/manifest.json" << EOF
{
    "environment": "$ENVIRONMENT",
    "timestamp": "$backup_timestamp",
    "services": $(printf '%s\n' "${SERVICES[@]}" | jq -R . | jq -s .),
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)"
}
EOF
    
    log "INFO" "Backup created at $backup_path"
    
    # Cleanup old backups
    find "$BACKUP_DIR" -name "${ENVIRONMENT}_*" -type d -mtime +${BACKUP_RETENTION_DAYS:-30} -exec rm -rf {} + || true
}

# Build services
build_services() {
    log "INFO" "Building services..."
    
    local services_to_build=("${SERVICES[@]}")
    if [[ "$SERVICE" != "all" ]]; then
        services_to_build=("$SERVICE")
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would build services: ${services_to_build[*]}"
        return 0
    fi
    
    # Build images
    if [[ "$PARALLEL_DEPLOY" == true ]]; then
        log "INFO" "Building services in parallel..."
        
        local pids=()
        for service in "${services_to_build[@]}"; do
            (
                log "INFO" "Building $service..."
                if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" build "$service"; then
                    log "INFO" "Successfully built $service"
                else
                    log "ERROR" "Failed to build $service"
                    exit 1
                fi
            ) &
            pids+=($!)
        done
        
        # Wait for all builds to complete
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                log "ERROR" "Build process failed"
                exit 1
            fi
        done
    else
        log "INFO" "Building services sequentially..."
        for service in "${services_to_build[@]}"; do
            log "INFO" "Building $service..."
            if ! docker-compose -f "$PROJECT_ROOT/docker-compose.yml" build "$service"; then
                log "ERROR" "Failed to build $service"
                exit 1
            fi
        done
    fi
    
    log "INFO" "All services built successfully"
}

# Deploy services
deploy_services() {
    log "INFO" "Deploying services..."
    
    local services_to_deploy=("${SERVICES[@]}")
    if [[ "$SERVICE" != "all" ]]; then
        services_to_deploy=("$SERVICE")
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would deploy services: ${services_to_deploy[*]}"
        return 0
    fi
    
    # Load environment variables
    export $(cat "$PROJECT_ROOT/.env.${ENVIRONMENT}" | grep -v '^#' | xargs)
    
    # Deploy services
    local compose_file="$PROJECT_ROOT/docker-compose.yml"
    if [[ -f "$PROJECT_ROOT/docker-compose.${ENVIRONMENT}.yml" ]]; then
        compose_file="$compose_file -f $PROJECT_ROOT/docker-compose.${ENVIRONMENT}.yml"
    fi
    
    if [[ "$PARALLEL_DEPLOY" == true ]]; then
        log "INFO" "Deploying services in parallel..."
        
        # Stop services first
        docker-compose -f $compose_file down --remove-orphans
        
        # Start services
        docker-compose -f $compose_file up -d "${services_to_deploy[@]}"
    else
        log "INFO" "Deploying services sequentially..."
        for service in "${services_to_deploy[@]}"; do
            log "INFO" "Deploying $service..."
            docker-compose -f $compose_file up -d "$service"
            
            # Wait for service to be ready
            if ! wait_for_service_health "$service"; then
                log "ERROR" "Service $service failed health check"
                if [[ "$FORCE_DEPLOY" != true ]]; then
                    exit 1
                fi
            fi
        done
    fi
    
    log "INFO" "Services deployed successfully"
}

# Wait for service health
wait_for_service_health() {
    local service=$1
    local timeout=${HEALTH_CHECK_TIMEOUT:-300}
    local retry_count=0
    local max_retries=${HEALTH_CHECK_RETRIES:-5}
    
    log "INFO" "Waiting for $service to be healthy..."
    
    while [[ $retry_count -lt $max_retries ]]; do
        if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" exec -T "$service" curl -f http://localhost:8000/health &> /dev/null; then
            log "INFO" "$service is healthy"
            return 0
        fi
        
        sleep 10
        ((retry_count++))
        log "INFO" "Health check attempt $retry_count/$max_retries for $service"
    done
    
    log "ERROR" "$service failed health check after $max_retries attempts"
    return 1
}

# Run comprehensive health checks
run_health_checks() {
    log "INFO" "Running comprehensive health checks..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would run health checks"
        return 0
    fi
    
    # Check system health
    if ! make health-check; then
        log "ERROR" "System health check failed"
        if [[ "$FORCE_DEPLOY" != true ]]; then
            exit 1
        fi
    fi
    
    # Check individual services
    local services_to_check=("${SERVICES[@]}")
    if [[ "$SERVICE" != "all" ]]; then
        services_to_check=("$SERVICE")
    fi
    
    for service in "${services_to_check[@]}"; do
        if ! wait_for_service_health "$service"; then
            log "ERROR" "Health check failed for $service"
            if [[ "$FORCE_DEPLOY" != true ]]; then
                exit 1
            fi
        fi
    done
    
    # Run smoke tests
    log "INFO" "Running smoke tests..."
    if ! make test-smoke; then
        log "ERROR" "Smoke tests failed"
        if [[ "$FORCE_DEPLOY" != true ]]; then
            exit 1
        fi
    fi
    
    log "INFO" "All health checks passed"
}

# Rollback deployment
rollback_deployment() {
    if [[ -z "$ROLLBACK_VERSION" ]]; then
        log "ERROR" "No rollback version specified"
        exit 1
    fi
    
    log "INFO" "Rolling back to version $ROLLBACK_VERSION..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN: Would rollback to version $ROLLBACK_VERSION"
        return 0
    fi
    
    # Find backup
    local backup_path="$BACKUP_DIR/${ENVIRONMENT}_${ROLLBACK_VERSION}"
    if [[ ! -d "$backup_path" ]]; then
        log "ERROR" "Backup not found: $backup_path"
        exit 1
    fi
    
    # Restore database
    log "INFO" "Restoring database..."
    if ! make db-restore BACKUP_FILE="$backup_path/database.sql"; then
        log "ERROR" "Database restore failed"
        exit 1
    fi
    
    # Restore configuration
    log "INFO" "Restoring configuration..."
    cp "$backup_path/.env.${ENVIRONMENT}" "$PROJECT_ROOT/"
    cp "$backup_path/docker-compose.yml" "$PROJECT_ROOT/"
    
    # Restart services
    log "INFO" "Restarting services..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" restart
    
    # Verify rollback
    if ! run_health_checks; then
        log "ERROR" "Rollback verification failed"
        exit 1
    fi
    
    log "INFO" "Rollback completed successfully"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        return 0
    fi
    
    local color="good"
    if [[ "$status" == "error" ]]; then
        color="danger"
    elif [[ "$status" == "warning" ]]; then
        color="warning"
    fi
    
    local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "fields": [
                {
                    "title": "RAG System Deployment",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Service",
                    "value": "$SERVICE",
                    "short": true
                },
                {
                    "title": "Git Commit",
                    "value": "$(git rev-parse --short HEAD)",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date '+%Y-%m-%d %H:%M:%S')",
                    "short": true
                }
            ]
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$payload" \
         "$SLACK_WEBHOOK_URL" &> /dev/null || true
}

# Main deployment function
main() {
    parse_args "$@"
    validate_environment
    setup_logging
    
    # Handle rollback
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        rollback_deployment
        send_notification "good" "Rollback to $ROLLBACK_VERSION completed successfully"
        exit 0
    fi
    
    # Start deployment
    log "INFO" "Starting deployment process..."
    send_notification "good" "Deployment started"
    
    # Trap for cleanup on exit
    trap 'send_notification "error" "Deployment failed"; exit 1' ERR
    
    # Deployment steps
    check_prerequisites
    run_tests
    create_backup
    build_services
    deploy_services
    run_health_checks
    
    log "INFO" "Deployment completed successfully!"
    send_notification "good" "Deployment completed successfully"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
