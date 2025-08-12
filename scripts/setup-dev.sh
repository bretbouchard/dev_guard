#!/bin/bash
# DevGuard Development Environment Setup Script
# This script sets up the complete development environment with testing infrastructure

set -e  # Exit on any error

echo "🚀 Setting up DevGuard development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_success "Python $python_version is compatible"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev]"
print_success "Dependencies installed"

# Install pre-commit hooks
print_status "Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
print_success "Pre-commit hooks installed"

# Create necessary directories
print_status "Creating test directories..."
mkdir -p tests/{unit,integration,performance,security}
mkdir -p test_repos
mkdir -p logs
mkdir -p .coverage_data
print_success "Directories created"

# Set up Git configuration for testing
print_status "Setting up Git configuration for testing..."
if ! git config user.name > /dev/null 2>&1; then
    git config user.name "DevGuard Test User"
    git config user.email "test@devguard.local"
    print_success "Git configuration set for testing"
else
    print_status "Git configuration already exists"
fi

# Run initial quality checks
print_status "Running initial quality checks..."
if command -v make > /dev/null 2>&1; then
    make format
    make lint
    print_success "Initial quality checks passed"
else
    print_warning "Make not available, skipping initial quality checks"
fi

# Run initial tests to verify setup
print_status "Running initial tests to verify setup..."
if pytest tests/unit/test_sample.py -v --tb=short; then
    print_success "Initial tests passed"
else
    print_warning "Some initial tests failed, but setup is complete"
fi

# Create environment file template
print_status "Creating environment file template..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# DevGuard Environment Configuration
# Copy this file and customize for your environment

# Database Configuration
DATABASE_URL=sqlite:///dev_guard.db
TEST_DATABASE_URL=sqlite:///:memory:

# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Local LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Vector Database Configuration
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=dev_guard_knowledge

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/dev_guard.log

# Testing Configuration
TEST_MODE=false
PYTEST_TIMEOUT=300

# Development Configuration
DEBUG=true
DEVELOPMENT=true
EOF
    print_success "Environment file template created"
else
    print_status "Environment file already exists"
fi

# Create development configuration
print_status "Creating development configuration..."
mkdir -p config
if [ ! -f "config/dev.yaml" ]; then
    cat > config/dev.yaml << EOF
# DevGuard Development Configuration

database:
  url: "sqlite:///dev_guard_dev.db"
  echo: true
  pool_size: 5

vector_db:
  path: "./data/chroma_dev"
  collection_name: "dev_guard_dev"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

llm:
  provider: "ollama"
  model: "qwen/qwen3-235b-a22b:free"
  base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 4096
  fallback_provider: "openrouter"
  fallback_model: "meta-llama/llama-3.1-8b-instruct:free"

agents:
  max_retries: 3
  timeout: 30
  heartbeat_interval: 10
  memory_limit: 1000

repositories: []

notifications:
  enabled: false
  channels: []

logging:
  level: "DEBUG"
  file: "logs/dev_guard_dev.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF
    print_success "Development configuration created"
else
    print_status "Development configuration already exists"
fi

# Create test data directory
print_status "Creating test data directory..."
mkdir -p tests/data
if [ ! -f "tests/data/sample_code.py" ]; then
    cat > tests/data/sample_code.py << EOF
"""
Sample code file for testing purposes.
This file contains various Python constructs for testing code analysis.
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SampleData:
    """Sample data class for testing"""
    id: str
    name: str
    value: int
    metadata: Dict[str, str] = None

    def validate(self) -> bool:
        """Validate the data"""
        return all([
            self.id and len(self.id) > 0,
            self.name and len(self.name) > 0,
            self.value >= 0
        ])


class SampleProcessor:
    """Sample processor class for testing"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.processed_count = 0
    
    def process_data(self, data: SampleData) -> Dict[str, str]:
        """Process sample data"""
        if not data.validate():
            raise ValueError("Invalid data provided")
        
        result = {
            "processed_id": data.id,
            "processed_name": data.name.upper(),
            "processed_value": str(data.value * 2)
        }
        
        self.processed_count += 1
        return result
    
    async def async_process(self, data_list: List[SampleData]) -> List[Dict[str, str]]:
        """Asynchronously process multiple data items"""
        tasks = []
        for data in data_list:
            task = asyncio.create_task(self._async_process_single(data))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _async_process_single(self, data: SampleData) -> Dict[str, str]:
        """Process a single data item asynchronously"""
        await asyncio.sleep(0.01)  # Simulate async work
        return self.process_data(data)


def sample_function(x: int, y: int) -> int:
    """Sample function for testing"""
    return x + y


async def sample_async_function(delay: float = 0.1) -> str:
    """Sample async function for testing"""
    await asyncio.sleep(delay)
    return "async_result"
EOF
    print_success "Sample test data created"
fi

# Display setup summary
echo ""
echo "🎉 DevGuard development environment setup complete!"
echo ""
echo "📋 Setup Summary:"
echo "  ✅ Python $python_version virtual environment"
echo "  ✅ Development dependencies installed"
echo "  ✅ Pre-commit hooks configured"
echo "  ✅ Test directories created"
echo "  ✅ Configuration files created"
echo "  ✅ Sample test data generated"
echo ""
echo "🚀 Next Steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Review and customize .env file"
echo "  3. Review and customize config/dev.yaml"
echo "  4. Run tests: make test"
echo "  5. Start developing!"
echo ""
echo "📚 Available Commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run all tests"
echo "  make quality       - Run quality checks"
echo "  make dev-test      - Run development testing cycle"
echo ""
echo "Happy coding! 🚀"