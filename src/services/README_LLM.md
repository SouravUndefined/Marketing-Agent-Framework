# LLM Service Documentation

A unified LLM service that allows easy switching between different LLM providers through configuration.

## Supported Providers

- **OpenAI** (GPT-4, GPT-3.5, etc.)
- **Azure OpenAI** (Your current setup)
- **Anthropic** (Claude models)
- **Google Gemini** (Gemini Pro, etc.)
- **Ollama** (Local LLMs)
- **Hugging Face** (Open-source models)

## Quick Start

### 1. Configure Provider in `.env`

Set your desired provider:
```bash
LLM_PROVIDER=azure_openai  # Current default
```

### 2. Basic Usage

```python
from src.services.llm import LLMService

# Initialize (reads provider from .env)
llm = LLMService()

# Generate text
response = await llm.generate(
    prompt="Your question here",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=2000
)
```

### 3. Override Provider Programmatically

```python
# Use specific provider regardless of .env
llm = LLMService(provider="anthropic")
response = await llm.generate(prompt="Hello!")
```

## Installation by Provider

### Azure OpenAI (Current Setup)
```bash
# Already installed
pip install openai>=1.0.0
```

### OpenAI
```bash
pip install openai>=1.0.0
# Set in .env: OPENAI_API_KEY=your_key
```

### Anthropic Claude
```bash
pip install anthropic>=0.18.0
# Set in .env: ANTHROPIC_API_KEY=your_key
```

### Google Gemini
```bash
pip install google-generativeai>=0.3.0
# Set in .env: GOOGLE_API_KEY=your_key
```

### Ollama (Local)
```bash
# Install Ollama from https://ollama.ai
# Run: ollama pull llama2
# No pip install needed (uses OpenAI-compatible API)
```

### Hugging Face
```bash
pip install huggingface-hub>=0.20.0
# Set in .env: HUGGINGFACE_API_KEY=your_key
```

## Features

### Basic Text Generation
```python
response = await llm.generate(
    prompt="Explain machine learning",
    system_prompt="You are an expert teacher.",
    temperature=0.7,
    max_tokens=500
)
```

### Tool/Function Calling (OpenAI & Anthropic)
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = await llm.generate_with_tools(
    prompt="What's the weather in NYC?",
    tools=tools
)
```

## Environment Variables

### Required for All
```bash
LLM_PROVIDER=azure_openai  # Choose your provider
```

### Azure OpenAI (Current)
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
```

### OpenAI
```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini  # optional, defaults to gpt-4o-mini
```

### Anthropic
```bash
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # optional
```

### Gemini
```bash
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-pro  # optional
```

### Ollama
```bash
OLLAMA_BASE_URL=http://localhost:11434  # optional
OLLAMA_MODEL=llama2  # optional
```

### Hugging Face
```bash
HUGGINGFACE_API_KEY=your_key
HUGGINGFACE_MODEL=meta-llama/Llama-2-7b-chat-hf  # optional
```

## Usage in Other Scripts

Simply import and use:

```python
from src.services.llm import LLMService
import asyncio

async def my_function():
    llm = LLMService()  # Auto-reads from .env
    result = await llm.generate("Your prompt")
    return result

# Run
response = asyncio.run(my_function())
```

## Switching Providers

To switch providers, simply change the `LLM_PROVIDER` in your `.env` file:

```bash
# Use OpenAI
LLM_PROVIDER=openai

# Use Claude
LLM_PROVIDER=anthropic

# Use local Ollama
LLM_PROVIDER=ollama
```

No code changes needed! The service automatically handles the routing.

## Error Handling

The service includes comprehensive error handling:
- Missing API keys
- Invalid providers
- Network errors
- Provider-specific errors

All errors are logged and propagated for proper handling in your application.
