"""
Example usage of the LLMService class

This demonstrates how to use the LLM service with different providers.
"""
import asyncio
import logging
from llm import LLMService

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


async def main():
    # # Example 1: Use provider from .env file (default)
    # llm = LLMService()
    # response = await llm.generate(
    #     prompt="Explain what is machine learning in simple terms",
    #     system_prompt="You are a helpful AI assistant.",
    #     temperature=0.7,
    #     max_tokens=500
    # )
    # print(f"Response: {response}\n")

    # Example 2: Explicitly specify a provider (overrides .env)
    # llm_openai = LLMService(provider="openai")
    # response = await llm_openai.generate(
    #     prompt="Write a short poem about AI",
    #     temperature=0.9
    # )
    # print(f"OpenAI Response: {response}\n")

    # Example 3: Use with Anthropic Claude
    # llm_anthropic = LLMService(provider="anthropic")
    # response = await llm_anthropic.generate(
    #     prompt="What are the benefits of async programming?",
    #     system_prompt="You are a Python expert.",
    #     temperature=0.5
    # )
    # print(f"Anthropic Response: {response}\n")

    # Example 4: Use with Google Gemini
    # llm_gemini = LLMService(provider="gemini")
    # response = await llm_gemini.generate(
    #     prompt="Explain quantum computing",
    #     temperature=0.7
    # )
    # print(f"Gemini Response: {response}\n")

    # Example 5: Use with local Ollama
    llm_ollama = LLMService(provider="ollama")
    response = await llm_ollama.generate(
        prompt="What is the capital of France?",
        temperature=0.3
    )
    print(f"Ollama Response: {response}\n")

    # Example 6: Use with tool calling (OpenAI/Anthropic only)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # response_with_tools = await llm.generate_with_tools(
    #     prompt="What's the weather like in New York?",
    #     tools=tools,
    #     system_prompt="You are a helpful assistant with access to weather data."
    # )
    # print(f"Response with tools: {response_with_tools}\n")


if __name__ == "__main__":
    asyncio.run(main())
