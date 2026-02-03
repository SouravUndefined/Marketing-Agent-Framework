import os
import json
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncAzureOpenAI, AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
import asyncio
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables (override=True ensures .env file takes precedence)
load_dotenv(override=True)

class LLMService:
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM Service with specified provider or from environment variable.

        Args:
            provider: LLM provider to use. Options: "openai", "azure_openai", "anthropic",
                     "gemini", "ollama", "huggingface". If None, reads from LLM_PROVIDER env var.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "azure_openai")
        self.client = None
        self._initialize_client()
        logger.info(f"LLMService initialized with provider: {self.provider}")

    def _initialize_client(self):
        """Initialize the appropriate client based on the provider."""
        try:
            if self.provider == "azure_openai":
                self.client = AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

            elif self.provider == "openai":
                self.client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            elif self.provider == "anthropic":
                try:
                    
                    self.client = AsyncAnthropic(
                        api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
                    self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                except ImportError:
                    raise ImportError("Please install anthropic: pip install anthropic")

            elif self.provider == "gemini":
                try:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY environment variable not set")
                    logger.info(f"Initializing Gemini with API key: {api_key[:10]}...{api_key[-10:]}")
                    self.client = genai.Client(api_key=api_key)
                    self.model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
                    logger.info(f"Gemini model: {self.model}")
                except ImportError:
                    raise ImportError("Please install google-genai: pip install google-genai")

            elif self.provider == "ollama":
                # Ollama uses OpenAI-compatible API (already imported at top)
                self.client = AsyncOpenAI(
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    api_key="ollama"  # Ollama doesn't require an API key
                )
                self.model = os.getenv("OLLAMA_MODEL", "llama3")

            elif self.provider == "huggingface":
                # Check if using local pipeline or API endpoint
                use_local = os.getenv("HF_USE_LOCAL", "false").lower() == "true"
                use_langchain = os.getenv("HF_USE_LANGCHAIN", "false").lower() == "true"

                self.model = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
                self.hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

                if use_langchain:
                    # Use LangChain wrappers
                    try:
                        if use_local:
                            # Local: HuggingFacePipeline (requires transformers + torch)
                            from langchain_huggingface import HuggingFacePipeline
                            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                            logger.info(f"Loading local model: {self.model}")
                            tokenizer = AutoTokenizer.from_pretrained(self.model)
                            model = AutoModelForCausalLM.from_pretrained(self.model)
                            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

                            self.client = HuggingFacePipeline(pipeline=pipe)
                            self.hf_mode = "langchain_pipeline"
                            logger.info("Using HuggingFacePipeline via LangChain (local)")
                        else:
                            # API: HuggingFaceEndpoint
                            from langchain_huggingface import HuggingFaceEndpoint

                            self.client = HuggingFaceEndpoint(
                                repo_id=self.model,
                                huggingfacehub_api_token=self.hf_api_token,
                                temperature=0.7,
                                max_new_tokens=2000
                            )
                            self.hf_mode = "langchain_endpoint"
                            logger.info("Using HuggingFaceEndpoint via LangChain (API)")
                    except ImportError as e:
                        raise ImportError(
                            f"Please install required packages: pip install langchain-huggingface transformers torch. Error: {str(e)}"
                        )
                else:
                    # Use direct implementations
                    try:
                        if use_local:
                            # Local: Direct transformers pipeline
                            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                            logger.info(f"Loading local model: {self.model}")
                            tokenizer = AutoTokenizer.from_pretrained(self.model)
                            model = AutoModelForCausalLM.from_pretrained(self.model)
                            self.client = pipeline("text-generation", model=model, tokenizer=tokenizer)
                            self.hf_mode = "direct_pipeline"
                            logger.info("Using transformers pipeline directly (local)")
                        else:
                            # API: Direct huggingface_hub InferenceClient
                            from huggingface_hub import InferenceClient

                            self.client = InferenceClient(
                                model=self.model,
                                token=self.hf_api_token
                            )
                            self.hf_mode = "direct_endpoint"
                            logger.info("Using InferenceClient directly (API)")
                    except ImportError as e:
                        raise ImportError(
                            f"Please install required packages: pip install huggingface-hub transformers torch. Error: {str(e)}"
                        )

            else:
                raise ValueError(f"Unsupported provider: {self.provider}. Choose from: openai, azure_openai, anthropic, gemini, ollama, huggingface")

        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {str(e)}")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            prompt: The user prompt/message
            system_prompt: System prompt for setting context (optional)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        try:
            if self.provider in ["openai", "azure_openai", "ollama"]:
                return await self._generate_openai_compatible(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "anthropic":
                return await self._generate_anthropic(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "gemini":
                return await self._generate_gemini(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "huggingface":
                return await self._generate_huggingface(prompt, system_prompt, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Error generating text with {self.provider}: {str(e)}")
            raise

    async def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI-compatible API (OpenAI, Azure OpenAI, Ollama)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Anthropic Claude API."""
        message_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }

        if system_prompt:
            message_params["system"] = system_prompt

        response = await self.client.messages.create(**message_params)
        return response.content[0].text

    async def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Google Gemini API."""
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }

        # Build the content parts
        contents = []
        if system_prompt:
            contents.append(system_prompt)
        contents.append(prompt)

        full_prompt = "\n\n".join(contents)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=config
        )
        return response.text

    async def _generate_huggingface(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Hugging Face (supports 4 modes: langchain/direct + local/API)."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if self.hf_mode == "langchain_endpoint":
            # LangChain HuggingFaceEndpoint (API mode)
            response = await asyncio.to_thread(self.client.invoke, full_prompt)
            return response

        elif self.hf_mode == "langchain_pipeline":
            # LangChain HuggingFacePipeline (local mode)
            response = await asyncio.to_thread(self.client.invoke, full_prompt)
            return response

        elif self.hf_mode == "direct_endpoint":
            # Direct InferenceClient (API mode)
            response = await asyncio.to_thread(
                self.client.text_generation,
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response

        elif self.hf_mode == "direct_pipeline":
            # Direct transformers pipeline (local mode)
            response = await asyncio.to_thread(
                self.client,
                full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                **kwargs
            )
            # Pipeline returns a list of dicts with 'generated_text' key
            return response[0]['generated_text'] if response else ""

        else:
            raise ValueError(f"Unknown HuggingFace mode: {self.hf_mode}")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with function/tool calling support (OpenAI and Anthropic).

        Args:
            prompt: The user prompt
            tools: List of tool/function definitions
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Dict containing response and any tool calls
        """
        if self.provider not in ["openai", "azure_openai", "anthropic"]:
            raise NotImplementedError(f"Tool calling not supported for {self.provider}")

        try:
            if self.provider in ["openai", "azure_openai"]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                return {
                    "content": response.choices[0].message.content,
                    "tool_calls": response.choices[0].message.tool_calls
                }

            elif self.provider == "anthropic":
                message_params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": tools,
                    **kwargs
                }

                if system_prompt:
                    message_params["system"] = system_prompt

                response = await self.client.messages.create(**message_params)

                return {
                    "content": response.content,
                    "stop_reason": response.stop_reason
                }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise
