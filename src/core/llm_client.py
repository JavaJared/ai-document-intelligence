"""
LLM client for interacting with language models.

Provides a unified interface for LLM operations with support for
streaming, retries, and multiple model backends.
"""

from typing import List, Dict, Any, Optional, Iterator
import time
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, AuthenticationError, APIConnectionError

from ..utils.config import config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class LLMClient:
    """
    Client for interacting with Large Language Models.
    
    Features:
    - Automatic retry logic with exponential backoff
    - Token counting and cost tracking
    - Streaming support for real-time responses
    - Context management for long conversations
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (e.g., gpt-3.5-turbo)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or config.llm.api_key
        self.model_name = model_name or config.llm.model_name
        self.temperature = temperature or config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        
        self.timeout = config.llm.timeout

        if not self.api_key:
            logger.warning("OpenAI API key not configured - using mock mode")
            self._client = None
        else:
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
        
        logger.info(
            "LLM client initialized",
            model=self.model_name,
            temperature=self.temperature
        )
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message for context
            temperature: Override default temperature
            max_retries: Number of retry attempts
            
        Returns:
            Generated text response
        """
        if not self._client:
            logger.warning("Using mock LLM response")
            return self._mock_response(prompt)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.temperature
        
        logger.debug(
            "Generating LLM response",
            model=self.model_name,
            prompt_length=len(prompt)
        )
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                
                content = response.choices[0].message.content
                
                logger.info(
                    "LLM response generated",
                    model=self.model_name,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
                
                return content
                
            except AuthenticationError as e:
                logger.error("Authentication failed", error=str(e))
                raise Exception(f"OpenAI Authentication Error: Invalid API key. Please check your OPENAI_API_KEY.")

            except APIConnectionError as e:
                logger.error("Connection failed", error=str(e))
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI Connection Error: Could not connect to OpenAI API. Error: {str(e)}")
                time.sleep(2)

            except RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    "Rate limit hit, retrying",
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI Rate Limit: Too many requests. You may need to add billing to your OpenAI account. Error: {str(e)}")
                time.sleep(wait_time)

            except APITimeoutError as e:
                logger.warning(
                    "API timeout, retrying",
                    attempt=attempt + 1
                )
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI Timeout: Request timed out after {self.timeout}s. Try again later.")
                time.sleep(1)

            except APIError as e:
                logger.error(
                    "API error occurred",
                    error=str(e),
                    attempt=attempt + 1
                )
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI API error: {str(e)}")

            except Exception as e:
                logger.error(
                    "Unexpected error during LLM call",
                    error=str(e),
                    error_type=type(e).__name__,
                    attempt=attempt + 1
                )
                if attempt == max_retries - 1:
                    raise Exception(f"LLM error ({type(e).__name__}): {str(e)}")
                time.sleep(1)

        raise Exception("Max retries exceeded - check your OPENAI_API_KEY in Streamlit secrets")
    
    def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Iterator[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Override default temperature
            
        Yields:
            Chunks of generated text
        """
        if not self._client:
            yield self._mock_response(prompt)
            return
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.temperature
        
        logger.debug("Starting streaming generation")
        
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            raise
    
    def _mock_response(self, prompt: str) -> str:
        """
        Generate a mock response when API key is not available.
        
        Args:
            prompt: User prompt
            
        Returns:
            Mock response text
        """
        return (
            f"[MOCK RESPONSE] This is a simulated response to the prompt: "
            f"'{prompt[:100]}...'. Configure OPENAI_API_KEY to use real LLM."
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        # For production, use tiktoken library
        return len(text) // 4
