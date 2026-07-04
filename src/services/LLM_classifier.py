"""
Structured Sentiment LLM Classifier.

Handles structured sentiment classification using Google Gemini (via the
google-genai SDK) or OpenAI. Outputs a validated Pydantic schema with
raw sentiment, exposure score, confidence, and rationale.

Supports:
  - Gemini: response_schema with JSON mode for native structured output.
  - OpenAI: beta.chat.completions.parse for structured outputs.
  - Auto-detection of provider from environment variables.
  - Lazy client initialization to defer API key validation.
"""
import os
import json
import logging
from typing import Optional, Any

from src.database.models import SentimentClassification, SentimentEnum, FusedPayload
from src.pipeline.fusion_core import FusionCore

logger = logging.getLogger(__name__)

# System prompt shared across providers
_SYSTEM_PROMPT = (
    "You are an expert market analyst. Analyze the provided news alert "
    "and its historical context to determine market sentiment, "
    "exposure score (degree of market impact), and classification confidence. "
    "Return a structured JSON response."
)


class LLMClassifier:
    """
    Dual-provider structured sentiment classifier.

    Wraps both Google Gemini and OpenAI behind a unified interface,
    returning validated SentimentClassification Pydantic objects.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Auto-detect provider if not explicitly given
        if not provider:
            if os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            elif os.environ.get("GEMINI_API_KEY"):
                provider = "gemini"
            else:
                # Default to Gemini if google-genai can be imported
                try:
                    import google.genai  # noqa: F401
                    provider = "gemini"
                except ImportError:
                    try:
                        import openai  # noqa: F401
                        provider = "openai"
                    except ImportError:
                        raise ValueError(
                            "Neither 'google-genai' nor 'openai' library is installed, "
                            "and no provider was explicitly requested."
                        )

        self.provider = provider.lower()
        if self.provider not in ("gemini", "openai"):
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'gemini' or 'openai'."
            )

        self.api_key = api_key or os.environ.get(
            "OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"
        )

        if not self.api_key:
            logger.warning(
                f"No API key provided for {self.provider}. "
                f"Ensure {self.provider.upper()}_API_KEY env variable is set "
                f"before calling classify."
            )

        # Set default models
        if not model_name:
            if self.provider == "openai":
                self.model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
            else:
                self.model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        else:
            self.model_name = model_name

        self._client = None
        logger.info(
            f"LLMClassifier initialized with provider: {self.provider}, "
            f"model: {self.model_name}"
        )

    def _get_client(self) -> Any:
        """Lazily initialize and return the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' library is not installed. "
                    "Install it using `pip install openai` to use the OpenAI provider."
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        else:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' library is not installed. "
                    "Install it using `pip install google-genai` to use the Gemini provider."
                )
            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def classify(self, prompt: str) -> SentimentClassification:
        """
        Executes structured sentiment classification on the provided prompt.

        Args:
            prompt: Formatted text containing the alert and its contexts.

        Returns:
            Validated SentimentClassification Pydantic object.
        """
        client = self._get_client()

        if self.provider == "openai":
            return self._classify_openai(client, prompt)
        else:
            return self._classify_gemini(client, prompt)

    def _classify_openai(self, client: Any, prompt: str) -> SentimentClassification:
        """Classify using OpenAI structured outputs."""
        try:
            response = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=SentimentClassification,
            )
            parsed_result = response.choices[0].message.parsed
            if parsed_result is None:
                raise ValueError("Failed to parse structured output from OpenAI.")
            return parsed_result
        except Exception as e:
            logger.error(f"OpenAI sentiment classification failed: {e}")
            raise

    def _classify_gemini(self, client: Any, prompt: str) -> SentimentClassification:
        """Classify using Gemini JSON mode with response_schema."""
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SentimentClassification,
                ),
            )

            # Try parsed response first, fall back to manual JSON parsing
            parsed_result = response.parsed
            if parsed_result is None:
                if response.text:
                    parsed_json = json.loads(response.text)
                    return SentimentClassification(**parsed_json)
                raise ValueError("Failed to parse structured output from Gemini.")

            if isinstance(parsed_result, SentimentClassification):
                return parsed_result
            elif isinstance(parsed_result, dict):
                return SentimentClassification(**parsed_result)
            else:
                return SentimentClassification(**dict(parsed_result))
        except Exception as e:
            logger.error(f"Gemini sentiment classification failed: {e}")
            raise

    def classify_payload(
        self, payload: FusedPayload, fusion_core: FusionCore
    ) -> SentimentClassification:
        """
        Convenience method: formats a FusedPayload into a prompt using
        FusionCore and executes classification.

        Args:
            payload: The FusedPayload with RSSAlert and time-decayed contexts.
            fusion_core: FusionCore instance for prompt generation.

        Returns:
            Validated SentimentClassification Pydantic object.
        """
        prompt = fusion_core.generate_llm_prompt(payload)
        return self.classify(prompt)
