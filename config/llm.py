"""
LLM Module for Gemini Pro 2.5 Integration

This module provides functionality to get LLM responses from Google's
Gemini Pro 2.5 model through Google Cloud's Vertex AI platform.
"""

import os
import time
import random
from typing import Optional
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

Model = "gemini-2.5-pro"
PROJECT_ID: str = "gpu-reservation-sarvam"
LOCATION: str = "us-east5"


class LLMClient:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        service_account_path: Optional[str] = None,
    ):
        self.project_id = (
            project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or PROJECT_ID
        )
        self.location = (
            location or os.getenv("GOOGLE_CLOUD_LOCATION") or LOCATION
        )
        if service_account_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "service-account.json"
            )
            if os.path.exists(default_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_path
                print(f"Using default service account: {default_path}")

        if not self.location or not self.project_id:
            raise ValueError("GCP project_id and location are required")

        print(
            f"Initializing Gemini client with project: {self.project_id}, "
            f"location: {self.location}"
        )

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project_id, location=self.location)
            self.GenerativeModel = GenerativeModel
            print("Gemini client initialized successfully")
        except ImportError as e:
            print(f"Failed to import Vertex AI SDK: {e}")
            raise RuntimeError(
                "Vertex AI SDK not available. Install google-cloud-aiplatform."
            ) from e
        except Exception as e:
            print(f"Failed to initialize Vertex AI: {e}")
            raise

    def generate(
        self,
        model: str = Model,
        system_prompt: str = "",
        user_prompt: str = "",
        **kwargs,
    ) -> str:
        print(f"Calling Gemini with model: {model}")

        max_retries = int(kwargs.pop("max_retries", 3))
        backoff_base = float(kwargs.pop("retry_backoff_base", 1.0))

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Calling Gemini with model: {model}")
                model_instance = self.GenerativeModel(model_name=model)
                contents = []

                # Add system prompt
                if system_prompt:
                    contents.append(
                        {"role": "system", "parts": [{"text": system_prompt}]}
                    )

                # Add user prompt
                contents.append(
                    {"role": "user", "parts": [{"text": user_prompt}]}
                )

                # Generation config
                generation_config = {
                    # "max_output_tokens": max_tokens,
                    # "temperature": temperature,
                    "thinking_config": {"thinking_budget": 16000},
                }

                response = model_instance.generate_content(
                    contents, generation_config=generation_config
                )

                print("Gemini call successful", response)
                return response.text

            except Exception as e:
                last_exc = e
                print(
                    f"Gemini call failed "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt >= max_retries:
                    break
                # Exponential backoff with small jitter
                sleep_seconds = backoff_base * (2 ** (attempt - 1))
                jitter = min(0.5, sleep_seconds * 0.25) * random.random()
                time.sleep(sleep_seconds + jitter)

        # If all retries failed, raise the last exception
        if last_exc:
            raise last_exc
        raise RuntimeError("Gemini call failed with unknown error")


def main():
    """Test Gemini client."""
    print("=== Gemini LLM Client Test ===\n")

    try:
        client = LLMClient()
        response = client.generate(
            user_prompt="What is 2+2? Answer in one sentence.",
        )
        print(f"✓ Gemini: {response}")
    except Exception as e:
        print(f"✗ Gemini failed: {e}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
