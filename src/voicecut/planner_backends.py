#!/usr/bin/env python3
"""Provider-neutral structured-JSON backends for VoiceCut planners.

The semantic narration planner and the pause classifier deliberately share
this small transport layer.  Validation, grounding, and the one-retry policy
remain in those planners; a backend has only one responsibility: return the
model's raw textual JSON response.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import select
import subprocess
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_QWEN_MODEL = "mlx-community/Qwen3.5-9B-4bit"
DEFAULT_GEMMA_MODEL = "mlx-community/gemma-3-12b-it-4bit"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LOCAL_PYTHON = Path(__file__).resolve().parents[2] / ".venv-mlx/bin/python"

PLANNER_BACKENDS = ("gemini", "openai", "deepseek", "local", "qwen", "gemma")
API_KEY_ENV_BY_BACKEND = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
DEFAULT_MODEL_BY_BACKEND = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "local": DEFAULT_QWEN_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "gemma": DEFAULT_GEMMA_MODEL,
}

NARRATION_SYSTEM_INSTRUCTION = (
    "You are a precise source-grounded narration editor. "
    "Return only the structured JSON response requested by the schema."
)
PAUSE_SYSTEM_INSTRUCTION = (
    "You classify narration transitions. Return only the exact JSON "
    "structure requested; never edit or reproduce narration."
)


@dataclass(frozen=True)
class PlannerRuntimeConfiguration:
    """Validated, non-secret planner configuration used by the pipeline."""

    provider: str
    base_url: str | None
    api_key_env: str | None


class PlannerBackend(Protocol):
    """Minimal transport contract shared by all semantic planners."""

    backend_name: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        """Return the raw model response without interpreting it."""

    def close(self) -> None:
        """Release network or local-model resources."""


def _dotenv_value(env_file: Path, key: str) -> str | None:
    """Read one secret with environment variables taking precedence."""

    environment_value = os.environ.get(key)
    if isinstance(environment_value, str) and environment_value.strip():
        return environment_value.strip()
    if not env_file.is_file():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError as error:
        raise RuntimeError(
            "Reading API keys from .env requires `python-dotenv`; install "
            "`voicecut[cloud]` or export the key in the environment"
        ) from error
    value = dotenv_values(env_file).get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_api_key(
    *,
    provider: str,
    env_file: Path,
    api_key_env: str | None,
) -> str:
    key_name = api_key_env or API_KEY_ENV_BY_BACKEND[provider]
    api_key = _dotenv_value(env_file, key_name)
    if api_key is None:
        raise RuntimeError(f"{key_name} is missing from the environment and {env_file}")
    return api_key


def sanitize_planner_base_url(value: str) -> str:
    """Validate and canonicalize a credential-free HTTP(S) API base URL."""

    raw = value.strip()
    if not raw:
        raise ValueError("planner base URL cannot be empty")
    if any(character.isspace() or ord(character) < 32 for character in raw):
        raise ValueError("planner base URL cannot contain whitespace or control bytes")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid planner base URL: {error}") from None
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("planner base URL must use http or https")
    if not parsed.hostname:
        raise ValueError("planner base URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "planner base URL must not contain credentials; use "
            "--planner-api-key-env instead"
        )
    if parsed.query:
        raise ValueError(
            "planner base URL must not contain query parameters because they "
            "may expose credentials"
        )
    if parsed.fragment:
        raise ValueError("planner base URL must not contain a URL fragment")

    hostname = parsed.hostname.casefold()
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (parsed.scheme.casefold() == "https" and port == 443) or (
        parsed.scheme.casefold() == "http" and port == 80
    )
    if port is not None and not default_port:
        rendered_host += f":{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            rendered_host,
            path,
            "",
            "",
        )
    )


def resolve_planner_base_url(
    *,
    provider: str,
    env_file: Path,
    base_url: str | None,
) -> str | None:
    """Resolve the effective endpoint without returning any credential."""

    if provider not in PLANNER_BACKENDS:
        raise ValueError(f"unsupported planner backend: {provider}")
    if provider not in {"openai", "deepseek"}:
        if base_url is not None:
            raise ValueError(
                "--planner-base-url is supported only for openai and deepseek"
            )
        return None

    selected = base_url
    if selected is None:
        selected = _dotenv_value(env_file, f"{provider.upper()}_BASE_URL")
    if selected is None:
        selected = (
            DEFAULT_OPENAI_BASE_URL
            if provider == "openai"
            else DEFAULT_DEEPSEEK_BASE_URL
        )
    return sanitize_planner_base_url(selected)


def preflight_planner_backend(
    *,
    provider: str,
    env_file: Path,
    base_url: str | None,
    api_key_env: str | None,
    local_python: Path,
) -> PlannerRuntimeConfiguration:
    """Fail before media work when planner credentials or SDKs are unavailable."""

    effective_base_url = resolve_planner_base_url(
        provider=provider,
        env_file=env_file,
        base_url=base_url,
    )
    if provider in API_KEY_ENV_BY_BACKEND:
        selected_key_env = api_key_env or API_KEY_ENV_BY_BACKEND[provider]
        _require_api_key(
            provider=provider,
            env_file=env_file,
            api_key_env=api_key_env,
        )
        if provider == "gemini":
            try:
                from google import genai  # noqa: F401
                from google.genai import types  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "Gemini planner support is unavailable; reinstall with "
                    "`pip install -e '.[cloud]'`"
                ) from error
        else:
            try:
                import openai  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    f"{provider} planner support is unavailable; reinstall with "
                    "`pip install -e '.[cloud]'`"
                ) from error
        return PlannerRuntimeConfiguration(
            provider=provider,
            base_url=effective_base_url,
            api_key_env=selected_key_env,
        )

    if not local_python.is_file():
        raise FileNotFoundError(f"local model Python does not exist: {local_python}")
    return PlannerRuntimeConfiguration(
        provider=provider,
        base_url=None,
        api_key_env=None,
    )


def _schema_prompt(prompt: str, response_schema: dict[str, Any]) -> str:
    return (
        prompt
        + "\n\nReturn JSON only, matching this JSON schema exactly:\n"
        + json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    )


def _response_text(response: Any, *, provider: str) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"{provider} returned an invalid chat-completion response"
        ) from error
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            text = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if isinstance(text, str):
                pieces.append(text)
        combined = "".join(pieces)
        if combined.strip():
            return combined
    raise RuntimeError(f"{provider} returned no textual JSON response")


class GeminiPlannerBackend:
    """Google Gen AI structured-output transport."""

    backend_name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        system_instruction: str = NARRATION_SYSTEM_INSTRUCTION,
        client: Any | None = None,
        types_module: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key is empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if client is None or types_module is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise RuntimeError(
                    "Gemini requires `pip install -e '.[cloud]'`"
                ) from error
            client = genai.Client(api_key=api_key)
            types_module = types
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self._types = types_module
        self._client = client

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path,
        model: str = DEFAULT_GEMINI_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        system_instruction: str = NARRATION_SYSTEM_INSTRUCTION,
        api_key_env: str | None = None,
    ) -> "GeminiPlannerBackend":
        return cls(
            model=model,
            api_key=_require_api_key(
                provider="gemini",
                env_file=env_file,
                api_key_env=api_key_env,
            ),
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        del request_id
        config_values: dict[str, Any] = {
            "system_instruction": self.system_instruction,
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
        }
        config_fields = getattr(
            self._types.GenerateContentConfig,
            "model_fields",
            {},
        )
        if "response_format" in config_fields:
            config_values["response_format"] = {
                "text": {
                    "mime_type": "application/json",
                    "schema": response_schema,
                }
            }
        else:
            config_values["response_mime_type"] = "application/json"
            config_values["response_json_schema"] = response_schema
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(**config_values),
        )
        raw = response.text
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("Gemini returned no textual JSON response")
        return raw

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class OpenAICompatiblePlannerBackend:
    """OpenAI Responses transport plus DeepSeek Chat Completions."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        system_instruction: str = NARRATION_SYSTEM_INSTRUCTION,
        base_url: str | None = None,
        strict_json_schema: bool,
        client: Any | None = None,
    ) -> None:
        if provider not in {"openai", "deepseek"}:
            raise ValueError(f"unsupported OpenAI-compatible provider: {provider}")
        if not api_key.strip():
            raise ValueError(f"{provider} API key is empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    f"{provider} requires `pip install -e '.[cloud]'`"
                ) from error
            client_arguments: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_arguments["base_url"] = base_url
            client = OpenAI(**client_arguments)
        self.backend_name = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self.base_url = base_url
        self.strict_json_schema = strict_json_schema
        self._client = client

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        del request_id
        if self.strict_json_schema:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "voicecut_planner_response",
                        "strict": True,
                        "schema": response_schema,
                    }
                },
            )
            raw = getattr(response, "output_text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("openai returned no textual JSON response")
            return raw

        response_format = {"type": "json_object"}
        request_prompt = _schema_prompt(prompt, response_schema)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": request_prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_output_tokens,
            response_format=response_format,
        )
        return _response_text(response, provider=self.backend_name)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class LocalMLXPlannerBackend:
    """Long-lived local MLX-LM worker for Hugging Face model repositories."""

    backend_name = "local"

    def __init__(
        self,
        *,
        python: Path,
        model: str = DEFAULT_QWEN_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        local_files_only: bool = False,
        enable_thinking: bool = False,
        system_instruction: str = NARRATION_SYSTEM_INSTRUCTION,
        timeout_seconds: float = 600.0,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.system_instruction = system_instruction
        self.timeout_seconds = timeout_seconds
        self.command = [
            str(python),
            "-m",
            "voicecut.local_llm_worker",
            "--model",
            model,
            "--max-tokens",
            str(max_output_tokens),
        ]
        if local_files_only:
            self.command.append("--local-files-only")
        if enable_thinking:
            self.command.append("--enable-thinking")
        self._process: subprocess.Popen[str] | None = None
        self._request_count = 0

    def _start(self) -> subprocess.Popen[str]:
        if self._process is None:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        if self._process.poll() is not None:
            raise RuntimeError(
                f"local model worker exited with code {self._process.returncode}"
            )
        return self._process

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        process = self._start()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("local model worker pipes are unavailable")
        self._request_count += 1
        process.stdin.write(
            json.dumps(
                {
                    "prompt": _schema_prompt(prompt, response_schema),
                    "request_id": f"{request_id}:{self._request_count}",
                    "system_instruction": self.system_instruction,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        process.stdin.flush()
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            process.kill()
            process.wait()
            self._process = None
            raise TimeoutError("local semantic planning timed out")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("local model worker ended before returning a response")
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("ok") is not True:
            detail = (
                response.get("error")
                if isinstance(response, dict)
                else "invalid worker response"
            )
            raise RuntimeError(f"local model generation failed: {detail}")
        output = response.get("output")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("local model returned no textual JSON output")
        return output

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class QwenPlannerBackend(LocalMLXPlannerBackend):
    """Backward-compatible local-Qwen adapter."""

    backend_name = "qwen"


class GemmaPlannerBackend(LocalMLXPlannerBackend):
    """Convenience adapter selecting the default local Gemma model."""

    backend_name = "gemma"


def default_model_for_backend(provider: str) -> str:
    try:
        return DEFAULT_MODEL_BY_BACKEND[provider]
    except KeyError as error:
        raise ValueError(f"unsupported planner backend: {provider}") from error


def create_planner_backend(
    *,
    provider: str,
    model: str | None,
    env_file: Path = Path(".env"),
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    system_instruction: str = NARRATION_SYSTEM_INSTRUCTION,
    base_url: str | None = None,
    api_key_env: str | None = None,
    local_python: Path = DEFAULT_LOCAL_PYTHON,
    local_files_only: bool = False,
) -> PlannerBackend:
    """Create one planner transport from provider-neutral configuration."""

    selected_model = model or default_model_for_backend(provider)
    selected_base_url = resolve_planner_base_url(
        provider=provider,
        env_file=env_file,
        base_url=base_url,
    )
    if provider == "gemini":
        return GeminiPlannerBackend.from_env(
            env_file=env_file,
            model=selected_model,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            api_key_env=api_key_env,
        )
    if provider in {"openai", "deepseek"}:
        return OpenAICompatiblePlannerBackend(
            provider=provider,
            model=selected_model,
            api_key=_require_api_key(
                provider=provider,
                env_file=env_file,
                api_key_env=api_key_env,
            ),
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            base_url=selected_base_url,
            strict_json_schema=provider == "openai",
        )
    backend_type: type[LocalMLXPlannerBackend]
    if provider == "qwen":
        backend_type = QwenPlannerBackend
    elif provider == "gemma":
        backend_type = GemmaPlannerBackend
    elif provider == "local":
        backend_type = LocalMLXPlannerBackend
    else:
        raise ValueError(f"unsupported planner backend: {provider}")
    if not local_python.is_file():
        raise FileNotFoundError(f"local model Python does not exist: {local_python}")
    return backend_type(
        python=local_python,
        model=selected_model,
        max_output_tokens=max_output_tokens,
        local_files_only=local_files_only,
        enable_thinking=(
            provider in {"qwen", "local"} and "qwen" in selected_model.casefold()
        ),
        system_instruction=system_instruction,
    )


def add_planner_backend_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_backend: str = "gemini",
) -> None:
    """Add the same model-selection surface to a VoiceCut subcommand."""

    parser.add_argument(
        "--planner-backend",
        choices=PLANNER_BACKENDS,
        default=default_backend,
        help=(
            "LLM transport. Use local with any MLX-LM-compatible Hugging Face "
            "model; qwen and gemma are local convenience aliases."
        ),
    )
    parser.add_argument(
        "--planner-model",
        help="Provider model name or local Hugging Face model/path.",
    )
    parser.add_argument(
        "--planner-base-url",
        help=(
            "Optional OpenAI-compatible API base URL. DeepSeek defaults to "
            "https://api.deepseek.com."
        ),
    )
    parser.add_argument(
        "--planner-api-key-env",
        help=(
            "Environment/.env variable containing the API key. Defaults to "
            "the standard variable for the selected provider."
        ),
    )
    parser.add_argument(
        "--planner-python",
        type=Path,
        default=DEFAULT_LOCAL_PYTHON,
        help="Python executable from the local MLX environment.",
    )
    parser.add_argument("--local-files-only", action="store_true")
