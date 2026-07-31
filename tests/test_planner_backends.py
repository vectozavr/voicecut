from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from voicecut.local_llm_worker import extract_final_json_object, render_chat_prompt
from voicecut.final_render import build_parser as build_final_parser
from voicecut.planner_backends import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_QWEN_MODEL,
    LocalMLXPlannerBackend,
    OpenAICompatiblePlannerBackend,
    default_model_for_backend,
    preflight_planner_backend,
    resolve_planner_base_url,
    sanitize_planner_base_url,
)
from voicecut.streaming_narration import build_parser as build_streaming_parser


class RecordingCompletions:
    def __init__(self, content: str = '{"ok":true}') -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RecordingResponses:
    def __init__(self, content: str = '{"ok":true}') -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.content)


class RecordingClient:
    def __init__(self, content: str = '{"ok":true}') -> None:
        self.completions = RecordingCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)
        self.responses = RecordingResponses(content)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_openai_uses_strict_json_schema() -> None:
    client = RecordingClient()
    backend = OpenAICompatiblePlannerBackend(
        provider="openai",
        model="test-openai",
        api_key="secret",
        strict_json_schema=True,
        client=client,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }

    assert (
        backend.generate(
            "classify this",
            response_schema=schema,
            request_id="ignored",
        )
        == '{"ok":true}'
    )
    call = client.responses.calls[0]
    assert call["model"] == "test-openai"
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "voicecut_planner_response",
            "strict": True,
            "schema": schema,
        },
    }
    assert call["input"][1]["content"] == "classify this"  # type: ignore[index]
    assert "temperature" not in call
    assert "max_tokens" not in call
    assert call["max_output_tokens"] == 8192

    backend.close()
    assert client.closed is True


def test_deepseek_uses_json_object_and_includes_schema_in_prompt() -> None:
    client = RecordingClient()
    backend = OpenAICompatiblePlannerBackend(
        provider="deepseek",
        model=DEFAULT_DEEPSEEK_MODEL,
        api_key="secret",
        base_url=DEFAULT_DEEPSEEK_BASE_URL,
        strict_json_schema=False,
        client=client,
    )
    schema = {
        "type": "object",
        "required": ["transitions"],
        "properties": {"transitions": {"type": "array"}},
    }

    backend.generate(
        "classify transitions",
        response_schema=schema,
        request_id="pause-1",
    )
    call = client.completions.calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["response_format"] == {"type": "json_object"}
    user_prompt = call["messages"][1]["content"]  # type: ignore[index]
    assert "classify transitions" in user_prompt
    assert '"transitions"' in user_prompt


def test_local_backend_accepts_qwen_gemma_or_arbitrary_hf_model(
    tmp_path: Path,
) -> None:
    python = tmp_path / "mlx-python"
    python.write_text("", encoding="utf-8")

    for model in (
        DEFAULT_QWEN_MODEL,
        DEFAULT_GEMMA_MODEL,
        "organization/custom-mlx-model",
    ):
        backend = LocalMLXPlannerBackend(
            python=python,
            model=model,
            max_output_tokens=123,
            local_files_only=True,
        )
        assert backend.model == model
        assert backend.command == [
            str(python),
            "-m",
            "voicecut.local_llm_worker",
            "--model",
            model,
            "--max-tokens",
            "123",
            "--local-files-only",
        ]


def test_model_defaults_are_provider_specific() -> None:
    assert default_model_for_backend("qwen") == DEFAULT_QWEN_MODEL
    assert default_model_for_backend("local") == DEFAULT_QWEN_MODEL
    assert default_model_for_backend("gemma") == DEFAULT_GEMMA_MODEL
    assert default_model_for_backend("deepseek") == "deepseek-v4-flash"
    with pytest.raises(ValueError, match="unsupported planner backend"):
        default_model_for_backend("unknown")


def test_planner_base_url_is_resolved_and_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "HTTPS://API.Example.TEST:443/v1/")
    assert (
        resolve_planner_base_url(
            provider="openai",
            env_file=tmp_path / "missing.env",
            base_url=None,
        )
        == "https://api.example.test/v1"
    )
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert (
        resolve_planner_base_url(
            provider="openai",
            env_file=tmp_path / "missing.env",
            base_url=None,
        )
        == DEFAULT_OPENAI_BASE_URL
    )
    assert (
        resolve_planner_base_url(
            provider="deepseek",
            env_file=tmp_path / "missing.env",
            base_url=None,
        )
        == DEFAULT_DEEPSEEK_BASE_URL
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            "https://token@example.test/v1",
            "must not contain credentials",
        ),
        (
            "https://example.test/v1?api_key=secret",
            "must not contain query parameters",
        ),
        (
            "https://example.test/v1#secret",
            "must not contain a URL fragment",
        ),
    ],
)
def test_planner_base_url_rejects_secret_bearing_forms(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sanitize_planner_base_url(value)


def test_cloud_preflight_reports_missing_key_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        preflight_planner_backend(
            provider="gemini",
            env_file=tmp_path / "missing.env",
            base_url=None,
            api_key_env=None,
            local_python=tmp_path / "unused-python",
        )


def test_local_worker_uses_model_chat_template_and_disables_thinking() -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class Tokenizer:
        def apply_chat_template(
            self,
            messages: object,
            **kwargs: object,
        ) -> str:
            calls.append((messages, kwargs))
            return "rendered"

    assert (
        render_chat_prompt(
            Tokenizer(),
            "prompt",
            system_instruction="system",
        )
        == "rendered"
    )
    messages, kwargs = calls[0]
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert kwargs["enable_thinking"] is False


def test_local_worker_can_enable_qwen_thinking_and_extract_final_json() -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        def apply_chat_template(
            self,
            messages: object,
            **kwargs: object,
        ) -> str:
            del messages
            calls.append(kwargs)
            return "rendered"

    render_chat_prompt(
        Tokenizer(),
        "prompt",
        system_instruction="system",
        enable_thinking=True,
    )
    assert calls[0]["enable_thinking"] is True
    assert (
        extract_final_json_object(
            '<think>Compare the attempts first.</think>\n```json\n{"ok":true}\n```'
        )
        == '{"ok":true}'
    )


def test_local_worker_folds_system_instruction_for_gemma_templates() -> None:
    calls: list[object] = []

    class GemmaTokenizer:
        def apply_chat_template(
            self,
            messages: object,
            **kwargs: object,
        ) -> str:
            del kwargs
            calls.append(messages)
            if len(messages) > 1:  # type: ignore[arg-type]
                raise ValueError("System role is not supported")
            return "gemma-rendered"

    assert (
        render_chat_prompt(
            GemmaTokenizer(),
            "classify",
            system_instruction="return JSON",
        )
        == "gemma-rendered"
    )
    assert calls[-1] == [
        {
            "role": "user",
            "content": "return JSON\n\nclassify",
        }
    ]


@pytest.mark.parametrize(
    ("parser_factory", "mode_argument", "input_arguments"),
    [
        (
            build_streaming_parser,
            "--stream-plan",
            ["--transcript", "words.json"],
        ),
        (
            build_final_parser,
            "--render-plan",
            ["--audio", "audio.wav", "--plan", "plan.json"],
        ),
    ],
)
def test_planner_clis_expose_same_provider_neutral_options(
    parser_factory: object,
    mode_argument: str,
    input_arguments: list[str],
    tmp_path: Path,
) -> None:
    parser = parser_factory()  # type: ignore[operator]
    args = parser.parse_args(
        [
            mode_argument,
            *input_arguments,
            "--output-dir",
            str(tmp_path / "out"),
            "--planner-backend",
            "deepseek",
            "--planner-model",
            "deepseek-v4-flash",
            "--planner-base-url",
            "https://example.test/v1",
            "--planner-api-key-env",
            "CUSTOM_KEY",
            "--planner-python",
            str(tmp_path / "python"),
        ]
    )
    assert args.planner_backend == "deepseek"
    assert args.planner_model == "deepseek-v4-flash"
    assert args.planner_base_url == "https://example.test/v1"
    assert args.planner_api_key_env == "CUSTOM_KEY"
    assert args.planner_python == tmp_path / "python"
