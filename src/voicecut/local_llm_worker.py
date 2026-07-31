#!/usr/bin/env python3
"""Long-lived JSON-lines worker for local MLX-LM chat models."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any


DEFAULT_SYSTEM_INSTRUCTION = (
    "Return only valid JSON matching the user's requested schema."
)


def render_chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    system_instruction: str,
    enable_thinking: bool = False,
) -> str:
    """Render a portable system/user chat prompt for Qwen, Gemma, and peers."""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        return system_instruction + "\n\n" + prompt

    def apply(messages_value: list[dict[str, str]]) -> str:
        try:
            return apply_template(
                messages_value,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return apply_template(
                messages_value,
                tokenize=False,
                add_generation_prompt=True,
            )

    try:
        return apply(messages)
    except Exception as error:
        # Gemma-family templates commonly reject a distinct system role.
        # Preserve the instruction by folding it into the first user turn.
        message = str(error).casefold()
        if "system" not in message or "role" not in message:
            raise
        return apply(
            [
                {
                    "role": "user",
                    "content": system_instruction + "\n\n" + prompt,
                }
            ]
        )


def extract_final_json_object(text: str) -> str:
    """Return the final JSON object after optional local-model reasoning."""

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return stripped

    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(text[index : index + consumed])
    return candidates[-1] if candidates else stripped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load one MLX-LM-compatible local model and answer JSON-lines "
            "planning requests."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Keep stdout reserved for the machine-readable JSON-lines protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        normalized_model_name = args.model.casefold()
        tokenizer_config = (
            {"fix_mistral_regex": True}
            if any(
                family in normalized_model_name for family in ("mistral", "ministral")
            )
            else None
        )
        model, tokenizer = load(
            args.model,
            tokenizer_config=tokenizer_config,
        )
        sampler = make_sampler(temp=0.0)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            prompt = request.get("prompt")
            request_id = request.get("request_id")
            system_instruction = request.get(
                "system_instruction",
                DEFAULT_SYSTEM_INSTRUCTION,
            )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("request.prompt must be a non-empty string")
            if (
                not isinstance(system_instruction, str)
                or not system_instruction.strip()
            ):
                raise ValueError(
                    "request.system_instruction must be a non-empty string"
                )
            rendered = render_chat_prompt(
                tokenizer,
                prompt,
                system_instruction=system_instruction,
                enable_thinking=args.enable_thinking,
            )
            with contextlib.redirect_stdout(sys.stderr):
                output = generate(
                    model,
                    tokenizer,
                    prompt=rendered,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            output = extract_final_json_object(output)
            response = {
                "ok": True,
                "request_id": request_id,
                "output": output,
            }
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {
                "ok": False,
                "request_id": request_id,
                "error": f"{type(error).__name__}: {error}",
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
