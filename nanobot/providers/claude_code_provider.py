"""Claude Code OAuth Provider — uses Claude Pro/Max subscription tokens."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_TOKEN_ENDPOINT = "https://platform.claude.com/v1/oauth/token"
_MESSAGES_URL = "https://api.anthropic.com/v1/messages?beta=true"
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

_SYSTEM_PREFIX = (
    "You are Claude Code, Anthropic's official CLI for Claude, "
    "running as a backend agent.\n\n"
)

_BILLING_HEADER = "x-anthropic-billing-header: cc_version=2.1.77; cc_entrypoint=cli; cch=00000;"

_BETA_FLAGS = ",".join([
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "claude-code-20250219",
])


class ClaudeCodeProvider(LLMProvider):
    """Use Claude Pro/Max OAuth tokens to call the Anthropic Messages API."""

    def __init__(
        self,
        default_model: str = "claude-code/claude-sonnet-4-5-20250514",
        api_key: str | None = None,
        credentials_path: str | None = None,
    ):
        super().__init__(api_key=api_key, api_base=None)
        self.default_model = default_model
        self._cred_path = Path(credentials_path or _CREDENTIALS_PATH).expanduser()
        self._credentials: dict | None = None
        self._refresh_lock = asyncio.Lock()

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _load_credentials(self) -> dict:
        """Read and cache the claudeAiOauth block from credentials file."""
        if self._credentials:
            return self._credentials
        if not self._cred_path.exists():
            raise FileNotFoundError(
                f"Claude credentials not found at {self._cred_path}. "
                "Run 'claude setup-token' to configure."
            )
        data = json.loads(self._cred_path.read_text())
        oauth = data.get("claudeAiOauth")
        if not oauth or not oauth.get("accessToken"):
            raise ValueError("No accessToken in Claude credentials file.")
        self._credentials = oauth
        return oauth

    async def _get_access_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        # Direct token from config takes priority (no refresh support)
        if self.api_key:
            return self.api_key
        creds = self._load_credentials()
        expires_at = creds.get("expiresAt", 0)
        now_ms = int(time.time() * 1000)
        if now_ms < expires_at - 60_000:
            return creds["accessToken"]
        # Token expired or about to expire
        async with self._refresh_lock:
            # Double-check after acquiring lock
            creds = self._credentials or self._load_credentials()
            if int(time.time() * 1000) < creds.get("expiresAt", 0) - 60_000:
                return creds["accessToken"]
            await self._refresh_token(creds)
            return self._credentials["accessToken"]

    async def _refresh_token(self, creds: dict) -> None:
        """Exchange refresh token for new access + refresh tokens."""
        refresh_token = creds.get("refreshToken")
        if not refresh_token:
            raise ValueError("No refreshToken in credentials — re-run 'claude setup-token'.")
        logger.info("Refreshing Claude OAuth token...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(_TOKEN_ENDPOINT, json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _CLIENT_ID,
            })
            if resp.status_code != 200:
                raise RuntimeError(f"Token refresh failed ({resp.status_code}): {resp.text}")
            data = resp.json()

        new_creds = {
            **creds,
            "accessToken": data["access_token"],
            "refreshToken": data["refresh_token"],
            "expiresAt": int(time.time() * 1000) + data.get("expires_in", 28800) * 1000,
        }
        self._credentials = new_creds

        # Atomic write-back (refresh tokens are single-use)
        full = {"claudeAiOauth": new_creds}
        tmp = self._cred_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(full, indent=2))
        os.replace(tmp, self._cred_path)
        logger.info("Claude OAuth token refreshed, expires in {}h", data.get("expires_in", 0) // 3600)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_headers(access_token: str) -> dict[str, str]:
        sys_name = platform.system()
        os_name = "MacOS" if sys_name == "Darwin" else sys_name
        return {
            "Authorization": f"Bearer {access_token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": _BETA_FLAGS,
            "user-agent": "claude-cli/2.1.2 (external, cli)",
            "x-app": "cli",
            "anthropic-dangerous-direct-browser-access": "true",
            "content-type": "application/json",
            "x-stainless-lang": "python",
            "x-stainless-os": os_name,
            "x-stainless-arch": platform.machine(),
            "x-stainless-runtime": "CPython",
            "x-stainless-runtime-version": platform.python_version(),
            "x-stainless-package-version": "0.52.0",
            "x-stainless-retry-count": "0",
            "x-stainless-timeout": "600",
        }

    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        if model.startswith(("claude-code/", "claude_code/")):
            return model.split("/", 1)[1]
        return model

    # ------------------------------------------------------------------
    # Message format conversion (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_content_to_anthropic(content: Any) -> list[dict]:
        """Convert OpenAI content (str or list) to Anthropic content blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, list):
            blocks = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                    if m:
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": m.group(1),
                                "data": m.group(2),
                            },
                        })
            return blocks
        return []

    @classmethod
    def _convert_messages(cls, messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract system prompt and convert messages to Anthropic format."""
        system_parts: list[str] = []
        anthropic_msgs: list[dict] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                text = msg["content"] if isinstance(msg["content"], str) else ""
                system_parts.append(text)
                continue

            if role == "user":
                blocks = cls._convert_content_to_anthropic(msg.get("content", ""))
                if blocks:
                    anthropic_msgs.append({"role": "user", "content": blocks})
                continue

            if role == "assistant":
                blocks = []
                content = msg.get("content")
                if content:
                    blocks.extend(cls._convert_content_to_anthropic(content))
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": f"mcp_{name}",
                        "input": args,
                    })
                if blocks:
                    anthropic_msgs.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                anthropic_msgs.append({"role": "user", "content": [tool_result]})
                continue

        # Merge consecutive same-role messages (Anthropic requires alternation)
        merged: list[dict] = []
        for m in anthropic_msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"].extend(m["content"])
            else:
                merged.append(m)

        system = _SYSTEM_PREFIX + "\n\n".join(system_parts)
        return system, merged

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool definitions to Anthropic format with mcp_ prefix."""
        result = []
        for tool in tools:
            fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
            name = fn.get("name")
            if not name:
                continue
            result.append({
                "name": f"mcp_{name}",
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return result

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        resolved = self._strip_model_prefix(model)

        token = await self._get_access_token()
        headers = self._build_headers(token)
        system, anthropic_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools or [])

        system_blocks = [
            {"type": "text", "text": _BILLING_HEADER},
            {"type": "text", "text": system},
        ]

        body: dict[str, Any] = {
            "model": resolved,
            "max_tokens": max(1, max_tokens),
            "system": system_blocks,
            "messages": anthropic_msgs,
            "tools": anthropic_tools,
            "stream": False,
        }

        if tool_choice is not None:
            if isinstance(tool_choice, str):
                # "auto" / "any" / "required" → {"type": ...}
                tc_type = "any" if tool_choice == "required" else tool_choice
                body["tool_choice"] = {"type": tc_type}
            elif isinstance(tool_choice, dict):
                # Convert OpenAI format to Anthropic format
                # OpenAI: {"type": "function", "function": {"name": "X"}}
                # Anthropic: {"type": "tool", "name": "mcp_X"}
                fn = tool_choice.get("function")
                if fn and isinstance(fn, dict) and fn.get("name"):
                    body["tool_choice"] = {
                        "type": "tool",
                        "name": f"mcp_{fn['name']}",
                    }
                else:
                    body["tool_choice"] = tool_choice

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(_MESSAGES_URL, headers=headers, json=body)
                if resp.status_code != 200:
                    error = resp.text[:500]
                    logger.error("Claude Code API error {}: {}", resp.status_code, error)
                    return LLMResponse(content=f"Error ({resp.status_code}): {error}", finish_reason="error")
                return self._parse_response(resp.json())
        except Exception as e:
            return LLMResponse(content=f"Error calling Claude Code API: {e}", finish_reason="error")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        """Parse Anthropic Messages API response into LLMResponse."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                content_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                name = block.get("name", "")
                # Strip mcp_ prefix
                if name.startswith("mcp_"):
                    name = name[4:]
                tool_calls.append(ToolCallRequest(
                    id=block.get("id", ""),
                    name=name,
                    arguments=block.get("input", {}),
                ))

        stop_reason = data.get("stop_reason", "end_turn")
        finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
        finish_reason = finish_map.get(stop_reason, "stop")

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)

        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            reasoning_content="\n".join(thinking_parts) if thinking_parts else None,
        )
