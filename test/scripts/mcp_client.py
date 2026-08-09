"""MCP 서버 직접 호출용 최소 클라이언트.

용도: `.mcp.json`에 등록된 MCP 서버(arxiv=stdio, huggingface=streamable HTTP)를
Claude Code/Codex 세션 밖에서도 호출한다. MCP 서버는 에이전트 세션 **시작 시점**에만
로드되므로, 등록 직후 같은 세션에서 바로 쓰려면 이 스크립트를 경유한다.

사용 예:
    python test/scripts/mcp_client.py arxiv tools
    python test/scripts/mcp_client.py arxiv search_papers '{"query": "volume volatility", "max_results": 5}'
    python test/scripts/mcp_client.py huggingface hub_repo_search '{"query": "garch", "type": "model"}'
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / ".mcp.json"
PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "quant-mcp-client", "version": "1.0"}


def _load_server(name: str) -> dict:
    servers = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["mcpServers"]
    if name not in servers:
        raise SystemExit(f"'{name}' 서버가 {CONFIG_PATH}에 없습니다. 등록된 서버: {list(servers)}")
    return servers[name]


def _init_params() -> dict:
    return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO}


def _call_stdio(server: dict, method: str, params: dict, timeout: int) -> dict:
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    env.update(server.get("env") or {})

    proc = subprocess.Popen(
        [server["command"], *server.get("args", [])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    def write(message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def read_until(request_id: int) -> dict:
        """stdin을 닫지 않고 응답을 한 줄씩 읽는다.

        모든 요청을 한 번에 써서 stdin을 곧바로 닫으면 서버가 처리 전에 EOF로 종료해
        빈 응답이 온다 — 그래서 요청/응답을 왕복시킨다.
        """
        while True:
            line = proc.stdout.readline()
            if not line:
                raise SystemExit(f"응답 없음(서버 조기 종료). stderr={proc.stderr.read()[:1000]}")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                return message

    try:
        write({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": _init_params()})
        read_until(1)
        write({"jsonrpc": "2.0", "method": "notifications/initialized"})
        write({"jsonrpc": "2.0", "id": 2, "method": method, "params": params})
        return read_until(2)
    finally:
        proc.kill()
        proc.wait()


def _http_post(url: str, body: dict, session_id: str | None, timeout: int) -> tuple[str | None, str]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json, text/event-stream")
    if session_id:
        request.add_header("Mcp-Session-Id", session_id)
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface" in url:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.headers.get("Mcp-Session-Id"), response.read().decode()


def _decode_sse(raw: str) -> dict:
    """streamable HTTP 응답은 순수 JSON 또는 SSE(`data:` 줄) 둘 다로 온다."""
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    merged = "".join(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    return json.loads(merged)


def _call_http(server: dict, method: str, params: dict, timeout: int) -> dict:
    url = server["url"]
    session_id, _ = _http_post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": _init_params()}, None, timeout)
    try:
        _http_post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id, timeout)
    except Exception:
        pass  # 서버에 따라 notification에 204/빈 응답을 주고 끊는다
    _, raw = _http_post(url, {"jsonrpc": "2.0", "id": 2, "method": method, "params": params}, session_id, timeout)
    return _decode_sse(raw)


def call(server_name: str, method: str, params: dict | None = None, timeout: int = 120) -> dict:
    server = _load_server(server_name)
    params = params or {}
    if server.get("type") == "http" or "url" in server:
        return _call_http(server, method, params, timeout)
    return _call_stdio(server, method, params, timeout)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    server_name, tool = argv[0], argv[1]
    arguments = json.loads(argv[2]) if len(argv) > 2 else {}

    if tool == "tools":
        response = call(server_name, "tools/list", {})
        for item in response.get("result", {}).get("tools", []):
            print(f"- {item['name']}: {(item.get('description') or '').splitlines()[0][:120]}")
        return

    response = call(server_name, "tools/call", {"name": tool, "arguments": arguments})
    if "error" in response:
        raise SystemExit(json.dumps(response["error"], ensure_ascii=False, indent=2))
    for block in response.get("result", {}).get("content", []):
        print(block.get("text", "") if block.get("type") == "text" else json.dumps(block, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
