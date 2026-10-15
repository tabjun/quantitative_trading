"""로컬 LLM 서버(llama.cpp `llama-server`)를 호출할 때만 띄우고 끝나면 내리는 실행기.

연구 트랙과 GPU 한 장을 나눠 쓰므로 모델을 상주시키지 않는다(docs/service_design.md §7).
띄우기 전에 AGENTS.md 2.14 여유선(GPU 4GB, 사용 가능 RAM 4.5GB)과 서비스 예산을 확인하고,
끝나면 자기가 띄운 PID만 종료한다(`pgrep -f` 패턴 종료는 셸 자신과 맞을 수 있어 쓰지 않는다).

경로는 환경변수로 바꿀 수 있다.
    LLAMA_CPP_DIR   llama-server와 CUDA 런타임이 있는 디렉터리
    LLM_MODEL_PATH  GGUF 모델 파일
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

HOME = Path.home()
DEFAULT_LLAMA_CPP_DIR = HOME / ".local/opt/llama.cpp-b11630"
DEFAULT_MODEL_PATH = HOME / ".local/share/llm_models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"

GPU_RESERVE_MIB = 4096  # AGENTS.md 2.14: GPU 여유 4GB
RAM_FLOOR_MIB = 4608  # AGENTS.md 2.14: 사용 가능 RAM 하한 4.5GB


@dataclass(frozen=True)
class LlmServerConfig:
    model_path: Path = Path(os.environ.get("LLM_MODEL_PATH", DEFAULT_MODEL_PATH))
    llama_cpp_dir: Path = Path(os.environ.get("LLAMA_CPP_DIR", DEFAULT_LLAMA_CPP_DIR))
    port: int = 18080
    ctx_per_slot: int = 8192
    parallel: int = 2
    threads: int = 8  # 2.14 서비스 몫 CPU 8스레드
    gpu_budget_mib: int = 4500  # 0단계 실측 3,760MiB(Q4_K_M, 8192×2)
    ram_budget_mib: int = 1500  # 0단계 실측 RSS 0.94GB
    startup_timeout_s: int = 180
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def gpu_free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    return total - used


def ram_available_mib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("/proc/meminfo에 MemAvailable이 없다")


def check_headroom(cfg: LlmServerConfig) -> dict[str, int]:
    """띄워도 2.14 여유선이 지켜지는지 확인한다. 부족하면 RuntimeError."""
    gpu, ram = gpu_free_mib(), ram_available_mib()
    need_gpu, need_ram = cfg.gpu_budget_mib + GPU_RESERVE_MIB, cfg.ram_budget_mib + RAM_FLOOR_MIB
    if gpu < need_gpu or ram < need_ram:
        raise RuntimeError(f"자원 부족: GPU 여유 {gpu}MiB(필요 {need_gpu}), RAM 여유 {ram}MiB(필요 {need_ram})")
    return {"gpu_free_mib": gpu, "ram_available_mib": ram}


@contextmanager
def llm_server(cfg: LlmServerConfig | None = None, log_path: Path | None = None) -> Iterator[LlmServerConfig]:
    """`with llm_server() as cfg:` 안에서만 서버가 떠 있다."""
    cfg = cfg or LlmServerConfig()
    check_headroom(cfg)
    env = dict(os.environ, LD_LIBRARY_PATH=str(cfg.llama_cpp_dir), OMP_NUM_THREADS="4")
    cmd = [
        str(cfg.llama_cpp_dir / "llama-server"), "-m", str(cfg.model_path),
        "--host", "127.0.0.1", "--port", str(cfg.port), "-ngl", "99",
        "-c", str(cfg.ctx_per_slot * cfg.parallel), "-np", str(cfg.parallel), "-t", str(cfg.threads),
        "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}', *cfg.extra_args,
    ]
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        deadline = time.time() + cfg.startup_timeout_s
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server가 시작 중 종료됐다(코드 {proc.returncode})")
            try:
                with urllib.request.urlopen(cfg.url + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError("llama-server 시작 시간 초과")
            time.sleep(1)
        yield cfg
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log_path:
            log.close()


def post_json(cfg: LlmServerConfig, path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(cfg.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def option_probs(cfg: LlmServerConfig, messages: list[dict], letters: tuple[str, ...]) -> tuple[dict[str, float], float]:
    """다음 1토큰의 상위 확률에서 보기 글자만 모아 정규화한다. (보기별 확률, 보기에 실린 질량)을 돌려준다."""
    d = post_json(cfg, "/v1/chat/completions", {
        "messages": messages, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 20, "cache_prompt": False,
    })
    raw: dict[str, float] = {}
    for t in d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        k = t["token"].strip()
        if k in letters:
            raw[k] = raw.get(k, 0.0) + math.exp(t["logprob"])
    mass = sum(raw.values())
    if mass == 0:
        raise RuntimeError("상위 확률에 보기 글자가 없다")
    return {k: raw.get(k, 0.0) / mass for k in letters}, mass
