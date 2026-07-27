"""
Data-Analyst Telegram Bot
=========================
FastAPI app (health check + public JSONL log) running alongside two
background threads: a Telegram long-poll loop and a keep-alive self-pinger.

Env vars required:
  BOT_TOKEN      - Telegram bot token from @BotFather
  GEMINI_API_KEY - free Google AI Studio API key (aistudio.google.com, no card needed)
  BASE_URL       - public base URL of this deployment, e.g. https://myapp.onrender.com
  OPENAI_MODEL   - optional, defaults to "gemini-2.5-flash"

Uses Gemini's OpenAI-compatible endpoint, so the rest of the code (tool
calling, JSON parsing, etc.) stays exactly the same as if it were OpenAI.
"""

import os
import re
import json
import time
import threading
import traceback
import contextlib
import io
from datetime import datetime, timezone
from collections import defaultdict, deque

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BASE_URL = os.environ["BASE_URL"].rstrip("/")
MODEL = os.environ.get("OPENAI_MODEL", "gemini-3.5-flash")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_PATH = "/tmp/run.jsonl"
LOG_URL = f"{BASE_URL}/run.jsonl"

WALL_CLOCK_BUDGET_SECONDS = 210   # hard stop for tool use; grader timeout ~300s
MAX_TOOL_STEPS = 10
MAX_TOOL_OUTPUT_CHARS = 8000
HISTORY_TURNS_PER_CHAT = 20

# Gemini's free-tier API is OpenAI-compatible, so we just point the OpenAI
# client at Google's endpoint instead of paying for OpenAI directly.
# timeout is explicit: without it, a hung/slow upstream call can block a
# chat forever with no error and no reply (silent stall).
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0,
    max_retries=1,
)

# --------------------------------------------------------------------------
# Logging (JSONL, publicly served at /run.jsonl)
# --------------------------------------------------------------------------
_log_lock = threading.Lock()


def log_event(event: dict):
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    line = json.dumps(event, default=str)
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------
# Per-chat conversation history
# --------------------------------------------------------------------------
_chat_history = defaultdict(lambda: deque(maxlen=HISTORY_TURNS_PER_CHAT))
_chat_lock = threading.Lock()


# --------------------------------------------------------------------------
# The one tool the model gets: run_python
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code on the server and return captured stdout. "
                "Use this to fetch public datasets (requests/BeautifulSoup), "
                "load them (pandas/openpyxl), and compute the numeric/textual "
                "answer. Never guess a number that can be computed — compute it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

_EXEC_GLOBALS_TEMPLATE = {"__name__": "__main__"}


import concurrent.futures

_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
PYTHON_TOOL_TIMEOUT_SECONDS = 45


def _exec_code(code: str) -> str:
    stdout_buf = io.StringIO()
    g = dict(_EXEC_GLOBALS_TEMPLATE)
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, g)
        return stdout_buf.getvalue()
    except Exception:
        return stdout_buf.getvalue() + "\n" + traceback.format_exc()


def run_python_tool(code: str) -> str:
    """Execute untrusted-ish analyst code with a hard wall-clock timeout.

    Without this, model-generated code that does e.g. requests.get(url)
    with no timeout can hang forever on a slow/unresponsive site (common
    with government data portals), silently stalling the whole chat.
    """
    future = _TOOL_EXECUTOR.submit(_exec_code, code)
    try:
        out = future.result(timeout=PYTHON_TOOL_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        out = (
            f"[execution timed out after {PYTHON_TOOL_TIMEOUT_SECONDS}s — "
            "the request likely hung on a slow/unresponsive URL. Try a "
            "different source, add timeout= to requests calls, or answer "
            "from known public statistics instead.]"
        )
    if len(out) > MAX_TOOL_OUTPUT_CHARS:
        out = out[-MAX_TOOL_OUTPUT_CHARS:]
    return out


SYSTEM_PROMPT = """You are a data-analyst agent replying inside a Telegram bot.

Rules:
- Answer the LATEST user message. Earlier messages in this chat are context for
  a multi-turn task (e.g. data sent in an earlier message).
- Use the run_python tool to fetch and compute answers (pandas, numpy, requests,
  BeautifulSoup, openpyxl are installed). Never guess a number you could compute.
  ALWAYS pass timeout=15 (or similar) to any requests.get/post call — untimed
  requests to slow government data portals can hang. If a source is slow or
  unreachable after one retry, fall back to your own knowledge rather than
  retrying indefinitely.
- If the latest message is only a setup message (e.g. "I will send data next")
  and does not itself ask a question, reply with a small JSON acknowledgement
  in the same required shape, using your best-effort placeholder answer field.
- Your FINAL reply (after any tool calls) must be ONLY a single JSON object.
  No markdown code fences, no prose before or after it, nothing else.
- Match the exact JSON shape requested in the message: correct keys, correct
  nesting, correct types (string vs number vs list), no extra keys.
- Always include a "log_url" key in your JSON; any placeholder value you put
  there will be overwritten by the caller with the real log URL.
"""


def _extract_json(text: str) -> dict:
    """Strip code fences and find the first balanced {...} block."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError("unbalanced braces in model output")


def agent_reply(chat_id: int, user_text: str) -> dict:
    """Run the tool-calling loop and return the final answer dict."""
    deadline = time.time() + WALL_CLOCK_BUDGET_SECONDS

    with _chat_lock:
        history = list(_chat_history[chat_id])
        _chat_history[chat_id].append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    log_event({"chat_id": chat_id, "type": "incoming_message", "text": user_text})

    final_text = None
    for step in range(MAX_TOOL_STEPS):
        time_left = deadline - time.time()
        use_tools = time_left > 15  # leave headroom to force a final answer

        kwargs = dict(model=MODEL, messages=messages)
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        else:
            # Past budget: force a plain-text final answer, no more tool calls.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Time budget exceeded. Reply NOW with only the final "
                        "JSON object, best-effort, no tool calls."
                    ),
                }
            )

        log_event({"chat_id": chat_id, "type": "llm_call_start", "step": step})
        resp = client.chat.completions.create(**kwargs)
        log_event({"chat_id": chat_id, "type": "llm_call_end", "step": step})
        msg = resp.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and use_tools:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    code = args.get("code", "")
                except Exception:
                    code = ""
                result = run_python_tool(code)
                log_event(
                    {
                        "chat_id": chat_id,
                        "type": "tool_call",
                        "code": code,
                        "output": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue  # loop again with tool results appended

        # No tool call -> this is the final text answer
        final_text = msg.content or ""
        break

    if final_text is None:
        final_text = '{"answer": "internal error: no final answer produced"}'

    log_event({"chat_id": chat_id, "type": "model_final_text", "text": final_text})

    try:
        parsed = _extract_json(final_text)
    except Exception:
        parsed = {"answer": final_text.strip()}

    if "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = LOG_URL

    with _chat_lock:
        _chat_history[chat_id].append({"role": "assistant", "content": final_text})

    return parsed


# --------------------------------------------------------------------------
# Telegram helpers
# --------------------------------------------------------------------------
def tg_get_updates(offset=None, timeout=30):
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=timeout + 10)
    r.raise_for_status()
    return r.json().get("result", [])


def tg_send_message(chat_id: int, text: str):
    r = requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def handle_message(update: dict):
    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return
    chat_id = msg["chat"]["id"]
    text = msg["text"]

    try:
        answer_obj = agent_reply(chat_id, text)
    except Exception:
        err = traceback.format_exc()
        log_event({"chat_id": chat_id, "type": "handler_error", "error": err})
        answer_obj = {"answer": "internal error", "log_url": LOG_URL}

    try:
        reply_text = json.dumps(answer_obj)
    except Exception:
        reply_text = json.dumps({"answer": "internal error", "log_url": LOG_URL})

    try:
        tg_send_message(chat_id, reply_text)
    except Exception:
        log_event(
            {
                "chat_id": chat_id,
                "type": "send_error",
                "error": traceback.format_exc(),
            }
        )

    log_event({"chat_id": chat_id, "type": "reply_sent", "reply": reply_text})


def telegram_poll_loop():
    offset = None
    while True:
        try:
            updates = tg_get_updates(offset=offset, timeout=30)
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    handle_message(update)
                except Exception:
                    log_event(
                        {"type": "poll_loop_error", "error": traceback.format_exc()}
                    )
        except Exception:
            log_event(
                {"type": "poll_loop_fatal", "error": traceback.format_exc()}
            )
            time.sleep(5)


def keep_alive_loop():
    while True:
        time.sleep(600)  # 10 minutes
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl")
def run_log():
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("", media_type="application/x-ndjson")
    with open(LOG_PATH, "r") as f:
        content = f.read()
    return PlainTextResponse(content, media_type="application/x-ndjson")


@app.on_event("startup")
def on_startup():
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, "a").close()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=keep_alive_loop, daemon=True).start()
    log_event({"type": "startup", "model": MODEL})
