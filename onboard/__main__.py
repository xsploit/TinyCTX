"""
TinyCTX Onboarding Wizard
Run with: python -m onboard  (from repo root)
         python -m onboard --reset
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import questionary
import yaml
from questionary import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .extra import Config

# ── constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_FILE = Path(__file__).parent / "providers.json"
CONFIG_PATH = REPO_ROOT / "config.yaml"

BANNER = r"""
 _______ _             _____ _______ _  __
|__   __(_)           / ____|__   __| |/ /
   | |   _ _ __  _   | |       | |  | ' /  __  __
   | |  | | '_ \| | | | |       | |  |  <   \ \/ /
   | |  | | | | | |_| | |____   | |  | . \   >  <
   |_|  |_|_| |_|\__, |\_____|  |_|  |_|\_\ /_/\_\
                  __/ |
                 |___/    Onboarding Wizard
"""

DEFAULT_WORKSPACE = "~/.tinyctx"
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8080

QSTYLE = Style([
    ("qmark",        "fg:#00cfff bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00ff99 bold"),
    ("pointer",      "fg:#00cfff bold"),
    ("highlighted",  "fg:#00cfff bold"),
    ("selected",     "fg:#00ff99"),
    ("separator",    "fg:#555555"),
    ("instruction",  "fg:#888888"),
])

c = Console()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_providers() -> dict[str, str]:
    with open(PROVIDERS_FILE) as f:
        return json.load(f)


def section(title: str) -> None:
    c.print()
    c.print(Rule(f"[bold cyan]{title}[/]", style="cyan"))


def success(msg: str) -> None:
    c.print(f"[bold green]✓[/] {msg}")


def warn(msg: str) -> None:
    c.print(f"[bold yellow]![/] {msg}")


def fail(msg: str) -> None:
    c.print(f"[bold red]✗[/] {msg}")


def api_key_env_for(provider_name: str) -> str:
    """Derive the conventional env var name from the provider name."""
    return provider_name.upper().replace(" ", "_").replace("-", "_") + "_API_KEY"


def fetch_models(base_url: str, api_key_env: str, timeout: float = 6.0) -> list[str]:
    """Query GET /v1/models and return a sorted list of model IDs, or [] on failure."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    url = url + "/models"

    api_key = os.environ.get(api_key_env, "") if api_key_env != "N/A" else "ignored"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            ids = sorted(m["id"] for m in models if "id" in m)
            return ids
    except Exception:
        return []


def health_ping(host: str, port: int, timeout: float = 4.0) -> bool:
    url = f"http://{host}:{port}/v1/health"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def load_existing_config() -> dict | None:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                d = yaml.safe_load(f)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def write_config(data: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ── steps ─────────────────────────────────────────────────────────────────────

def step_detect(reset: bool) -> str:
    """Returns 'new' | 'modify' | 'reset'."""
    if not CONFIG_PATH.exists():
        return "new"

    section("Existing Configuration Detected")
    c.print(f"Found [bold]{CONFIG_PATH}[/]")

    existing = load_existing_config()
    if existing is None:
        fail("Config is invalid or unreadable.")
        c.print("Run [bold]python -m onboard --reset[/] to start fresh.")
        sys.exit(1)

    if reset:
        return "reset"

    choice = questionary.select(
        "What would you like to do?",
        choices=["Keep existing config (exit)", "Modify (update specific sections)", "Reset (start from scratch)"],
        default="Modify (update specific sections)",
        style=QSTYLE,
    ).ask()

    if choice is None or "Keep" in choice:
        success("Nothing changed.")
        sys.exit(0)
    if "Reset" in choice:
        return "reset"
    return "modify"


def step_llm(providers: dict[str, str]) -> dict[str, Any]:
    section("Step 1 — LLM Provider & Model")

    provider_name = questionary.select(
        "Which provider will power this agent?",
        choices=list(providers.keys()),
        style=QSTYLE,
    ).ask()
    if provider_name is None:
        sys.exit(0)

    base_url = providers[provider_name]

    # Derive env var; local providers get N/A
    local_providers = {"Ollama", "LMStudio", "vLLM", "llama-cpp"}
    if provider_name in local_providers:
        api_key_env = "N/A"
    else:
        api_key_env = api_key_env_for(provider_name)
        if not os.environ.get(api_key_env):
            warn(f"{api_key_env} is not set — set it before starting the agent.")

    # Try to list models from the endpoint
    c.print(f"  Querying {base_url}/models …", end=" ")
    model_ids = fetch_models(base_url, api_key_env)

    if model_ids:
        c.print(f"[green]{len(model_ids)} models found[/]")
        model = questionary.select(
            "Select a model",
            choices=model_ids,
            style=QSTYLE,
        ).ask()
        if model is None:
            sys.exit(0)
    else:
        c.print("[yellow]could not reach endpoint[/]")
        model = questionary.text(
            "Enter model name manually",
            style=QSTYLE,
        ).ask()
        if not model:
            sys.exit(0)

    max_tokens = int(questionary.text("max_tokens", default="4096", style=QSTYLE).ask() or "4096")
    temperature = float(questionary.text("temperature", default="0.7", style=QSTYLE).ask() or "0.7")

    success(f"Primary: [bold]{model}[/] via {provider_name}")
    return {
        "base_url": base_url,
        "model": model,
        "api_key_env": api_key_env,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def step_embedding(providers: dict[str, str]) -> dict[str, Any] | None:
    section("Step 2 — Embedding Model (optional)")
    c.print("Enables hybrid BM25+vector memory search. Skip for BM25-only mode.\n")

    if not questionary.confirm("Configure an embedding model?", default=True, style=QSTYLE).ask():
        warn("BM25-only memory search will be used.")
        return None

    provider_name = questionary.select(
        "Embedding provider",
        choices=list(providers.keys()),
        style=QSTYLE,
    ).ask()
    if provider_name is None:
        sys.exit(0)

    base_url = providers[provider_name]
    local_providers = {"Ollama", "LMStudio", "vLLM", "llama-cpp"}
    api_key_env = "N/A" if provider_name in local_providers else api_key_env_for(provider_name)

    c.print(f"  Querying {base_url}/models …", end=" ")
    model_ids = fetch_models(base_url, api_key_env)

    # Filter to likely embedding models if we got a big list
    embed_hints = ("embed", "nomic", "mxbai", "e5-", "bge-", "voyage", "gte-", "minilm", "all-")
    embed_ids = [m for m in model_ids if any(h in m.lower() for h in embed_hints)] or model_ids

    if embed_ids:
        c.print(f"[green]{len(model_ids)} models found ({len(embed_ids)} likely embedding)[/]")
        model = questionary.select(
            "Select embedding model",
            choices=embed_ids if embed_ids else model_ids,
            style=QSTYLE,
        ).ask()
        if model is None:
            sys.exit(0)
    else:
        c.print("[yellow]could not reach endpoint[/]")
        model = questionary.text(
            "Enter embedding model name manually",
            default="nomic-embed-text",
            style=QSTYLE,
        ).ask()
        if not model:
            sys.exit(0)

    success(f"Embedding: [bold]{model}[/] via {provider_name}")
    return {
        "kind": "embedding",
        "base_url": base_url,
        "api_key_env": api_key_env,
        "model": model,
    }


def step_workspace() -> str:
    section("Step 3 — Workspace")
    c.print("Stores sessions, memory index, SOUL.md, AGENTS.md, etc.\n")

    workspace = questionary.text(
        "Workspace path",
        default=DEFAULT_WORKSPACE,
        style=QSTYLE,
    ).ask()
    if not workspace:
        workspace = DEFAULT_WORKSPACE

    Path(workspace).expanduser().mkdir(parents=True, exist_ok=True)
    success(f"Workspace: [bold]{Path(workspace).expanduser()}[/]")
    return workspace


def step_gateway() -> dict[str, Any]:
    section("Step 4 — Gateway (HTTP/SSE API)")
    c.print("Exposes TinyCTX to SillyTavern, curl, and other external clients.\n")

    enabled = questionary.confirm("Enable the gateway?", default=True, style=QSTYLE).ask()

    if not enabled:
        warn("Gateway disabled.")
        return {"enabled": False, "host": DEFAULT_GATEWAY_HOST, "port": DEFAULT_GATEWAY_PORT, "api_key": ""}

    host = questionary.text("Bind host", default=DEFAULT_GATEWAY_HOST, style=QSTYLE).ask() or DEFAULT_GATEWAY_HOST
    port = int(questionary.text("Port", default=str(DEFAULT_GATEWAY_PORT), style=QSTYLE).ask() or DEFAULT_GATEWAY_PORT)

    import secrets
    generated = secrets.token_hex(16)
    raw = questionary.text(f"API key (blank = auto-generate)", default="", style=QSTYLE).ask()
    api_key = raw.strip() or generated

    success(f"Gateway: http://{host}:{port}  key=[bold]{api_key}[/]")
    return {"enabled": True, "host": host, "port": port, "api_key": api_key}


def step_bridges() -> dict[str, Any]:
    section("Step 5 — Bridges")
    c.print("CLI is always enabled. Select additional bridges:\n")

    choices = questionary.checkbox(
        "Bridges to enable",
        choices=["Discord", "Matrix"],
        style=QSTYLE,
    ).ask() or []

    bridges: dict[str, Any] = {"cli": {"enabled": True}}

    if "Discord" in choices:
        c.print("\n[bold]Discord[/] — set [bold]DISCORD_BOT_TOKEN[/] in your environment.")
        c.print("  Bot setup: https://discord.com/developers/applications")
        c.print("  Required intents: Message Content, Server Members")
        bridges["discord"] = {
            "enabled": True,
            "options": {
                "token_env": "DISCORD_BOT_TOKEN",
                "allowed_users": [],
                "dm_enabled": True,
                "guild_ids": [],
                "prefix_required": True,
                "command_prefix": "!",
                "max_reply_length": 1900,
                "typing_indicator": True,
            },
        }
        if os.environ.get("DISCORD_BOT_TOKEN"):
            success("DISCORD_BOT_TOKEN found in environment.")
        else:
            warn("DISCORD_BOT_TOKEN not set.")

    if "Matrix" in choices:
        c.print("\n[bold]Matrix[/] — set [bold]MATRIX_PASSWORD[/] in your environment.\n")
        homeserver = questionary.text("Homeserver URL", default="https://matrix.org", style=QSTYLE).ask() or "https://matrix.org"
        username = questionary.text("Bot MXID (@bot:server.tld)", default="@yourbot:matrix.org", style=QSTYLE).ask() or "@yourbot:matrix.org"
        bridges["matrix"] = {
            "enabled": True,
            "options": {
                "homeserver": homeserver,
                "username": username,
                "password_env": "MATRIX_PASSWORD",
                "device_name": "TinyCTX",
                "store_path": "matrix_store",
                "allowed_users": [],
                "dm_enabled": True,
                "room_ids": [],
                "prefix_required": True,
                "command_prefix": "!",
                "max_reply_length": 16000,
                "sync_timeout_ms": 30000,
            },
        }
        if os.environ.get("MATRIX_PASSWORD"):
            success("MATRIX_PASSWORD found in environment.")
        else:
            warn("MATRIX_PASSWORD not set.")

    return bridges


def step_health_check(gateway: dict[str, Any]) -> None:
    if not gateway.get("enabled"):
        return
    section("Step 6 — Health Check")
    host, port = gateway["host"], gateway["port"]
    c.print(f"  Pinging http://{host}:{port}/v1/health …", end=" ")
    if health_ping(host, port):
        success("Gateway is up!")
    else:
        warn("No response — start the agent with [bold]python -m main[/] and verify.")


def step_summary() -> None:
    section("Done!")
    c.print(Panel(
        Markdown(f"""
**Config written to:** `{CONFIG_PATH}`

**Next steps:**

1. Set any missing environment variables (API keys, bot tokens).
2. Start the agent:
   ```
   python -m main
   ```
3. To reconfigure: re-run `python -m onboard`
4. To edit advanced options: edit `config.yaml` directly  
   (see `example.config.yaml` for all knobs).
"""),
        title="[bold green]TinyCTX is ready[/]",
        border_style="green",
    ))


# ── config assembly ───────────────────────────────────────────────────────────

def assemble_config(
    model_cfg: dict,
    embed_cfg: dict | None,
    workspace: str,
    gateway: dict,
    bridges: dict,
    existing: dict | None,
) -> dict:
    base = existing or {}

    models = base.get("models", {})
    models["primary"] = model_cfg
    if embed_cfg:
        models["embed"] = embed_cfg
    base["models"] = models

    base["llm"] = base.get("llm") or {
        "primary": "primary",
        "fallback": [],
        "fallback_on": {"any_error": False, "http_codes": [429, 500, 502, 503, 504]},
    }

    base.setdefault("context", 16384)
    base["workspace"] = {"path": workspace}
    base["gateway"] = gateway

    # Merge bridges — preserve hand-edited options for bridges not touched
    existing_bridges = base.get("bridges", {})
    for name, bcfg in bridges.items():
        existing_bridges[name] = bcfg
    base["bridges"] = existing_bridges

    # Memory: always auto_inject; wire embed key if present
    mem = base.get("memory_search", {})
    mem["auto_inject"] = True
    if embed_cfg:
        mem["embedding_model"] = "embed"
    elif "embedding_model" not in mem:
        pass  # leave unset → BM25-only
    base["memory_search"] = mem

    base.setdefault("logging", {"level": "INFO"})
    base.setdefault("max_tool_cycles", 10)

    return base


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python -m onboard", description="TinyCTX Onboarding Wizard")
    p.add_argument("--reset", action="store_true", help="Wipe existing config and start fresh")
    args = p.parse_args()

    providers = load_providers()
    existing = load_existing_config()

    c.print(f"[bold cyan]{BANNER}[/]")

    # 0. detect
    mode = step_detect(args.reset)
    if mode == "reset":
        CONFIG_PATH.unlink(missing_ok=True)
        existing = None
        success("Config reset.")
    elif mode == "new":
        c.print(Panel("[bold]Welcome![/] Let's get TinyCTX configured.", border_style="cyan"))

    # 1–5 steps
    model_cfg  = step_llm(providers)
    embed_cfg  = step_embedding(providers)
    workspace  = step_workspace()
    gateway    = step_gateway()
    bridges    = step_bridges()

    # assemble + write
    data = assemble_config(model_cfg, embed_cfg, workspace, gateway, bridges, existing)
    write_config(data)
    success(f"Config written to [bold]{CONFIG_PATH}[/]")

    step_health_check(gateway)
    step_summary()


if __name__ == "__main__":
    main()
