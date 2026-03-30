"""
TinyCTX Onboarding Wizard
Run with: python -m onboard  (from repo root)
         python -m onboard --reset
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
import zipfile
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

REPO_ROOT    = Path(__file__).resolve().parent.parent
BUNDLED_DIR  = Path(__file__).parent / "bundled"
PROVIDERS_FILE = Path(__file__).parent / "providers.json"
CONFIG_PATH  = REPO_ROOT / "config.yaml"

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

DEFAULT_WORKSPACE    = "~/.tinyctx"
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8080

# Markdown files the wizard will copy from bundled/ if missing in workspace
BUNDLED_MD = ["SOUL.md", "AGENTS.md", "MEMORY.md"]

QSTYLE = Style([
    ("qmark",       "fg:#00cfff bold"),
    ("question",    "bold"),
    ("answer",      "fg:#00ff99 bold"),
    ("pointer",     "fg:#00cfff bold"),
    ("highlighted", "fg:#00cfff bold"),
    ("selected",    "fg:#00ff99"),
    ("separator",   "fg:#555555"),
    ("instruction", "fg:#888888"),
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


def api_key_env_for(provider_name: str) -> str:
    return provider_name.upper().replace(" ", "_").replace("-", "_") + "_API_KEY"


LOCAL_PROVIDERS = {"Ollama", "LMStudio", "vLLM", "llama-cpp", "Custom (local)"}


def fetch_models(base_url: str, api_key_env: str, timeout: float = 6.0) -> list[str]:
    """Query GET /v1/models. Returns sorted model ID list, or [] on failure."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"

    api_key = os.environ.get(api_key_env, "") if api_key_env != "N/A" else "ignored"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return sorted(m["id"] for m in data.get("data", []) if "id" in m)
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


def pick_model(base_url: str, api_key_env: str, label: str = "model") -> str:
    """
    Try to list models from the endpoint. If successful, show arrow-key select.
    Fall back to free-text entry if the endpoint is unreachable.
    For embedding step, float likely embedding models to the top.
    """
    c.print(f"  Querying {base_url}/models …", end=" ")
    model_ids = fetch_models(base_url, api_key_env)

    if not model_ids:
        c.print("[yellow]could not reach endpoint[/]")
        answer = questionary.text(f"Enter {label} name manually", style=QSTYLE).ask()
        return answer or ""

    if "embed" in label.lower():
        hints = ("embed", "nomic", "mxbai", "e5-", "bge-", "voyage", "gte-", "minilm", "all-")
        priority = [m for m in model_ids if any(h in m.lower() for h in hints)]
        rest     = [m for m in model_ids if m not in priority]
        ordered  = priority + rest
        c.print(f"[green]{len(model_ids)} models ({len(priority)} likely embedding)[/]")
    else:
        ordered = model_ids
        c.print(f"[green]{len(model_ids)} models found[/]")

    answer = questionary.select(f"Select {label}", choices=ordered, style=QSTYLE).ask()
    return answer or ""


# ── steps ─────────────────────────────────────────────────────────────────────

def step_detect(reset: bool) -> str:
    """Returns 'new' | 'modify' | 'reset'."""
    if not CONFIG_PATH.exists():
        return "new"

    section("Existing Configuration Detected")
    c.print(f"Found [bold]{CONFIG_PATH}[/]")

    if load_existing_config() is None:
        c.print("[bold red]✗[/] Config is invalid — run [bold]python -m onboard --reset[/] to start fresh.")
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
    return "reset" if "Reset" in choice else "modify"


def _configure_provider(providers: dict[str, str], label: str) -> tuple[str, str, str]:
    """
    Shared logic: pick provider (with Custom at top), resolve base_url and api_key_env.
    Returns (base_url, api_key_env, provider_name).
    """
    provider_choices = ["Custom"] + list(providers.keys())

    provider_name = questionary.select(
        f"{label} provider",
        choices=provider_choices,
        style=QSTYLE,
    ).ask()
    if provider_name is None:
        sys.exit(0)

    if provider_name == "Custom":
        base_url = questionary.text(
            "Base URL (e.g. http://localhost:8000/v1)",
            style=QSTYLE,
        ).ask() or ""
        use_key = questionary.confirm("Does this endpoint require an API key?", default=False, style=QSTYLE).ask()
        if use_key:
            api_key_env = questionary.text("Env var name for API key", default="CUSTOM_API_KEY", style=QSTYLE).ask() or "CUSTOM_API_KEY"
        else:
            api_key_env = "N/A"
    else:
        base_url    = providers[provider_name]
        api_key_env = "N/A" if provider_name in LOCAL_PROVIDERS else api_key_env_for(provider_name)
        if api_key_env != "N/A" and not os.environ.get(api_key_env):
            warn(f"{api_key_env} is not set — set it before starting the agent.")

    return base_url, api_key_env, provider_name


def step_llm(providers: dict[str, str]) -> dict[str, Any]:
    section("Step 1 — LLM Provider & Model")

    base_url, api_key_env, provider_name = _configure_provider(providers, "LLM")

    model = pick_model(base_url, api_key_env, label="model")
    if not model:
        sys.exit(0)

    max_tokens  = int(questionary.text("max_tokens",  default="4096", style=QSTYLE).ask() or "4096")
    temperature = float(questionary.text("temperature", default="0.7",  style=QSTYLE).ask() or "0.7")

    success(f"Primary: [bold]{model}[/] via {provider_name}")
    return {
        "base_url":    base_url,
        "model":       model,
        "api_key_env": api_key_env,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }


def step_embedding(providers: dict[str, str]) -> dict[str, Any] | None:
    section("Step 2 — Embedding Model (optional)")
    c.print("Enables hybrid BM25+vector memory search. Skip for BM25-only mode.\n")

    if not questionary.confirm("Configure an embedding model?", default=True, style=QSTYLE).ask():
        warn("BM25-only memory search will be used.")
        return None

    base_url, api_key_env, provider_name = _configure_provider(providers, "Embedding")

    model = pick_model(base_url, api_key_env, label="embedding model")
    if not model:
        sys.exit(0)

    success(f"Embedding: [bold]{model}[/] via {provider_name}")
    return {
        "kind":        "embedding",
        "base_url":    base_url,
        "api_key_env": api_key_env,
        "model":       model,
    }


def step_workspace() -> str:
    section("Step 3 — Workspace")
    c.print("Stores sessions, memory index, SOUL.md, AGENTS.md, etc.\n")

    workspace = questionary.text("Workspace path", default=DEFAULT_WORKSPACE, style=QSTYLE).ask() or DEFAULT_WORKSPACE
    Path(workspace).expanduser().mkdir(parents=True, exist_ok=True)
    success(f"Workspace: [bold]{Path(workspace).expanduser()}[/]")
    return workspace


def step_gateway() -> dict[str, Any]:
    section("Step 4 — Gateway (HTTP/SSE API)")
    c.print("Exposes TinyCTX to SillyTavern, curl, and other external clients.\n")

    host = questionary.text("Bind host", default=DEFAULT_GATEWAY_HOST, style=QSTYLE).ask() or DEFAULT_GATEWAY_HOST
    port = int(questionary.text("Port", default=str(DEFAULT_GATEWAY_PORT), style=QSTYLE).ask() or DEFAULT_GATEWAY_PORT)

    import secrets
    raw = questionary.text("API key (blank = auto-generate)", default="", style=QSTYLE).ask()
    api_key = raw.strip() if raw and raw.strip() else secrets.token_hex(16)

    success(f"Gateway: http://{host}:{port}  key=[bold]{api_key}[/]")
    return {"enabled": True, "host": host, "port": port, "api_key": api_key}


def step_bridges() -> dict[str, Any]:
    section("Step 5 — Bridges")
    c.print("CLI is always enabled. Select additional bridges:\n")

    choices = questionary.checkbox(
        "Additional bridges to enable",
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
                "token_env":        "DISCORD_BOT_TOKEN",
                "allowed_users":    [],
                "dm_enabled":       True,
                "guild_ids":        [],
                "prefix_required":  True,
                "command_prefix":   "!",
                "max_reply_length": 1900,
                "typing_indicator": True,
            },
        }
        if os.environ.get("DISCORD_BOT_TOKEN"):
            success("DISCORD_BOT_TOKEN found.")
        else:
            warn("DISCORD_BOT_TOKEN not set.")

    if "Matrix" in choices:
        c.print("\n[bold]Matrix[/] — set [bold]MATRIX_PASSWORD[/] in your environment.\n")
        homeserver = questionary.text("Homeserver URL", default="https://matrix.org", style=QSTYLE).ask() or "https://matrix.org"
        username   = questionary.text("Bot MXID (@bot:server.tld)", default="@yourbot:matrix.org", style=QSTYLE).ask() or "@yourbot:matrix.org"
        bridges["matrix"] = {
            "enabled": True,
            "options": {
                "homeserver":       homeserver,
                "username":         username,
                "password_env":     "MATRIX_PASSWORD",
                "device_name":      "TinyCTX",
                "store_path":       "matrix_store",
                "allowed_users":    [],
                "dm_enabled":       True,
                "room_ids":         [],
                "prefix_required":  True,
                "command_prefix":   "!",
                "max_reply_length": 16000,
                "sync_timeout_ms":  30000,
            },
        }
        if os.environ.get("MATRIX_PASSWORD"):
            success("MATRIX_PASSWORD found.")
        else:
            warn("MATRIX_PASSWORD not set.")

    return bridges


def step_skills(workspace: str) -> None:
    """
    Extract bundled skill zips into workspace/skills/ — only if that directory
    is empty (or doesn't exist). Skipped entirely if there are no bundled zips.
    """
    zip_files = sorted(BUNDLED_DIR.glob("*.zip"))
    if not zip_files:
        return  # nothing bundled, skip silently

    section("Step 6 — Skills")

    skills_dir = Path(workspace).expanduser() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Check if skills dir already has content
    existing = list(skills_dir.iterdir())
    if existing:
        c.print(f"  Skills directory already populated ({len(existing)} entries) — skipping.")
        return

    if not questionary.confirm(
        f"Install {len(zip_files)} bundled skill(s) into workspace?",
        default=True,
        style=QSTYLE,
    ).ask():
        warn("Skipping skill installation.")
        return

    for zp in zip_files:
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(skills_dir)
            success(f"Extracted [bold]{zp.name}[/]")
        except Exception as e:
            warn(f"Failed to extract {zp.name}: {e}")


def step_bootstrap_md(workspace: str) -> None:
    """
    Copy SOUL.md / AGENTS.md / MEMORY.md from bundled/ into the workspace root,
    but only if they don't already exist there. Runs silently with no prompts.
    """
    ws = Path(workspace).expanduser()
    for fname in BUNDLED_MD:
        src  = BUNDLED_DIR / fname
        dest = ws / fname
        if not src.exists():
            continue          # nothing bundled for this file
        if dest.exists():
            continue          # already present, never overwrite
        try:
            shutil.copy2(src, dest)
            success(f"Installed bundled [bold]{fname}[/] → {dest}")
        except Exception as e:
            warn(f"Could not copy {fname}: {e}")


def step_health_check(gateway: dict[str, Any]) -> None:
    section("Step 7 — Health Check")
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
    gateway:   dict,
    bridges:   dict,
    existing:  dict | None,
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
    base["gateway"]   = gateway

    existing_bridges = base.get("bridges", {})
    for name, bcfg in bridges.items():
        existing_bridges[name] = bcfg
    base["bridges"] = existing_bridges

    mem = base.get("memory_search", {})
    mem["auto_inject"] = True
    if embed_cfg:
        mem["embedding_model"] = "embed"
    base["memory_search"] = mem

    base.setdefault("logging", {"level": "INFO"})
    base.setdefault("max_tool_cycles", 25)

    return base


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python -m onboard", description="TinyCTX Onboarding Wizard")
    p.add_argument("--reset", action="store_true", help="Wipe existing config and start fresh")
    args = p.parse_args()

    providers = load_providers()
    existing  = load_existing_config()

    c.print(f"[bold cyan]{BANNER}[/]")

    # 0. detect / reset
    mode = step_detect(args.reset)
    if mode == "reset":
        CONFIG_PATH.unlink(missing_ok=True)
        existing = None
        success("Config reset.")
    elif mode == "new":
        c.print(Panel("[bold]Welcome![/] Let's get TinyCTX configured.", border_style="cyan"))

    # 1–5: model, embedding, workspace, gateway, bridges
    model_cfg = step_llm(providers)
    embed_cfg = step_embedding(providers)
    workspace = step_workspace()
    gateway   = step_gateway()
    bridges   = step_bridges()

    # 6: skills — only runs if bundled zips exist and skills dir is empty
    step_skills(workspace)

    # bootstrap md files silently (no prompt)
    step_bootstrap_md(workspace)

    # assemble + write
    data = assemble_config(model_cfg, embed_cfg, workspace, gateway, bridges, existing)
    write_config(data)
    success(f"Config written to [bold]{CONFIG_PATH}[/]")

    step_health_check(gateway)
    step_summary()


if __name__ == "__main__":
    main()
