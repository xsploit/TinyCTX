"""
config.py — Configuration loader.
Imports only from stdlib and PyYAML. Never imports from contracts or gateway.
"""
from __future__ import annotations
import logging, os
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class LLMConfig:
    model:       str
    base_url:    str
    api_key_env: str = "ANTHROPIC_API_KEY"  # name of the env var holding the key
    @property
    def api_key(self) -> str:
        # api_key_env can be set to "N/A" or left blank for local endpoints
        # that don't require authentication.
        if not self.api_key_env or self.api_key_env.upper() == "N/A":
            return ""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise EnvironmentError(f"LLM API key not set. Export {self.api_key_env} before starting.")
        return key

@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8765

@dataclass
class BridgeConfig:
    enabled: bool = False
    options: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        # allows bridge_cfg.token style access into options
        try:
            return self.options[name]
        except KeyError:
            raise AttributeError(name)

@dataclass
class MemoryConfig:
    workspace_path: Path = field(default_factory=lambda: Path("~/.agent/workspace").expanduser())
    def __post_init__(self):
        self.workspace_path = Path(self.workspace_path).expanduser().resolve()

@dataclass
class LoggingConfig:
    level: str = "INFO"
    def __post_init__(self):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid:
            raise ValueError(f"Invalid log level '{self.level}'.")
        self.level = self.level.upper()

@dataclass
class Config:
    llm:             LLMConfig
    gateway:         GatewayConfig = field(default_factory=GatewayConfig)
    bridges:         dict[str, BridgeConfig] = field(default_factory=dict)
    memory:          MemoryConfig  = field(default_factory=MemoryConfig)
    logging:         LoggingConfig = field(default_factory=LoggingConfig)
    max_tool_cycles: int           = 10
    context:         int           = 16384  # add this

def load(path="config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}

    llm_raw = raw.get("llm")
    if not llm_raw:
        raise ValueError("Config missing required section: [llm]")
    if not llm_raw.get("base_url"):
        raise ValueError("Config missing required field: llm.base_url")
    if not llm_raw.get("model"):
        raise ValueError("Config missing required field: llm.model")

    gw_raw  = raw.get("gateway", {})
    br_raw  = raw.get("bridges", {})
    mem_raw = raw.get("memory", {})
    log_raw = raw.get("logging", {})

    bridges: dict[str, BridgeConfig] = {}
    for name, br in raw.get("bridges", {}).items():
        if isinstance(br, dict):
            enabled = bool(br.get("enabled", False))
            options = {k: v for k, v in br.items() if k != "enabled"}
            bridges[name] = BridgeConfig(enabled=enabled, options=options)

    return Config(
        llm=LLMConfig(
            model=llm_raw["model"],
            base_url=llm_raw["base_url"],
            api_key_env=llm_raw.get("api_key_env", "ANTHROPIC_API_KEY"),
        ),
        gateway=GatewayConfig(
            host=gw_raw.get("host", "127.0.0.1"),
            port=int(gw_raw.get("port", 8765)),
        ),
        bridges = bridges,
        memory=MemoryConfig(workspace_path=Path(mem_raw.get("workspace_path", "~/.agent/workspace"))),
        logging=LoggingConfig(level=log_raw.get("level", "INFO")),
        max_tool_cycles=int(raw.get("max_tool_cycles", 10)),
        context=int(raw.get("context", 16384)),
    )

def apply_logging(cfg: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )