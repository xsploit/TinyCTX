"""
main.py — Application entrypoint.

Startup order:
  1. Load config, init gateway.
  2. Start server (if config.server.enabled) — runs as a peer task.
  3. Scan bridges/, start each enabled bridge as a task.
  4. Wait for any task to complete (normal exit or crash).
  5. Cancel remaining tasks, shutdown gateway.

The server and bridges are peers — either can trigger shutdown if they exit.
The server is started before bridges so API clients can connect as soon as
the process is up, even before bridge tasks are running.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from pathlib import Path

from config import load as load_config, apply_logging
from router import Router

logger = logging.getLogger(__name__)

BRIDGES_DIR = Path("bridges")
GATEWAY_MOD = "gateway.__main__"


async def main() -> None:
    cfg = load_config()
    apply_logging(cfg.logging)

    gw = Router(config=cfg)

    tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ gateway
    if cfg.gateway.enabled:
        if not cfg.gateway.api_key:
            logger.warning("gateway.api_key is empty — gateway is unauthenticated!")
        try:
            gateway_mod = importlib.import_module(GATEWAY_MOD)
            tasks.append(asyncio.create_task(
                gateway_mod.run(gw, cfg.gateway),
                name="gateway",
            ))
            logger.info(
                "Started gateway on %s:%d",
                cfg.gateway.host, cfg.gateway.port,
            )
        except Exception:
            logger.exception("Failed to start gateway")

    # ------------------------------------------------------------------ bridges
    if BRIDGES_DIR.exists():
        for entry in sorted(BRIDGES_DIR.iterdir()):
            if not entry.is_dir() or not (entry / "__main__.py").exists():
                continue

            name = entry.name
            bridge_cfg = cfg.bridges.get(name)
            if bridge_cfg is None:
                logger.debug("Bridge '%s' has no config entry — skipping", name)
                continue
            if not bridge_cfg.enabled:
                logger.debug("Bridge '%s' is disabled — skipping", name)
                continue

            try:
                mod = importlib.import_module(f"bridges.{name}.__main__")
                if not hasattr(mod, "run"):
                    logger.warning("Bridge '%s' has no run() — skipping", name)
                    continue
                tasks.append(asyncio.create_task(mod.run(gw), name=f"bridge:{name}"))
                logger.info("Started bridge '%s'", name)
            except Exception:
                logger.exception("Failed to load bridge '%s'", name)

    if not tasks:
        logger.error("Nothing started — enable at least one bridge or the server in config.yaml.")
        return

    # ------------------------------------------------------------------ run
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for t in done:
        if t.exception():
            logger.error("Task '%s' crashed: %s", t.get_name(), t.exception())
        else:
            logger.info("Task '%s' exited cleanly.", t.get_name())

    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    await gw.shutdown()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
