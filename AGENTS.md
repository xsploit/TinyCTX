# AGENTS.md

This file exists to stop path confusion and wasted time.

## Primary Working Repo

The main app repo is:

- `C:\Users\SUBSECT\Documents\GitHub\clawdcode\TinyCTX`

If the task is about TinyCTX behavior, CLI/TUI, gateway behavior, tools, memory, compaction, subagents, config, or OpenAI-compatible API behavior, work here first.

Important TinyCTX entry points:

- `main.py`
- `agent.py`
- `ai.py`
- `context.py`
- `gateway/__main__.py`
- `bridges/cli/__main__.py`
- `config/__main__.py`

Primary local config used during manual runs:

- `C:\Users\SUBSECT\Documents\GitHub\clawdcode\TinyCTX\config.yaml`

## External llama.cpp Repo

The local llama.cpp / TurboQuant repo is separate and lives here:

- `D:\tq\llama-cpp-turboquant-cuda`

If the task is about `llama-server`, `/v1/chat/completions`, `/v1/responses`, prompt cache, sticky slots, CUDA build issues, or server flags, that work is in the external repo, not TinyCTX.

Relevant server files there:

- `D:\tq\llama-cpp-turboquant-cuda\tools\server\server.cpp`
- `D:\tq\llama-cpp-turboquant-cuda\tools\server\server-context.cpp`
- `D:\tq\llama-cpp-turboquant-cuda\tools\server\README.md`
- `D:\tq\llama-cpp-turboquant-cuda\common\arg.cpp`

## Current Authoritative Server Binary

Use this binary unless it is explicitly rebuilt elsewhere and verified:

- `D:\tq\llama-cpp-turboquant-cuda\build\bin\Release\llama-server.exe`

This is currently the newest verified `llama-server.exe` and it exposes:

- `--responses-store-path`

Do **not** assume `build-tq4-latest` is authoritative just because of the folder name. That directory name is stale and misleading.

## Build Directory Reality

Current source branch in the external repo may be newer than the build folder name.

Examples:

- Source branch can be `codex/turboquant-kv-cache-latest`
- Build folder can still be named `build-tq4-latest`

That does **not** mean the source is old. It only means the build directory was named badly.

Always verify by checking:

1. current git branch in `D:\tq\llama-cpp-turboquant-cuda`
2. actual `llama-server.exe` timestamp
3. actual `llama-server.exe --help` output

## Responses API Guidance

If the goal is **real** OpenAI-style `/v1/responses` support with server-side chaining, the intended setup is:

1. run direct local `llama-server`
2. use `/v1/responses`
3. verify `previous_response_id` support against the server binary actually being run

Do not confuse:

- TinyCTX gateway `/v1/responses`
- direct llama.cpp `/v1/responses`
- llama.cpp router mode `/v1/responses`

Router mode has different limitations and may reject features that direct local mode supports.

## Current Project Priorities

Current active priorities are:

1. stabilize TinyCTX CLI/TUI behavior
2. keep tool/debug visibility usable without clogging the transcript
3. make TinyCTX work cleanly with local llama.cpp `/v1/responses`
4. avoid path/build confusion between TinyCTX and the external TurboQuant server repo

## Before Changing Anything

Before editing or rebuilding, confirm:

- Are we working in TinyCTX or in `D:\tq\llama-cpp-turboquant-cuda`?
- Which `llama-server.exe` is actually being launched?
- Is the issue in TinyCTX client behavior, TinyCTX gateway behavior, or llama.cpp server behavior?

If those three things are not confirmed first, stop and confirm them before doing more work.
