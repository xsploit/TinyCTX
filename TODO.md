# TinyCTX — TODO

## Bridges / input
- [ ] **Image and file attachments** — when a user attaches an image or file in any bridge:
      1. Always save to `workspace/uploads/<filename>` (permanent record, agent can read later).
      2. Inline strategy (few/small files): encode as base64 and attach directly to the
         user `HistoryEntry` as a content block list (Anthropic vision API format for images;
         raw text injection for small text files).
      3. Reference strategy (many files or total size over threshold): append a system note
         to the user message — e.g. `[N file(s) uploaded to workspace/uploads/: foo.png, bar.pdf]`
         — and let the agent read them explicitly via filesystem tools.
      4. Thresholds (config): `upload_inline_max_files` (default 3),
         `upload_inline_max_bytes` (default ~200KB total); if either is exceeded, fall
         back to reference strategy.
      5. Bridge responsibility: Matrix and Discord bridges detect attachments and hand off
         to a shared `attachments.py` utility that handles saving + strategy selection.
         Gateway `POST /v1/sessions/{id}/message` accepts multipart or a JSON
         `attachments: [{name, data_b64, mime_type}]` field.

## Ops / management
- [ ] **Web UI** — not a chat interface; an admin/stats panel.
      Likely a single-page aiohttp route serving a small HTML+JS dashboard.
      Should surface: uptime, per-session queue depth + turn count, memory index
      stats (file count, chunk count, last sync), active bridges, active modules,
      token budget gauges per session. Could reuse the existing `/v1/health` data
      plus a new `/v1/stats` endpoint.

- [ ] **Webhooks** — inbound webhook bridge (`bridges/webhook/`).
      Expose a `POST /webhook/{id}` endpoint; route payload into a configured
      session as an `InboundMessage`. Useful for external triggers (GitHub, n8n,
      Zapier, etc.). Auth via a per-hook secret in config. Optionally support
      outbound webhooks on `AgentTextFinal` events (call a URL with the reply).
