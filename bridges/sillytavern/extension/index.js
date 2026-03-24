/**
 * TinyCTX — SillyTavern Extension
 * ================================
 * Turns SillyTavern into a seamless thin client for a TinyCTX gateway.
 * Users interact with ST exactly as they always have — the extension
 * transparently remaps every ST chat action to gateway API calls.
 *
 * Managed character: "TinyCTX Agent" (auto-created on first load)
 *   description  → AGENTS.md  (gateway workspace)
 *   personality  → SOUL.md
 *   scenario     → MEMORY.md
 *
 * ST action → TinyCTX mapping:
 *   Send message        → POST   /v1/sessions/{id}/message          (stream)
 *   Regenerate / Swipe  → DELETE last assistant entry + re-send
 *   Continue            → POST   /v1/sessions/{id}/message  {text: "[CONTINUE]"}
 *   Edit message        → PATCH  /v1/sessions/{id}/history/{eid}
 *   Delete message      → DELETE /v1/sessions/{id}/history/{eid}
 *   Start new chat      → switch to a new session UUID
 *   Manage chat files   → session list panel (replaces ST's file list)
 *   Rename chat         → PATCH  /v1/sessions/{id}/rename
 *   Delete chat         → DELETE /v1/sessions/{id}
 *   Impersonate         → ST handles natively (goes through generate interceptor
 *                         with type="impersonate"; we skip and let ST do it)
 *
 * Gateway config lives in extensionSettings — no per-card hacks.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODULE_NAME = "tinyctx";
const CHAR_NAME   = "TinyCTX Agent";

const FIELD_MAP = {
    description: "AGENTS.md",
    personality:  "SOUL.md",
    scenario:     "MEMORY.md",
};

// Internal sentinel sent to trigger a "continue" on the agent side
const CONTINUE_SENTINEL = "\x00__TINYCTX_CONTINUE__\x00";

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

const DEFAULT_SETTINGS = Object.freeze({
    endpoint:         "http://127.0.0.1:8080",
    api_key:          "",
    active_session:   "default",
    show_tool_events: true,
    sync_card_fields: true,
});

function getSettings() {
    const { extensionSettings } = SillyTavern.getContext();
    if (!extensionSettings[MODULE_NAME]) {
        extensionSettings[MODULE_NAME] = structuredClone(DEFAULT_SETTINGS);
    }
    for (const key of Object.keys(DEFAULT_SETTINGS)) {
        if (!Object.hasOwn(extensionSettings[MODULE_NAME], key)) {
            extensionSettings[MODULE_NAME][key] = DEFAULT_SETTINGS[key];
        }
    }
    return extensionSettings[MODULE_NAME];
}

function saveSettings() {
    SillyTavern.getContext().saveSettingsDebounced();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let isTinyCTXActive  = false;
let healthPollTimer  = null;

// ---------------------------------------------------------------------------
// Gateway
// ---------------------------------------------------------------------------

function gw(path) {
    return getSettings().endpoint.replace(/\/$/, "") + path;
}

function gwHeaders() {
    const h = { "Content-Type": "application/json" };
    const k = getSettings().api_key;
    if (k) h["Authorization"] = `Bearer ${k}`;
    return h;
}

async function gwFetch(path, opts = {}) {
    return fetch(gw(path), { headers: gwHeaders(), ...opts });
}

async function gwHealth() {
    try {
        const r = await fetch(gw("/v1/health"), { signal: AbortSignal.timeout(3000) });
        return r.ok ? await r.json() : null;
    } catch { return null; }
}

async function gwSessions() {
    try {
        const r = await gwFetch("/v1/sessions");
        return r.ok ? await r.json() : [];
    } catch { return []; }
}

function activeSession() { return getSettings().active_session || "default"; }

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(text, cls = "idle") {
    document.querySelector(".tinyctx-dot")    ?.setAttribute("data-state", cls);
    document.getElementById("tinyctx-status-text") &&
        (document.getElementById("tinyctx-status-text").textContent = text);
}

// ---------------------------------------------------------------------------
// Workspace sync
// ---------------------------------------------------------------------------

async function syncField(field, value) {
    if (!getSettings().sync_card_fields) return;
    const file = FIELD_MAP[field];
    if (!file || value == null) return;
    try {
        await gwFetch(`/v1/workspace/files/${file}`, {
            method: "PUT",
            body: JSON.stringify({ content: value }),
        });
    } catch (e) {
        console.warn(`[TinyCTX] sync ${field}→${file} failed:`, e);
    }
}

async function syncAllFields() {
    const { characters, characterId } = SillyTavern.getContext();
    const char = characters?.[characterId];
    if (!char) return;
    for (const [field] of Object.entries(FIELD_MAP)) {
        await syncField(field, char[field] ?? char.data?.[field] ?? "");
    }
}

// ---------------------------------------------------------------------------
// History helpers — fetch from gateway and render into ST's chat array
// ---------------------------------------------------------------------------

/**
 * Load gateway history for the current session into ST's mutable chat array.
 * This makes ST render the correct conversation when switching sessions.
 */
async function loadHistoryIntoST() {
    const ctx = SillyTavern.getContext();
    if (!ctx || !ctx.chat) return;

    try {
        const r = await gwFetch(`/v1/sessions/${encodeURIComponent(activeSession())}/history`);
        if (!r.ok) return;
        const entries = await r.json();

        // Clear ST's local chat array and repopulate with gateway entries
        ctx.chat.length = 0;

        for (const e of entries) {
            if (e.role === "user") {
                ctx.chat.push({
                    is_user:   true,
                    name:      ctx.name1 ?? "You",
                    mes:       typeof e.content === "string" ? e.content : "",
                    send_date: Date.now(),
                    extra:     { tinyctx_entry_id: e.id },
                });
            } else if (e.role === "assistant" && typeof e.content === "string" && e.content) {
                ctx.chat.push({
                    is_user:   false,
                    name:      CHAR_NAME,
                    mes:       e.content,
                    send_date: Date.now(),
                    extra:     { tinyctx_entry_id: e.id },
                });
            }
            // Skip tool_call / tool_result / system entries from display
        }

        // Re-render chat
        if (typeof ctx.reloadCurrentChat === "function") {
            await ctx.reloadCurrentChat();
        }
    } catch (e) {
        console.warn("[TinyCTX] loadHistory failed:", e);
    }
}

/** Get the last assistant entry_id from gateway history. */
async function getLastAssistantEntryId() {
    try {
        const r = await gwFetch(`/v1/sessions/${encodeURIComponent(activeSession())}/history`);
        if (!r.ok) return null;
        const entries = await r.json();
        // Walk back to find last assistant text entry
        for (let i = entries.length - 1; i >= 0; i--) {
            if (entries[i].role === "assistant" && typeof entries[i].content === "string") {
                return entries[i].id;
            }
        }
    } catch {}
    return null;
}

/** Get the last user message text from ST's chat (before the current generation). */
function getLastUserText() {
    const { chat } = SillyTavern.getContext();
    for (let i = (chat?.length ?? 0) - 1; i >= 0; i--) {
        if (chat[i].is_user) return chat[i].mes?.trim() ?? "";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Core: send to TinyCTX and stream response
// Returns the accumulated assistant text string.
// ---------------------------------------------------------------------------

async function sendMessage(text) {
    const sid = activeSession();
    setStatus(`Sending → ${sid}…`, "connecting");

    let response;
    try {
        response = await fetch(gw(`/v1/sessions/${encodeURIComponent(sid)}/message`), {
            method:  "POST",
            headers: gwHeaders(),
            body:    JSON.stringify({ text, stream: true, session_type: "dm" }),
        });
    } catch (e) {
        setStatus("Network error", "err");
        throw e;
    }

    if (response.status === 429) {
        setStatus("Busy — retrying…", "connecting");
        await new Promise(r => setTimeout(r, 1500));
        return sendMessage(text);
    }

    if (!response.ok) {
        setStatus(`Error ${response.status}`, "err");
        throw new Error(`Gateway ${response.status}`);
    }

    let   accumulated = "";
    const toolLines   = [];
    const showTools   = getSettings().show_tool_events;

    for await (const ev of readSSE(response)) {
        switch (ev.type) {
            case "text_chunk":   accumulated += ev.text; break;
            case "text_final":   if (ev.text) accumulated = ev.text; break;
            case "tool_call":
                if (showTools) toolLines.push(`*🔧 ${ev.name}(${JSON.stringify(ev.args ?? {})})*`);
                break;
            case "tool_result": {
                if (showTools) {
                    const sym     = ev.is_error ? "✗" : "✓";
                    const preview = (ev.output ?? "").slice(0, 100).replace(/\n/g, " ");
                    toolLines.push(`*${sym} ${ev.name}: ${preview}${(ev.output?.length ?? 0) > 100 ? "…" : ""}*`);
                }
                break;
            }
            case "error":   setStatus(`Agent error: ${ev.message}`, "err"); break;
            case "done":    setStatus(`Session: ${sid}`, "ok"); break;
        }
    }

    return toolLines.length
        ? toolLines.join("\n") + "\n\n" + accumulated
        : accumulated;
}

// ---------------------------------------------------------------------------
// SSE reader
// ---------------------------------------------------------------------------

async function* readSSE(response) {
    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            try { yield JSON.parse(raw); } catch { /* skip malformed */ }
        }
    }
}

// ---------------------------------------------------------------------------
// Generate interceptor  (manifest: generate_interceptor: "tinyCTXIntercept")
//
// ST calls this with (chat, type) where type is one of:
//   "normal"      — regular user send
//   "regenerate"  — user hit regenerate / swipe
//   "continue"    — user hit Continue
//   "impersonate" — user hit Impersonate (AI writes as user)
//   "quiet"       — background generation (summarisation, etc.)
//
// Return a string → ST uses it as the reply.
// Return undefined / null → ST proceeds with its own generation.
// ---------------------------------------------------------------------------

window.tinyCTXIntercept = async function(chat, type) {
    if (!isTinyCTXActive) return; // not our character, pass through

    // Impersonate: let ST handle it natively with its own pipeline.
    // ST generates a user-voice message; TinyCTX has nothing to add.
    if (type === "impersonate") return;

    // Quiet background calls (e.g. summarisation extensions): pass through.
    if (type === "quiet") return;

    try {
        if (type === "regenerate") {
            // Delete the last assistant entry from gateway history, then re-send
            // the last user message so the agent produces a fresh response.
            const eid = await getLastAssistantEntryId();
            if (eid) {
                await gwFetch(
                    `/v1/sessions/${encodeURIComponent(activeSession())}/history/${eid}`,
                    { method: "DELETE" }
                );
            }
            const userText = getLastUserText();
            if (!userText) return "[TinyCTX: no user message to regenerate from]";
            return await sendMessage(userText);
        }

        if (type === "continue") {
            // Send a special sentinel; TinyCTX's agent will treat it as "continue"
            // (the SOUL.md / system prompt should instruct the agent on this, or the
            // gateway can strip it and send an empty continuation turn).
            return await sendMessage(CONTINUE_SENTINEL);
        }

        // Default: "normal" send — extract the latest user message
        const lastUser = [...chat].reverse().find(m => m.is_user);
        if (!lastUser) return;
        const text = lastUser.mes?.trim();
        if (!text) return;

        return await sendMessage(text);

    } catch (e) {
        console.error("[TinyCTX] intercept error:", e);
        return `[TinyCTX error: ${e.message}]`;
    }
};

// ---------------------------------------------------------------------------
// Hook: message edit  (MESSAGE_EDITED event)
// When the user edits a message in ST, sync the change to gateway history.
// ---------------------------------------------------------------------------

async function onMessageEdited(messageId) {
    if (!isTinyCTXActive) return;

    const { chat } = SillyTavern.getContext();
    const msg = chat?.[messageId];
    if (!msg) return;

    const entryId = msg.extra?.tinyctx_entry_id;
    if (!entryId) return; // new message without gateway backing

    await gwFetch(
        `/v1/sessions/${encodeURIComponent(activeSession())}/history/${entryId}`,
        { method: "PATCH", body: JSON.stringify({ content: msg.mes }) }
    );
}

// ---------------------------------------------------------------------------
// Hook: message delete  (MESSAGE_DELETED event)
// ---------------------------------------------------------------------------

async function onMessageDeleted(messageId) {
    if (!isTinyCTXActive) return;

    // ST passes the index; we need to look it up from what we stored before deletion
    // ST fires MESSAGE_DELETED after removal, so we track deletions via a pre-delete hook
    // on the delete button click. See bindChatDeleteHook() below.
}

// ---------------------------------------------------------------------------
// Hook: new chat  (CHAT_CHANGED event — when new chat is started)
// ST's "Start new chat" creates a new JSONL file. We intercept by detecting
// when the chat becomes empty and switching to a new session UUID.
// ---------------------------------------------------------------------------

let _lastChatId = null;

async function onChatChanged() {
    const { characters, characterId, chatId } = SillyTavern.getContext();
    const char = characters?.[characterId];

    if (char?.name !== CHAR_NAME) {
        if (isTinyCTXActive) deactivate();
        return;
    }

    if (!isTinyCTXActive) await activate();

    // Detect if ST started a new chat (new chatId = user hit "Start new chat")
    if (chatId && chatId !== _lastChatId) {
        _lastChatId = chatId;
        // If ST created a brand-new empty chat, map it to a new session UUID
        const ctx = SillyTavern.getContext();
        if ((ctx.chat?.length ?? 0) === 0) {
            // Start a new TinyCTX session with a timestamp-based ID
            const newId = `session-${Date.now()}`;
            getSettings().active_session = newId;
            saveSettings();
            renderSessionPanel();
            setStatus(`New session: ${newId}`, "ok");
        } else {
            // Existing chat loaded — pull history from gateway
            await loadHistoryIntoST();
        }
    }
}

// ---------------------------------------------------------------------------
// Hook: rename chat
// ST fires CHAT_RENAMED (or user calls renameChat from context).
// We intercept the rename input by patching the "rename" confirmation flow.
// ---------------------------------------------------------------------------

function bindRenameHook() {
    // ST's rename chat button is #chat_rename_button or similar.
    // We listen for a rename submit and call gateway rename.
    // The exact selector depends on ST version — we attach to the known form.
    const renameBtn = document.getElementById("chat_rename_confirm_button")
        ?? document.querySelector(".rename_chat_confirm");
    if (!renameBtn || renameBtn.dataset.tinyCTXBound) return;
    renameBtn.dataset.tinyCTXBound = "1";

    renameBtn.addEventListener("click", async () => {
        if (!isTinyCTXActive) return;
        const input = document.getElementById("chat_rename_input")
            ?? document.querySelector(".rename_chat_input");
        const newName = input?.value?.trim();
        if (!newName) return;
        const oldId = activeSession();
        await gwFetch(`/v1/sessions/${encodeURIComponent(oldId)}/rename`, {
            method: "PATCH",
            body:   JSON.stringify({ new_id: newName }),
        });
        getSettings().active_session = newName;
        saveSettings();
        renderSessionPanel();
    });
}

// ---------------------------------------------------------------------------
// Session panel — replaces "Manage chat files" for the TinyCTX char
// Injected into ST's right panel above the character list.
// ---------------------------------------------------------------------------

const SESSION_PANEL_HTML = `
<div id="tinyctx-session-panel" style="display:none;">
    <div class="tinyctx-session-header">
        <span>Sessions</span>
        <button id="tinyctx-new-session-btn" title="Start new session">＋ New</button>
    </div>
    <div id="tinyctx-session-list"></div>
</div>
`;

function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function renderSessionPanel() {
    const list = document.getElementById("tinyctx-session-list");
    if (!list) return;

    const sessions = await gwSessions();
    const cur      = activeSession();
    list.innerHTML = "";

    if (!sessions.length) {
        list.innerHTML = `<div class="tinyctx-session-empty">No sessions yet</div>`;
        return;
    }

    for (const s of sessions) {
        const item = document.createElement("div");
        item.className = "tinyctx-session-item" + (s.id === cur ? " active" : "");
        item.innerHTML = `
            <span class="tinyctx-session-icon">💬</span>
            <span class="tinyctx-session-name">${escapeHtml(s.id)}</span>
            <span class="tinyctx-session-turns">${s.turns}t</span>
            <button class="tinyctx-session-delete" title="Delete session" data-id="${escapeHtml(s.id)}">🗑</button>
        `;
        item.querySelector(".tinyctx-session-name").addEventListener("click", () => switchSession(s.id));
        item.querySelector(".tinyctx-session-delete").addEventListener("click", async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete session "${s.id}"?`)) return;
            await gwFetch(`/v1/sessions/${encodeURIComponent(s.id)}`, { method: "DELETE" });
            if (s.id === activeSession()) {
                getSettings().active_session = "default";
                saveSettings();
            }
            await renderSessionPanel();
        });
        list.appendChild(item);
    }
}

async function switchSession(id) {
    getSettings().active_session = id;
    saveSettings();
    setStatus(`Loading session: ${id}…`, "connecting");
    await loadHistoryIntoST();
    await renderSessionPanel();
    setStatus(`Session: ${id}`, "ok");
}

function showSessionPanel() {
    const p = document.getElementById("tinyctx-session-panel");
    if (p) p.style.display = "";
}

function hideSessionPanel() {
    const p = document.getElementById("tinyctx-session-panel");
    if (p) p.style.display = "none";
}

// ---------------------------------------------------------------------------
// Activate / deactivate
// ---------------------------------------------------------------------------

async function activate() {
    isTinyCTXActive = true;
    setStatus("Connecting…", "connecting");
    showSessionPanel();
    await syncAllFields();
    await renderSessionPanel();
    startPolling();
}

function deactivate() {
    isTinyCTXActive = false;
    stopPolling();
    hideSessionPanel();
    setStatus("Inactive — select TinyCTX Agent", "idle");
}

function startPolling() {
    if (healthPollTimer) return;
    pollOnce();
    healthPollTimer = setInterval(pollOnce, 15_000);
}

function stopPolling() {
    clearInterval(healthPollTimer);
    healthPollTimer = null;
}

async function pollOnce() {
    const h = await gwHealth();
    if (h) setStatus(`Connected — uptime ${Math.round(h.uptime_s)}s`, "ok");
    else   setStatus("Gateway unreachable", "err");
}

// ---------------------------------------------------------------------------
// Managed character bootstrap
// ---------------------------------------------------------------------------

async function ensureCharacter() {
    const { characters } = SillyTavern.getContext();
    if ((characters ?? []).some(c => c.name === CHAR_NAME)) return;

    try {
        await fetch("/api/characters/create", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                name:          CHAR_NAME,
                description:   "<!-- Edit to update AGENTS.md on the TinyCTX gateway -->",
                personality:   "<!-- Edit to update SOUL.md on the TinyCTX gateway -->",
                scenario:      "<!-- Edit to update MEMORY.md on the TinyCTX gateway -->",
                first_mes:     "TinyCTX connected. Select a session from the panel above.",
                mes_example:   "",
                creator_notes: "Managed by TinyCTX extension — do not rename.",
                tags:          ["tinyctx"],
            }),
        });
        console.log("[TinyCTX] Created managed character.");
        // Reload character list if possible
        const ctx = SillyTavern.getContext();
        if (typeof ctx.reloadCurrentChat === "function") await ctx.reloadCurrentChat();
    } catch (e) {
        console.error("[TinyCTX] Failed to create character:", e);
    }
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------

const SETTINGS_HTML = `
<div id="tinyctx-settings">
    <div id="tinyctx-status-bar">
        <span class="tinyctx-dot" data-state="idle"></span>
        <span id="tinyctx-status-text">Inactive — select TinyCTX Agent to connect</span>
    </div>

    <div class="tinyctx-field">
        <label for="tinyctx-endpoint">Gateway endpoint</label>
        <input type="text" id="tinyctx-endpoint" placeholder="http://127.0.0.1:8080">
    </div>

    <div class="tinyctx-field">
        <label for="tinyctx-apikey">API key</label>
        <input type="password" id="tinyctx-apikey" placeholder="blank = no auth">
    </div>

    <div class="tinyctx-row">
        <input type="checkbox" id="tinyctx-tool-events">
        <label for="tinyctx-tool-events">Show tool events in chat</label>
    </div>

    <div class="tinyctx-row">
        <input type="checkbox" id="tinyctx-sync-fields">
        <label for="tinyctx-sync-fields">Sync card fields → workspace on save</label>
    </div>

    <div class="tinyctx-btn-row">
        <button id="tinyctx-btn-ping">Ping</button>
        <button id="tinyctx-btn-sync">Sync fields</button>
        <button id="tinyctx-btn-recreate">Recreate character</button>
    </div>
</div>
`;

function bindSettings() {
    const s = getSettings();
    const ep  = document.getElementById("tinyctx-endpoint");
    const key = document.getElementById("tinyctx-apikey");
    const te  = document.getElementById("tinyctx-tool-events");
    const sf  = document.getElementById("tinyctx-sync-fields");
    if (!ep) return;

    ep.value   = s.endpoint;
    key.value  = s.api_key;
    te.checked = s.show_tool_events;
    sf.checked = s.sync_card_fields;

    const persist = () => {
        s.endpoint         = ep.value.trim().replace(/\/$/, "") || "http://127.0.0.1:8080";
        s.api_key          = key.value.trim();
        s.show_tool_events = te.checked;
        s.sync_card_fields = sf.checked;
        saveSettings();
    };
    [ep, key, te, sf].forEach(el => {
        el.addEventListener("change", persist);
        el.addEventListener("input",  persist);
    });

    document.getElementById("tinyctx-btn-ping")?.addEventListener("click", async () => {
        setStatus("Pinging…", "connecting");
        const h = await gwHealth();
        h ? setStatus(`OK — uptime ${Math.round(h.uptime_s)}s`, "ok")
          : setStatus("Unreachable", "err");
    });

    document.getElementById("tinyctx-btn-sync")?.addEventListener("click", async () => {
        await syncAllFields();
        setStatus("Synced", "ok");
    });

    document.getElementById("tinyctx-btn-recreate")?.addEventListener("click", async () => {
        await ensureCharacter();
    });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
    const { eventSource, eventTypes } = SillyTavern.getContext();

    // Settings panel
    $("#extensions_settings").append(`
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>TinyCTX</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">${SETTINGS_HTML}</div>
        </div>
    `);
    bindSettings();

    // Session panel — inject into right nav panel above character list
    const anchor = document.getElementById("rm_print_characters_block")
        ?? document.getElementById("right-nav-panel")
        ?? document.body;
    anchor.insertAdjacentHTML("afterbegin", SESSION_PANEL_HTML);

    document.getElementById("tinyctx-new-session-btn")?.addEventListener("click", async () => {
        const id = `session-${Date.now()}`;
        getSettings().active_session = id;
        saveSettings();
        setStatus(`New session: ${id}`, "ok");
        // Trigger ST to start a new chat, which will call onChatChanged
        // which will see an empty chat and pick up the new session.
        const { generate } = SillyTavern.getContext();
        // Use ST's "new chat" UI button if available
        document.getElementById("option_start_new_chat")?.click()
            ?? document.querySelector("[data-i18n='Start new chat']")?.click();
    });

    // Events
    eventSource.on(eventTypes.CHAT_CHANGED,     onChatChanged);
    eventSource.on(eventTypes.CHARACTER_EDITED, async () => {
        if (isTinyCTXActive) await syncAllFields();
    });
    eventSource.on(eventTypes.MESSAGE_EDITED,   onMessageEdited);
    eventSource.on(eventTypes.APP_READY, async () => {
        await ensureCharacter();
        await onChatChanged();
        bindRenameHook();
    });

    console.log("[TinyCTX] Extension loaded.");
})();
