EXTENSION_META = {
    "name":    "heartbeat",
    "version": "1.0",
    "description": (
        "Periodic agent turns on a configurable interval. "
        "Injects a synthetic message into the session so the agent can surface "
        "anything that needs attention. Reads HEARTBEAT.md if present. "
        "Replies of HEARTBEAT_OK are silently dropped."
    ),
    "default_config": {
        # Interval in minutes. 0 = disabled.
        "every_minutes":   30,
        # Prompt sent verbatim as the heartbeat user message.
        "prompt":          (
            "Read HEARTBEAT.md if it exists (workspace context). "
            "Follow it strictly. Do not infer or repeat old tasks from prior chats. "
            "If nothing needs attention, reply HEARTBEAT_OK."
        ),
        # After HEARTBEAT_OK, how many chars are still allowed before we deliver anyway.
        "ack_max_chars":   300,
        # Prompt sent to the agent when it hasn't returned HEARTBEAT_OK yet,
        # to nudge it to either finish the task or acknowledge.
        "continuation_prompt": (
            "Continue the task, or reply HEARTBEAT_OK when you are done."
        ),
        # Max number of continuation turns before giving up and moving on.
        "max_continuations":   5,
        # Session to run heartbeat turns in.
        # "main" = agent's own DM session (default).
        # "dm:<id>" or "group:<platform>:<id>" for an explicit session.
        "session":         "main",
        # active_hours: restrict heartbeats to a time window. null = always run.
        # Example: {"start": "09:00", "end": "22:00"}
        "active_hours":    None,
    },
}