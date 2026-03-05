EXTENSION_META = {
    "name":    "web",
    "version": "1.0",
    "description": (
        "Web tools: async HTTP requests, DuckDuckGo search, "
        "and Playwright browser automation (navigate, click, type, extract, screenshot). "
        "Screenshots are saved to workspace/downloads/. "
        "One browser instance per agent session."
    ),
    "default_config": {
        "headless":              False,
        "timeout_ms":            30000,
        "wait_until":            "domcontentloaded",
        "shift_enter_for_newline": True,
        "ignore_tags":           ["script", "style"],
        "max_discovery_elements": 40,
        "search_results":        5,
        "downloads_dir":         "downloads",
    },
}