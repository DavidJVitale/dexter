# Integration Guide: Google Workspace CLI (`gws`) for Local GenAI / LLM Agents

**Goal**  
Enable your local coding agent / LLM agent to natively read, search, send, create, and manage Gmail, Google Calendar, Drive files, Docs, Sheets, and more — without custom OAuth libraries, scraping, or brittle APIs. Use the official **Google Workspace CLI** (`gws`) via shell execution.

**Repo & Official Docs**  
- GitHub: https://github.com/googleworkspace/cli  
- Main binary: `gws` (installed via npm or prebuilt binaries)  
- License: Apache 2.0 (open-source)  
- Released: ~March 2026 (very recent — check for updates often)

**Why this is ideal for agents**  
- Dynamic command surface: auto-discovers all Workspace APIs via Google's Discovery Service  
- Clean **JSON output** by default with `--json` (perfect for LLM parsing)  
- 40+ built-in "agent skills" (pre-packaged high-level commands like `+triage`, `+agenda`, `+send`)  
- Supports MCP server mode (`gws mcp`) if your agent uses Model Context Protocol  
- Local token storage — no server needed  
- Free for personal @gmail.com accounts (uses standard OAuth + generous personal quotas)

## 1. Prerequisites (One-Time Human Setup)

The agent cannot do initial OAuth — you must do this once:

1. Install Node.js (if not already present) — `gws` is distributed via npm.
2. Install the CLI globally:
   ```bash
   npm install -g @googleworkspace/cli