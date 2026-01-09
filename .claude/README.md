# Claude Code Project Configuration

This directory contains project-specific configuration for Claude Code.

## Current Configuration: Liberal Permissions

The `settings.local.json` file has been configured with **very liberal permissions** to allow the coding agent to work autonomously without asking for permission.

### What's Enabled

**All Tools with Wildcards:**
- ✅ `Bash(*)` - Execute any bash command
- ✅ `Read(*)` - Read any file
- ✅ `Write(*)` - Write any file
- ✅ `Edit(*)` - Edit any file
- ✅ `Glob(*)` - Search for any files
- ✅ `Grep(*)` - Search in any files
- ✅ `WebFetch(*)` - Fetch from any website
- ✅ `WebSearch` - Search the web
- ✅ `Task(*)` - Spawn any subagent tasks
- ✅ All MCP Memory tools
- ✅ All Context7 documentation tools

**Auto-Approval Settings:**
- Bash commands auto-approved
- File operations auto-approved
- Web access auto-approved
- Task spawning auto-approved
- Plan mode auto-approved

**Skip Confirmation:**
- Tool use
- File edits and writes
- Bash commands
- Web fetching
- Task spawning

### Security Note

⚠️ **These are very liberal permissions**. The agent can:
- Execute any bash command
- Read/write/edit any file in the project
- Access any website
- Spawn unlimited subagents

This is appropriate for:
- ✅ POC/development projects
- ✅ Projects where you trust the agent completely
- ✅ Local development environments
- ✅ Projects without sensitive data

This may NOT be appropriate for:
- ❌ Production environments
- ❌ Projects with sensitive credentials
- ❌ Shared systems
- ❌ Projects requiring strict change control

### To Restrict Permissions

If you want to restrict permissions later, edit `settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(ls:*)",           // Only allow specific commands
      "Read(/specific/path)", // Only allow specific paths
      "Write(/specific/path)"
    ]
  },
  "autoApprove": {
    "allTools": false         // Disable auto-approval
  }
}
```

### Documentation

For more on Claude Code configuration:
- Local: Run `/help` in Claude Code
- Online: https://docs.anthropic.com/claude/docs/claude-code

---

**Current Status:** ✅ Liberal permissions active - agent works autonomously
