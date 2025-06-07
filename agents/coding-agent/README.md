# Coding Agent

An AI-powered coding assistant that combines multiple LLM providers with a comprehensive tool system for code generation, file manipulation, and web search capabilities.

## Installation

```bash
pip install together anthropic pydantic
```

## API Keys & Credentials

Set the required environment variables (only need one LLM provider):

```bash
# LLM Provider (choose one)
export TOGETHER_API_KEY="your_together_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional, only if using --anthropic

# GitHub Integration (optional, to give agent version control features)
export GITHUB_USERNAME="your_github_username"
export GITHUB_PAT="your_github_personal_access_token"
export GITHUB_EMAIL="your_email@example.com"
export GITHUB_NOREPLY_EMAIL="your_username@users.noreply.github.com"
```

## Usage

```bash
python agent.py [options]
```

**Default:** Llama 3.3 70B Instruct Turbo with tools enabled

### Options

| Flag | Description |
|------|-------------|
| `--small` | Use Llama 3.1 8B model |
| `--deepseek` | Use DeepSeek V3 model |
| `--huge` | Use Llama 3.1 405B model |
| `--anthropic` | Use Claude-4 Sonnet |
| `--notools` | Disable tool usage |
| `--verbose` | Enable detailed output (agent thoughts) |

## File Structure

```
agents/coding-agent/
├── agent.py                      # Main agent orchestrator with tool interaction loop
├── api.py                        # Unified API wrapper for Together AI and Anthropic clients
├── memory.py                     # Conversation memory management with compression
├── global_constants.py           # Configuration constants (scratch directory path)
├── prompts/                      # System and tool prompts
│   ├── system_prompts.py         # Core agent behavior prompts
│   ├── tool_prompts.py           # Individual tool instruction prompts
│   ├── memory_prompts.py         # Memory compression prompts
│   └── sandbox_prompts.py        # Code execution environment prompts
├── tools/                        # Tool implementations
│   ├── base_tool.py              # Abstract base class for all tools
│   ├── search_tool.py            # Web search functionality
│   ├── run_code_tool.py          # Code execution in isolated environment
│   ├── read_file_tool.py         # File reading from sandbox
│   ├── write_file_tool.py        # File writing to sandbox
│   ├── registry.py               # Tool registration and management
│   └── tool_validation_utils.py  # Tool call parsing and validation
└── agent_scratch/                # Isolated workspace for file operations (auto-created)
```

## Key Architecture

- **Multi-LLM Support**: Seamlessly switches between Together AI (Llama models, DeepSeek) and Anthropic (Claude)
- **Tool-Augmented Generation**: Iterative tool calling with reflection until task completion
- **Memory Management**: Automatic conversation compression when context exceeds 200k characters
- **Sandboxed Execution**: Isolated code execution and file operations in `agent_scratch/` directory
- **Modular Tool System**: Extensible tool registry for adding new capabilities
- **GitHub Integration**: Built-in version control with repository creation, cloning, and push/pull operations

## Available Tools

1. **Search Tool**: Real-time web search with shallow/deep modes
2. **Run Code Tool**: Execute Python/shell code in sandboxed environment
3. **Write File Tool**: Create/overwrite files in agent sandbox
4. **Read File Tool**: Read existing files from agent sandbox

## GitHub Integration

The agent includes built-in GitHub integration for version control operations. When GitHub credentials are configured, the agent can:

- Create new repositories
- Clone existing repositories
- Commit and push changes
- Pull updates
- Manage branches

Git operations are executed directly through the run_code tool using actual git commands, not simulated.
