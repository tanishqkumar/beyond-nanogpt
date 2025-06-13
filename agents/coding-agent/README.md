# Coding Agent

A minimal AI-powered coding assistant that uses only vanilla python and 
LLM completions. The agent can make entire apps, add features, fix bugs, make PRs end to end on simple GitHub repos. The repo includes logic defining tool use allowing it to read/write files, execute code, search the web, sandboxed environments, long and short term memory, etc. 

**This repository demonstrates the core logic requires to go from 
"base LLM" to "autonomous system in the wild,"** as well as key engineering/design decisions that have 
to be made on the way. We do not use any features of the LLM API besides sampling from LLMs given a 
prompt. 

## Installation

```bash
pip install together anthropic pydantic
```

## API Keys & Credentials

Set the required environment variables

```bash
# inference provider (choose one, or add OpenRouter/OpenAI etc)
export TOGETHER_API_KEY="your_together_api_key" # default is L3-70B hosted on Together AI 
export ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional, only if using --anthropic

# GitHub Integration (optional, to give agent version control features)
export GITHUB_USERNAME="your_github_username"
export GITHUB_PAT="your_github_personal_access_token"
export GITHUB_EMAIL="your_email@example.com"
export GITHUB_NOREPLY_EMAIL="your_username@users.noreply.github.com"

# Google Search API (required for web search tool in @search_tool.py)
export GOOGLE_SEARCH_KEY="your_google_custom_search_api_key"
export SEARCH_ENGINE_ID="your_google_custom_search_engine_id"
```

## Usage

```bash
python agent.py [options]
```

**Default:** Llama 3.3 70B Instruct 

### Options

| Flag | Description |
|------|-------------|
| `--small` | Use Llama 3.1 8B model |
| `--deepseek` | Use DeepSeek V3 model |
| `--huge` | Use Llama 3.1 405B model |
| `--anthropic` | Use Claude-4 Sonnet |
| `--notools` | Disable our tool logic (kills perf) |
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

## Key Architectural Details 

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

Git operations are executed directly through the run_code tool using actual git commands.
