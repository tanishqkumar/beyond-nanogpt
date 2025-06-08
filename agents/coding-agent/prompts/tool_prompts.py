import os

SEARCH_TOOL_PROMPT = """
    ----- 
    SEARCH TOOL - Use for real-time web information and factual data.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "search", "query": "search_query", "top_k": 3, "deep": false, "max_len": 20000}
    </tool_request>
    
    PARAMETERS:
    - query (required): Search query string
    - top_k (optional, default 3): Number of results (1-10)
    - deep (optional, default false): false = shallow (snippets), true = deep (full page contents)
    - max_len (optional, default 20000): Max character limit for results
    
    EXAMPLES:
    
    <tool_request>
    {"tool_name": "search", "query": "Bitcoin price current USD", "top_k": 2}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "weather forecast New York today"}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "Python tutorial", "deep": true}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "quantum computing applications 2024", "top_k": 5, "deep": true}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "web_search", "query": "Bitcoin price"}</tool_request>
    
    ❌ WRONG - Using "text" instead of "query":
    <tool_request>{"tool_name": "search", "text": "Bitcoin price current USD"}</tool_request>
    
    ❌ WRONG - Invalid wrapping tags:
    <tool_name>{"tool_name": "search", "query": "NFL playoff results 2024"}</tool_name>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'search', 'query': 'Python tutorial'}</tool_request>
    
    ❌ WRONG - Missing query parameter:
    <tool_request>{"tool_name": "search", "top_k": 5}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in value:
    <tool_request>{"tool_name": "search", "query": "What is a "quantum computer"?"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "search", "query": "AI news",}</tool_request>
    
    ❌ WRONG - Newlines in value not properly escaped:
    <tool_request>{"tool_name": "search", "query": "First line
    Second line"}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "search", "query": "C:\new_folder\file.txt"}</tool_request>
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use search results to inform other tool usage (e.g., search for documentation, then write_file to save code)
    - Multiple searches are encouraged for multi-hop information gathering
    - Search can complement run_code for research-driven development
    
    IMPORTANT: Never claim you "searched" or "used the search tool" unless you actually made a <tool_request> with the search tool.
    
    GUIDELINES:
    - Use shallow (deep: false) for quick facts, current events
    - Use deep (deep: true) for tutorials, detailed explanations, involved references
    - Use specific, precise search terms
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "search" and the search query MUST go in "query" field
    - Multiple searches are perfectly fine for comprehensive research
    ----- 
"""

# you'll need to set these in the env
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")
GITHUB_EMAIL = os.getenv("GITHUB_EMAIL", "")
GITHUB_NOREPLY_EMAIL = os.getenv("GITHUB_NOREPLY_EMAIL", "")

GITHUB_TEXT = """
    GITHUB INTEGRATION: 
    
    You have access to GitHub under username "{GITHUB_USERNAME}" with this personal access token:
    {GITHUB_PAT}
    In case you need it, the email is {GITHUB_EMAIL} 
    
    CRITICAL: When working with Git/GitHub, ALWAYS actually execute the commands using run_code with shell language.
    Do NOT just print or simulate git commands - ACTUALLY RUN THEM. 
    
    ⚠️ IMPORTANT GITHUB WORKFLOW RULES:
    1. ALWAYS check command outputs - never assume success
    2. ALWAYS verify branches exist and have shared history before creating PRs
    3. NEVER fabricate or simulate API responses - check actual output
    4. Use curl with -i flag to see response headers and status codes
    5. When cloning, ALWAYS cd into the repository directory before git operations
    6. ALWAYS verify you're pointing to the correct repository before making changes/PRs/pushes/pulls
    
    ⚠️ CRITICAL REPOSITORY VERIFICATION:
    Before ANY git operations (push, pull, PR creation), ALWAYS verify the remote repository:
    <tool_request>
    {{"tool_name": "run_code", "code": "pwd && git remote -v", "language": "shell"}}
    </tool_request>
    
    This is especially important because we often switch between different repositories in different runs.
    Make sure the remote URL matches the repository you intend to work with!
    
    GitHub Setup Commands (run these first when working with GitHub):
    <tool_request>
    {{"tool_name": "run_code", "code": "git config --global user.name '{GITHUB_USERNAME}' && git config --global user.email '{GITHUB_NOREPLY_EMAIL}'", "language": "shell"}}
    </tool_request>
    
    GitHub Authentication (use PAT for HTTPS operations):
    - Clone: git clone https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/repo-name.git
    - Push: git push https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/repo-name.git
    
    ✅ CORRECT WORKFLOW FOR CLONING AND CREATING FEATURE BRANCH:
    1. Clone the repository and change into directory:
    <tool_request>
    {{"tool_name": "run_code", "code": "git clone https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/repo-name.git && cd repo-name && pwd", "language": "shell"}}
    </tool_request>
    
    2. Verify you're in the correct repository and check branches:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git remote -v && git branch -v", "language": "shell"}}
    </tool_request>
    
    3. Create feature branch FROM main (ensures shared history):
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git checkout main && git checkout -b feature-branch && git branch -v && git log --oneline -5", "language": "shell"}}
    </tool_request>
    
    ✅ CORRECT WORKFLOW FOR MAKING CHANGES AND CREATING PR:
    1. Check issue and verify repository:
    <tool_request>
    {{"tool_name": "run_code", "code": "curl -s -H 'Authorization: token {GITHUB_PAT}' https://api.github.com/repos/{GITHUB_USERNAME}/repo-name/issues/1 | grep -E '(title|body)' | head -10 && cd repo-name && pwd && git remote -v", "language": "shell"}}
    </tool_request>
    
    2. Make changes, add, commit and verify:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git status && git add -A && git commit -m 'Clear description of changes' && git status", "language": "shell"}}
    </tool_request>
    
    3. Verify remote and push branch:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git remote -v && git push -u origin feature-branch && git branch -r | grep feature-branch", "language": "shell"}}
    </tool_request>
    
    4. Double-check repository and create PR:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git remote -v | head -1 && curl -i -X POST -H 'Authorization: token {GITHUB_PAT}' -H 'Content-Type: application/json' -d '{{\\"title\\":\\"Descriptive PR Title\\",\\"body\\":\\"Fixes #1\\\\n\\\\nDetailed description\\",\\"head\\":\\"feature-branch\\",\\"base\\":\\"main\\"}}' https://api.github.com/repos/{GITHUB_USERNAME}/repo-name/pulls", "language": "shell"}}
    </tool_request>
    
    ⚠️ COMMON PITFALLS TO AVOID:
    1. Creating branches before cloning is complete
    2. Not changing into the repository directory after cloning
    3. Creating orphan branches with no shared history
    4. Not checking if operations succeeded before proceeding
    5. Making up fake successful responses
    6. **Working with the wrong repository - ALWAYS verify git remote -v**
    7. **Pushing/pulling to/from wrong repo because you switched repos between runs**
    
    SANITY CHECK COMMANDS:
    
    Check current directory and repository:
    <tool_request>
    {{"tool_name": "run_code", "code": "pwd && git remote -v && git branch -v", "language": "shell"}}
    </tool_request>
    
    Verify branches share history (before creating PR):
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo-name && git merge-base main feature-branch", "language": "shell"}}
    </tool_request>
    
    Check if PR already exists:
    <tool_request>
    {{"tool_name": "run_code", "code": "curl -s -H 'Authorization: token {GITHUB_PAT}' https://api.github.com/repos/{GITHUB_USERNAME}/repo-name/pulls?state=all | grep -E '(title|number|state)' | head -20", "language": "shell"}}
    </tool_request>
    
    GitHub API Error Handling:
    - 422 Unprocessable Entity: Often means branches have no shared history
    - 404 Not Found: Repository doesn't exist or no access
    - 401 Unauthorized: PAT is invalid or expired
    - 403 Forbidden: PAT lacks required permissions
    
    Always use -i flag with curl to see status codes:
    <tool_request>
    {{"tool_name": "run_code", "code": "curl -i -H 'Authorization: token {GITHUB_PAT}' https://api.github.com/repos/{GITHUB_USERNAME}/repo-name", "language": "shell"}}
    </tool_request>
    
    Always use -i flag with curl to see status codes:
    <tool_request>
    {{"tool_name": "run_code", "code": "curl -i -H 'Authorization: token {GITHUB_PAT}' https://api.github.com/repos/{GITHUB_USERNAME}/repo-name", "language": "shell"}}
    </tool_request>
""".format(
    GITHUB_USERNAME=GITHUB_USERNAME,
    GITHUB_PAT=GITHUB_PAT,
    GITHUB_EMAIL=GITHUB_EMAIL,
    GITHUB_NOREPLY_EMAIL=GITHUB_NOREPLY_EMAIL,
)

RUN_CODE_TOOL_PROMPT = """
    ----- 
    RUN CODE TOOL - Execute Python or shell code safely in a sandboxed environment.
    
    FORMAT: 
    <tool_request>
    {{"tool_name": "run_code", "code": "code_goes_here", "language": "python"}}
    </tool_request>
    
    PARAMETERS:
    - code (required): Code string to execute (max 100 lines)
    - language (optional, default "python"): "python" or "shell" -- NO other languages supported
    
    EXECUTION ENVIRONMENT:
    - Runs in isolated sandbox/container 
    - Fresh container for each execution - no state persistence
    - Limited execution time: 10 seconds timeout
    - Resource limits: 100MB memory, 10 second CPU time, 1MB file size
    - No internet access or external network connections
    - You have access to a GitHub account/credentials to manage version control 
    - Be mindful of which directory you are in and where you are importing from when importing files you created 
        - If imports are failing or giving ModuleNotFound, double check this: imports should be made to work!
    
    AVAILABLE PYTHON IMPORTS:
    ✅ Standard Library (safe subset): math, random, datetime, json, csv, re, itertools, collections, statistics, decimal, fractions, typing, dataclasses, enum
    ❌ BLOCKED: os, sys, subprocess, socket, urllib.request, requests, threading, multiprocessing, any third-party packages
    
    EXAMPLES:
    
    Python Examples:
    <tool_request>
    {{"tool_name": "run_code", "code": "print('Hello World!')"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n\\nprint(factorial(5))", "language": "python"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "import json  \\ndata = {{'name': 'AI', 'tasks': ['search', 'code', 'analyze']}}\\nprint(json.dumps(data, indent=2))"}}
    </tool_request>
    
    Shell Examples:
    <tool_request>
    {{"tool_name": "run_code", "code": "echo 'Hello from shell'", "language": "shell"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "for i in {{1..5}}; do echo \\"Count: $i\\"; done", "language": "shell"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "ls -la", "language": "shell"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "pwd && find . -type f -name '*.txt' | head -10", "language": "shell"}}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{{"tool_name": "execute", "code": "print('test')"}}</tool_request>
    
    ❌ WRONG - Using "script" instead of "code":
    <tool_request>{{"tool_name": "run_code", "script": "print('test')"}}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{{'tool_name': 'run_code', 'code': 'print("test")'}}</tool_request>
    
    ❌ WRONG - Using unsupported language:
    <tool_request>{{"tool_name": "run_code", "code": "console.log('hello')", "language": "javascript"}}</tool_request>
    
    ❌ WRONG - Expecting third-party libraries:
    <tool_request>{{"tool_name": "run_code", "code": "import numpy as np\\nprint(np.array([1,2,3]))"}}</tool_request>
    
    ❌ WRONG - Unescaped newlines in code:
    <tool_request>{{"tool_name": "run_code", "code": "print('Hello'\\n'World!')"}}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in code:
    <tool_request>{{"tool_name": "run_code", "code": "print(\\"Hello\\")"}}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{{"tool_name": "run_code", "code": "print('test')",}}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{{"tool_name": "run_code", "code": "with open('C:\\temp\\file.txt') as f: print(f.read())"}}</tool_request>

    ❌ WRONG - Creating files outside cloned repository:
    <tool_request>{{"tool_name": "write_file", "path": "script.py", "content": "# This won't be in the repo!"}}</tool_request>
        
        ✅ CORRECT - Creating files inside cloned repository:
        <tool_request>{{"tool_name": "write_file", "path": "repo-name/script.py", "content": "# This will be in the repo"}}</tool_request>
    
    SECURITY RESTRICTIONS:
    - Code executes in sandboxed container with 10-second timeout
    - NO file I/O outside isolated directory, NO network access (except GitHub via git/curl), NO system commands
    - NO dangerous imports (os, sys, subprocess), NO third-party packages
    - Resource limits: 100MB memory, 10 second CPU time, 1MB file size
    - Fresh container for each execution - no persistence between runs
    
    DEBUGGING & ITERATION:
    - Tool returns both stdout and stderr for error analysis
    - Feel free to iteratively debug and improve code until it runs correctly
    - Each execution is independent - no variables persist between runs
    ----
    GITHUB INTEGRATION DETAILS: 

        {GITHUB_TEXT}
    ---- 
    
        TOOL INTERACTIONS & WORKFLOWS:
    - Use read_file first to understand existing file structure
    - Combine with run_code to test code before writing files
    - Test written code with run_code after file creation
    
    ⚠️ CRITICAL FOR GITHUB REPOSITORIES:
    - When working with cloned repositories, ALWAYS create files INSIDE the repository directory
    - Use paths like "repo-name/filename.py" or first cd into the repo directory
    - Files created in the agent_scratch root won't be part of the repository
    - WRONG: write_file with path "script.py" when you need "repo-name/script.py"
    - ALWAYS verify file location with "cd repo-name && ls -la" after creating files

    ⚠️ CRITICAL DIRECTORY AWARENESS:
    - Each run_code call starts fresh in agent_scratch directory
    - cd commands DO NOT persist between run_code calls
    - ALWAYS chain commands with && or use full paths
    - WRONG: run_code("cd repo") then run_code("git status")
    - RIGHT: run_code("cd repo && git status")
    
    ⚠️ CRITICAL SHELL PROCESS INITIALIZATION:
    - BEFORE beginning any series of shell processes, ALWAYS check your current state:
    - Run: pwd && ls -la && git remote -v (if in a git repository)
    - This helps you understand where you are and what repository you're working with
    - Example initialization check:
    <tool_request>
    {{"tool_name": "run_code", "code": "pwd && ls -la && git remote -v", "language": "shell"}}
    </tool_request>
    - This prevents working in wrong directories or wrong repositories
    
    ⚠️ CRITICAL BRANCH WORKFLOW - NEVER PUSH TO MAIN:
    - ALWAYS create a feature branch IMMEDIATELY after cloning, BEFORE making any changes
    - ALL work must happen on the feature branch, NEVER on main
    - PRs are created from feature branch to main, NEVER main to main
    - NEVER commit directly to main branch
    
    ✅ CORRECT BRANCH WORKFLOW:
    1. Clone repository
    2. IMMEDIATELY create and switch to feature branch (before any changes)
    3. Make all changes on feature branch
    4. Commit and push feature branch
    5. Create PR from feature branch to main
    
    ❌ WRONG - Making changes on main:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo && git checkout main && echo 'changes' > file.txt && git add . && git commit -m 'changes'", "language": "shell"}}
    </tool_request>
    
    ❌ WRONG - Creating PR from main to main:
    curl -X POST -H "Authorization: token TOKEN" \\
      -d '{{"title": "Fix", "head": "main", "base": "main"}}' \\
      https://api.github.com/repos/USERNAME/repo/pulls
    
    ✅ CORRECT - Feature branch workflow:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd repo && git checkout -b feature-fix && echo 'changes' > file.txt && git add . && git commit -m 'changes' && git push -u origin feature-fix", "language": "shell"}}
    </tool_request>
    
    ✅ CORRECT - Creating PR from feature to main:
    curl -X POST -H "Authorization: token TOKEN" \\
      -d '{{"title": "Fix", "head": "feature-fix", "base": "main"}}' \\
      https://api.github.com/repos/USERNAME/repo/pulls



    IMPORTANT: Never claim you "ran code" or "executed git commands" unless you actually made a <tool_request> with the run_code tool.
    
    CODE GUIDELINES:
    - Keep code under 100 lines for performance
    - Use "python" for calculations, algorithms, data processing
    - Use "shell" for basic system commands within sandbox AND for Git operations
    - Don't expect third-party libraries or network access (except GitHub via git/curl)
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Don't hesitate to debug iteratively - multiple tool calls are encouraged
    - When asked to work with GitHub: ACTUALLY RUN the git commands using run_code, don't just print or simulate them
    - Always use the provided PAT for GitHub authentication
    - **ALWAYS verify git remote -v before any push/pull/PR operations - we often switch repos between runs**
    ----- 
""".format(
    GITHUB_TEXT=GITHUB_TEXT, 
)



WRITE_FILE_TOOL_PROMPT = """
    ----- 
    WRITE FILE TOOL - Create or overwrite files safely within the sandbox directory.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "write_file", "path": "filename.ext", "content": "file_content_here"}
    </tool_request>
    
    PARAMETERS:
    - path (required): Relative file path (no absolute paths or directory traversal)
    - content (required): File content as string (max 1MB)
    
    EXAMPLES:
    
    Basic File Creation:
    <tool_request>
    {"tool_name": "write_file", "path": "script.py", "content": "print('Hello World')"}
    </tool_request>
    <tool_request>
    {"tool_name": "write_file", "path": "data.json", "content": "{\"name\": \"value\", \"number\": 42}"}
    </tool_request>
    <tool_request>
    {"tool_name": "write_file", "path": "README.md", "content": "# My Project\n\nThis is a sample project."}
    </tool_request>
    Nested Directory Creation:
    <tool_request>
    {"tool_name": "write_file", "path": "src/neural_network.py", "content": "class NeuralNet:\n    def __init__(self, layers):\n        self.layers = layers\n        self.weights = []\n    \n    def forward(self, x):\n        # Simple feedforward implementation\n        return x"}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "create_file", "path": "file.py", "content": "code"}</tool_request>
    
    ❌ WRONG - Wrong field names:
    <tool_request>{"tool_name": "write_file", "filename": "file.py", "text": "code"}</tool_request>
    
    ❌ WRONG - Absolute paths:
    <tool_request>{"tool_name": "write_file", "path": "/home/user/file.py", "content": "code"}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'write_file', 'path': 'file.py', 'content': 'code'}</tool_request>
    
    ❌ WRONG - Missing content field:
    <tool_request>{"tool_name": "write_file", "path": "file.py"}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in content:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "print("Hello")"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "print('hi')",}</tool_request>
    
    ❌ WRONG - Newlines in content not properly escaped:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "line1
    line2"}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "C:\folder\file.txt"}</tool_request>
    
    SECURITY RESTRICTIONS:
    - Files created only within the sandbox directory
    - Directory traversal attempts (../) are blocked
    - Maximum file size: 1MB
    - Automatically creates parent directories as needed
    - Completely overwrites existing files (no append mode)
    - Only allowed extensions: .txt, .md, .py, .sh, .json, .yaml, .yml, .html, .css, .js
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use read_file first to understand existing file structure
    - Combine with run_code to test code before writing files
    - Test written code with run_code after file creation

    ⚠️ CRITICAL FOR REPOSITORIES:
    - write_file ALWAYS will writes relative to agent_scratch root
    - To write inside a cloned repo, use: "repo-name/filename.py"
    - NEVER assume you're "inside" a directory from previous commands
    - ALWAYS use the full path including the repository name
    
    IMPORTANT: Never claim you "wrote a file" unless you actually made a <tool_request> with the write_file tool.
    
    GUIDELINES:
    - Use relative paths only - no absolute paths or directory traversal
    - Ensure content is properly escaped for JSON
    - Parent directories are created automatically
    - Files are completely overwritten - use read_file first if preserving content
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "write_file" with exact field names "path" and "content"
    ----- 
"""

READ_FILE_TOOL_PROMPT = """
    ----- 
    READ FILE TOOL - Read contents of existing files safely within the sandbox directory.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "read_file", "path": "filename.ext"}
    </tool_request>
    
    PARAMETERS:
    - path (required): Relative file path (no absolute paths or directory traversal)
    
    EXAMPLES:
    
    Basic File Reading:
    <tool_request>
    {"tool_name": "read_file", "path": "script.py"}
    </tool_request>
    <tool_request>
    {"tool_name": "read_file", "path": "data.json"}
    </tool_request>
    <tool_request>
    {"tool_name": "read_file", "path": "README.md"}
    </tool_request>
    
    Advanced Use Case:
    <tool_request>
    {"tool_name": "read_file", "path": "models/transformer_config.yaml"}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "read_from_file", "path": "file.py"}</tool_request>
    
    ❌ WRONG - Wrong field names:
    <tool_request>{"tool_name": "read_file", "filename": "file.py"}</tool_request>
    
    ❌ WRONG - Absolute paths:
    <tool_request>{"tool_name": "read_file", "path": "/etc/passwd"}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'read_file', 'path': 'file.py'}</tool_request>
    
    ❌ WRONG - Missing path field:
    <tool_request>{"tool_name": "read_file"}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in path:
    <tool_request>{"tool_name": "read_file", "path": "my"file.py"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "read_file", "path": "file.py",}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "read_file", "path": "C:\folder\file.txt"}</tool_request>
    
    SECURITY RESTRICTIONS:
    - Files read only from the sandbox directory
    - Directory traversal attempts (../) are blocked
    - Returns empty content if file doesn't exist or can't be read
    - UTF-8 encoding assumed for all files
    
    FILE EXISTENCE HANDLING:
    - If file exists: Returns full file content
    - If file doesn't exist: Returns empty string (no error)
    - Always check if returned content is empty to handle missing files
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use before write_file to understand existing code structure
    - Read multiple related files to understand project structure
    - Combine with search tool for understanding existing codebases
    - Use with run_code to read, test, and modify existing code iteratively
    
    IMPORTANT: Never claim you "read a file" unless you actually made a <tool_request> with the read_file tool.
    
    GUIDELINES:
    - Use relative paths only - no absolute paths or directory traversal
    - Check if returned content is empty (might indicate missing file)
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "read_file" with exact field name "path"
    - Multiple read operations are fast and encouraged
    ----- 
"""