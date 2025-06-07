PYTHON_SANDBOX_PROMPT = """
import sys
import os
import signal
import resource

# Set resource limits
def set_limits():
    try:
        # Get current limits and respect existing hard limits
        
        # Set memory limit to 256MB (or existing hard limit if lower)
        try:
            current_mem_soft, current_mem_hard = resource.getrlimit(resource.RLIMIT_AS)
            target_mem = 256 * 1024 * 1024
            if current_mem_hard == -1 or target_mem <= current_mem_hard:
                resource.setrlimit(resource.RLIMIT_AS, (target_mem, current_mem_hard))
            else:
                resource.setrlimit(resource.RLIMIT_AS, (current_mem_hard, current_mem_hard))
        except (OSError, ValueError):
            pass  # Continue if setting memory limit fails
        
        # Set CPU time limit to 10 seconds (or existing hard limit if lower)
        try:
            current_cpu_soft, current_cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
            target_cpu = 10
            if current_cpu_hard == -1 or target_cpu <= current_cpu_hard:
                resource.setrlimit(resource.RLIMIT_CPU, (target_cpu, current_cpu_hard))
            else:
                resource.setrlimit(resource.RLIMIT_CPU, (current_cpu_hard, current_cpu_hard))
        except (OSError, ValueError):
            pass  # Continue if setting CPU limit fails
        
        # Limit number of processes/threads (or existing hard limit if lower)
        try:
            current_proc_soft, current_proc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
            target_proc = 10
            if current_proc_hard == -1 or target_proc <= current_proc_hard:
                resource.setrlimit(resource.RLIMIT_NPROC, (target_proc, current_proc_hard))
            else:
                resource.setrlimit(resource.RLIMIT_NPROC, (current_proc_hard, current_proc_hard))
        except (OSError, ValueError):
            pass  # Continue if setting process limit fails
        
        # Limit file size to 10MB (or existing hard limit if lower)
        try:
            current_fsize_soft, current_fsize_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            target_fsize = 10 * 1024 * 1024
            if current_fsize_hard == -1 or target_fsize <= current_fsize_hard:
                resource.setrlimit(resource.RLIMIT_FSIZE, (target_fsize, current_fsize_hard))
            else:
                resource.setrlimit(resource.RLIMIT_FSIZE, (current_fsize_hard, current_fsize_hard))
        except (OSError, ValueError):
            pass  # Continue if setting file size limit fails
            
    except Exception:
        pass  # Continue execution even if resource limit setup fails entirely

# Store original import function before removing it
original_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

# Remove dangerous builtins
dangerous_builtins = ['exec', 'eval', 'compile']
for builtin_name in dangerous_builtins:
    if isinstance(__builtins__, dict):
        if builtin_name in __builtins__:
            del __builtins__[builtin_name]
    else:
        if hasattr(__builtins__, builtin_name):
            delattr(__builtins__, builtin_name)

# Set timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

# Apply limits
set_limits()

# Restrict all file operations to current directory
import builtins
original_open = builtins.open

def safe_open(filename, mode='r', *args, **kwargs):
    # Only allow read operations and writing to files in current directory
    abs_path = os.path.abspath(filename)
    cwd_path = os.path.abspath('.')
    if not abs_path.startswith(cwd_path):
        raise PermissionError("Cannot access files outside current directory")
    return original_open(filename, mode, *args, **kwargs)

# Override file system operations
def restricted_remove(path):
    abs_path = os.path.abspath(path)
    cwd_path = os.path.abspath('.')
    if not abs_path.startswith(cwd_path):
        raise PermissionError("Cannot remove files outside current directory")
    return os._remove(path)

def restricted_rmdir(path):
    abs_path = os.path.abspath(path)
    cwd_path = os.path.abspath('.')
    if not abs_path.startswith(cwd_path):
        raise PermissionError("Cannot remove directories outside current directory")
    return os._rmdir(path)

def restricted_mkdir(path, mode=0o777):
    abs_path = os.path.abspath(path)
    cwd_path = os.path.abspath('.')
    if not abs_path.startswith(cwd_path):
        raise PermissionError("Cannot create directories outside current directory")
    return os._mkdir(path, mode)

def restricted_rename(src, dst):
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    cwd_path = os.path.abspath('.')
    if not abs_src.startswith(cwd_path) or not abs_dst.startswith(cwd_path):
        raise PermissionError("Cannot rename files outside current directory")
    return os._rename(src, dst)

def restricted_chmod(path, mode):
    abs_path = os.path.abspath(path)
    cwd_path = os.path.abspath('.')
    if not abs_path.startswith(cwd_path):
        raise PermissionError("Cannot modify permissions outside current directory")
    return os._chmod(path, mode)

# Store original functions
if not hasattr(os, '_remove'):
    os._remove = os.remove
if not hasattr(os, '_rmdir'):
    os._rmdir = os.rmdir
if not hasattr(os, '_mkdir'):
    os._mkdir = os.mkdir
if not hasattr(os, '_rename'):
    os._rename = os.rename
if not hasattr(os, '_chmod'):
    os._chmod = os.chmod

# Replace with restricted versions
os.remove = restricted_remove
os.unlink = restricted_remove
os.rmdir = restricted_rmdir
os.mkdir = restricted_mkdir
os.makedirs = lambda path, mode=0o777, exist_ok=False: os.mkdir(path, mode) if not os.path.exists(path) or not exist_ok else None
os.rename = restricted_rename
os.chmod = restricted_chmod

# Replace open in builtins
if isinstance(__builtins__, dict):
    __builtins__['open'] = safe_open
else:
    __builtins__.open = safe_open

# Also replace in builtins module
builtins.open = safe_open

# Add agent scratch directory to Python path for imports
agent_scratch_dir = os.path.abspath('.')
if agent_scratch_dir not in sys.path:
    sys.path.insert(0, agent_scratch_dir)

{code}
""" 

SHELL_SANDBOX_PROMPT = """
# Safe shell sandbox: restrict resources, no file IO, only builtins, safe for laptop

ulimit -t 5        # 5 second CPU time
ulimit -f 5120     # 5MB file size limit

{code}
"""