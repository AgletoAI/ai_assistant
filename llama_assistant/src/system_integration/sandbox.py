
import subprocess

def safe_system_call(command):
    # Implement safety checks here
    if is_safe_command(command):
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
    else:
        return "Error: Unsafe command"

def is_safe_command(command):
    # Implement command safety validation logic
    unsafe_keywords = ['rm', 'del', 'format', 'mkfs']
    return not any(keyword in command for keyword in unsafe_keywords)
