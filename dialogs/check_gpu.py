import subprocess
import platform

def check_nvidia_gpu():
    try:
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8").strip()
        
        # Parse the output
        gpu_info = output.split(',')
        gpu_name = gpu_info[0].strip()
        cuda_version = gpu_info[2].strip()
        
        return True, gpu_name, cuda_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None, None

def check_amd_gpu():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(["wmic", "path", "win32_videocontroller", "get", "name"], universal_newlines=True)
            if "AMD" in output or "Radeon" in output:
                return True
        elif platform.system() == "Linux":
            output = subprocess.check_output(["lspci"], universal_newlines=True)
            if "AMD" in output or "Radeon" in output:
                return True
        elif platform.system() == "Darwin":
            output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], universal_newlines=True)
            if "AMD" in output or "Radeon" in output:
                return True
    except subprocess.CalledProcessError:
        return False
    return False

def has_gpu():
    has_nvidia, gpu_name, cuda_version = check_nvidia_gpu()
    if has_nvidia:
        return cuda_version
    if check_amd_gpu():
        return 'amd'
    return 'cpu'
