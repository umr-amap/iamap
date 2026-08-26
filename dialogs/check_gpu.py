import subprocess
import platform
import shutil


def check_nvidia_gpu():

    nvidia_smi_path = shutil.which("nvidia-smi")
    print(nvidia_smi_path)
    if nvidia_smi_path :
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(
            [nvidia_smi_path, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            shell=False, ## explicitely state for bandit against shell injection
        )
        output = output.decode("utf-8").strip()

        # Parse the output
        gpu_info = output.split(",")
        gpu_name = gpu_info[0].strip()

        output_cuda_version = subprocess.run(
            [nvidia_smi_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False
        )
        for line in output_cuda_version.stdout.split("\n"):
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version: ")[1].split()[0]

        return True, gpu_name, cuda_version
    else:
        return False, None, None


def check_amd_gpu():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                universal_newlines=True,
                shell=False,
            )
            if "AMD" in output or "Radeon" in output:
                return True
        elif platform.system() == "Linux":
            output = subprocess.check_output(["lspci"], universal_newlines=True, shell=False)
            if "AMD" in output or "Radeon" in output:
                return True
        elif platform.system() == "Darwin":
            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], universal_newlines=True, shell=False
            )
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
        return "amd"
    return "cpu"
