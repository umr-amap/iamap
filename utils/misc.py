import shutil
import os
import torch
import logging

class QGISLogHandler(logging.Handler):
    def __init__(self, feedback):
        super().__init__()
        self.feedback = feedback

    def emit(self, record):
        msg = self.format(record)
        self.feedback.pushInfo(msg)

def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

def check_disk_space(path):
    # Get disk usage statistics about the given path
    total, used, free = shutil.disk_usage(path)
    
    # Convert bytes to a more readable format (e.g., GB)
    total_gb = total / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    
    return total_gb, used_gb, free_gb

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total / (1024 ** 3)

def remove_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")


def get_unique_filename(directory, filename, layer_name='merged features'):
    """
    Check if the filename exists in the given directory. If it does, append a numbered suffix.
    :param directory: The directory where the file will be saved.
    :param filename: The desired filename (e.g. in our case, "merged.tif").
    :return: A unique filename with an incremented suffix if necessary, a unique layer name.
    """
    base, ext = os.path.splitext(filename)
    candidate = filename
    updated_layer_name = layer_name
    i = 1
    
    # Check if file exists and update filename
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}-{i}{ext}"
        updated_layer_name = f"{layer_name} {i}"
        i += 1

    return os.path.join(directory, candidate), updated_layer_name

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    ## https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
