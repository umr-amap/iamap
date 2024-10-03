import shutil
import os
import torch
import logging
import hashlib
from PyQt5.QtCore import QVariant

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

def compute_md5_hash(parameters,keys_to_remove = ['MERGE_METHOD', 'WORKERS', 'PAUSES']):
        param_encoder = {key: parameters[key] for key in parameters if key not in keys_to_remove}
        return hashlib.md5(str(param_encoder).encode("utf-8")).hexdigest()

def convert_qvariant_obj(obj):
    if isinstance(obj, QVariant):
        return obj.value()  # Extract the native Python value from QVariant
    else:
        return obj

def convert_qvariant(obj):
    if isinstance(obj, QVariant):
        return obj.value()  # Extract the native Python value from QVariant
    elif isinstance(obj, dict):
        return {key: convert_qvariant_obj(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_qvariant_obj(item) for item in obj]
    else:
        return obj
def save_parameters_to_json(parameters, output_dir):
    dst_path = os.path.join(output_dir, 'parameters.json')
    ## convert_qvariant does not work properly for 'CKPT'
    ## converting it to a str
    converted_parameters = convert_qvariant(parameters) 
    print(parameters)
    converted_parameters['CKPT'] = str(converted_parameters['CKPT'])

    for key, item in converted_parameters.items():
        print(key, type(item))
    with open(dst_path, "w") as json_file:
        json.dump(converted_parameters, json_file, indent=4)

