import threading
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt

# Initialize lists for metrics
time_points = []
gpu_usage = []
gpu_memory = []
ram_usage = []
monitoring_flag = True
def monitor_resources(interval=1):
    """Monitor GPU and RAM usage."""
    print("Monitoring resources... (Press Ctrl+C to stop after training ends)")
    while monitoring_flag:
        # Log current time
        time_points.append(time.time())

        # GPU stats
        gpus = GPUtil.getGPUs()
        gpu_usage.append(gpus[0].load * 100)  # GPU usage in %
        gpu_memory.append(gpus[0].memoryUsed)  # GPU memory in MB

        # RAM stats
        ram = psutil.virtual_memory()
        ram_usage.append(ram.used / 1e9)  # Convert to GB

        time.sleep(interval)
