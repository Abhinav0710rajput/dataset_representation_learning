import torch

# Step 1: Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()

# Step 2: List all CUDA devices
num_devices = torch.cuda.device_count()
print(f"Available CUDA devices ({num_devices}):")
for i in range(num_devices):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

# Step 3: Select a specific device (change index here)
device_index = 1  # ← Change this to the desired device number

if device_index >= num_devices:
    print(f"Device index {device_index} is out of range.")
    exit()

device = torch.device(f"cuda:{device_index}")
torch.cuda.set_device(device)

print(f"\n✅ Using device: {device} — {torch.cuda.get_device_name(device_index)}")


import torch

if torch.cuda.is_available():
    default_device_index = torch.cuda.current_device()
    default_device = torch.device(f"cuda:{default_device_index}")
    device_name = torch.cuda.get_device_name(default_device_index)
    print(f"Default CUDA device is: {default_device} — {device_name}")
else:
    print("CUDA is not available.")




# import psutil

# # Get overall CPU usage percentage (averaged across all cores)
# cpu_percent = psutil.cpu_percent(interval=1)  # interval=1 means it waits 1 sec to get usage
# print(f"🧠 Current CPU usage: {cpu_percent}%")

# # Get per-core CPU usage
# per_core = psutil.cpu_percent(interval=1, percpu=True)
# for i, usage in enumerate(per_core):
#     print(f"  Core {i}: {usage}%")


# import psutil

# # List all Python processes
# workers = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) if 'python' in (p.info['name'] or '')]

# print(f"🔧 Found {len(workers)} Python worker process(es):")
# for p in workers:
#     print(f"  PID {p.pid}: {' '.join(p.info['cmdline'])}")
 
"""

  [0] NVIDIA TITAN Xp
  [1] NVIDIA TITAN X (Pascal)
  [2] NVIDIA GeForce GTX 1080 Ti
  [3] NVIDIA GeForce GTX 1080 Ti
  [4] NVIDIA GeForce GTX 1080 Ti
  [5] NVIDIA GeForce GTX 1080 Ti
  [6] NVIDIA GeForce GTX 1060 6GB

"""