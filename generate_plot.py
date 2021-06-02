import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read all the timings
all_data = {}
for f in os.listdir("report"):
    if f != ".gitignore":
        print(f)
        with open("report/" + f) as fl:
            data = json.load(fl)
        all_data[f] = data

# mean timings by config
mean_times = []
for k, v in all_data.items():
    for batch_size, timings in v.items():
        mean_times.append((k, batch_size, np.mean(timings)))

mean_times = pd.DataFrame(
    mean_times, columns=["config", "batch_size", "latency"]
)

# categories
updated_configs = []
for c in mean_times.config:
    updated_config = ""
    if "cpu_True" in c:
        updated_config = "cpu"
    else:
        updated_config = "gpu"
    if "onnx_True" in c:
        updated_config += "/pytorch+onnx"
    else:
        updated_config += "/pytorch"
    if "fp16_True" in c:
        updated_config += "/fp16"
    else:
        updated_config += "/fp32"
    updated_configs.append(updated_config)
mean_times["updated_config"] = updated_configs

mean_times["device"] = mean_times.updated_config.apply(
    lambda x: "cpu" if "cpu" in x else "gpu"
)

# cpu plots
cpu_data = mean_times[mean_times.device == "cpu"]
cpu_data.sort_values(by=["batch_size", "latency"], inplace=True)
fig, axs = plt.subplots(cpu_data.batch_size.nunique(), 1, figsize=(10, 6))
for i, batch_size in enumerate(
    sorted(list(map(int, cpu_data.batch_size.unique().tolist())))
):
    sns.barplot(
        ax=axs[i],
        data=cpu_data[mean_times.batch_size == str(batch_size)].sort_values(
            by="latency", ascending=False
        ),
        y="batch_size",
        x="latency",
        hue="updated_config",
    )
plt.tight_layout()
plt.savefig("imgs/cpu.png")

# gpu plots
cpu_data = mean_times[mean_times.device == "gpu"]
cpu_data.sort_values(by=["batch_size", "latency"], inplace=True)
fig, axs = plt.subplots(cpu_data.batch_size.nunique(), 1, figsize=(10, 6))
for i, batch_size in enumerate(
    sorted(list(map(int, cpu_data.batch_size.unique().tolist())))
):
    sns.barplot(
        ax=axs[i],
        data=cpu_data[mean_times.batch_size == str(batch_size)].sort_values(
            by="latency", ascending=False
        ),
        y="batch_size",
        x="latency",
        hue="updated_config",
    )
plt.tight_layout()
plt.savefig("imgs/gpu.png")
