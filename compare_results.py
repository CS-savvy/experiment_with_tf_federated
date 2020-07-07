from matplotlib import pyplot as plt
from utils import plot_graph
from pathlib import Path

this_dir = Path.cwd()

experiment_name = "mnist"
method_1 = "keras_training"
method_2 = "tff_training"

metric_file_1 = this_dir / "results" / experiment_name / method_1 / (experiment_name + ".txt")
metric_file_2 = this_dir / "results" / experiment_name / method_2 / (experiment_name + ".txt")

output_dir = this_dir / "results" / experiment_name / "compare"
if not output_dir.exists():
    output_dir.mkdir(parents=True)


metric_1 = {}
with open(metric_file_1, "r") as f:
    data = f.read()
    for line in data.split("\n"):
        line = line.split(" ")
        metric_1[line[0]] = [float(l) for l in line[1:]]

metric_2 = {}
with open(metric_file_2, "r") as f:
    data = f.read()
    for line in data.split("\n"):
        line = line.split(" ")
        metric_2[line[0]] = [float(l) for l in line[1:]]

common_metrics = set(metric_1.keys()).intersection(set(metric_2.keys()))
print("Common metrics : ", common_metrics)

for m in common_metrics:
    plt.figure(figsize=(10, 6))
    plt.suptitle(m, fontsize=15)
    plot_graph(range(1, len(metric_1[m]) + 1), metric_1[m], label=method_1)
    plot_graph(list(range(1, len(metric_2[m])*5 + 1))[4::5], metric_2[m], label=method_2)
    plt.legend()
    plt.savefig(output_dir / (m + ".png"))