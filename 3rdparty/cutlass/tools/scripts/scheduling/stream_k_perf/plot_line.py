import argparse
import csv
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str)
parser.add_argument("outdir", type=str)
args = parser.parse_args()

if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

# Keys for CSV fields
KEY_K = "K"
KEY_TILES = "Tiles"
KEY_DP = "Baseline-GFLOPs/s"
KEY_SK = "New-GFLOPs/s"

results_by_k = {}
with open(args.infile, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    for row in csv_reader:
        k = int(row[KEY_K])
        tiles = int(row[KEY_TILES])
        perf_dp = float(row[KEY_DP])
        perf_sk = float(row[KEY_SK])

        if k not in results_by_k:
            results_by_k[k] = {"dp": [], "sk": [], "tiles": []}
        results_by_k[k]["dp"].append(perf_dp)
        results_by_k[k]["sk"].append(perf_sk)
        results_by_k[k]["tiles"].append(tiles)

for k, vals in sorted(results_by_k.items()):
    _, ax = plt.subplots()
    ax.plot(vals["tiles"], vals["dp"], label="Baseline")
    ax.plot(vals["tiles"], vals["sk"], label="New")
    ax.set_xlabel("Output tiles")
    ax.set_ylabel("GFLOPs/s")
    ax.set_title(f"K = {k}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    plt.savefig(os.path.join(args.outdir, f"{k}.png"), bbox_inches="tight")
    plt.clf()
