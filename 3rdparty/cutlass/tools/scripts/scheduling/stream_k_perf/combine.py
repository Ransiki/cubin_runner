import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("baseline")
parser.add_argument("new")
args = parser.parse_args()

# Keys for CSV fields
KEY_M = "M"
KEY_N = "N"
KEY_K = "K"
KEY_TILES = "Tiles"
KEY_SK = "New-GFLOPs/s"

results = []
with open(args.baseline, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    for row in csv_reader:
        m = int(row[KEY_M])
        n = int(row[KEY_N])
        k = int(row[KEY_K])
        tiles = int(row[KEY_TILES])
        perf_sk = float(row[KEY_SK])
        results.append([m,n,k,tiles,perf_sk])

with open(args.new, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    for i, row in enumerate(csv_reader):
        m = int(row[KEY_M])
        n = int(row[KEY_N])
        k = int(row[KEY_K])
        tiles = int(row[KEY_TILES])
        perf_sk = float(row[KEY_SK])
        assert m == results[i][0]
        assert n == results[i][1]
        assert k == results[i][2]
        assert tiles == results[i][3]
        perf_old = results[i][-1]
        results[i].append(perf_sk)
        speedup = 100 * (perf_sk - perf_old) / perf_old
        results[i].append(speedup)

print("K,M,N,Tiles,Baseline-GFLOPs/s,New-GFLOPs/s,Speedup")
for m,n,k,tiles,perf_base,perf_new,speedup in results:
    print(f"{k},{m},{n},{tiles},{perf_base},{perf_new},{speedup:.2f}")
