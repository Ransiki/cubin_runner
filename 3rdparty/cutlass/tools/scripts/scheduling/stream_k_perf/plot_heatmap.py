import argparse
import csv
import matplotlib
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str)
parser.add_argument("outfile", type=str)
parser.add_argument("--limit_tiles", type=int, help="Only plot entries with tiles less than or equal to this value")
args = parser.parse_args()

# Keys for CSV fields
KEY_K = "K"
KEY_TILES = "Tiles"
KEY_SPEEDUP = "Speedup"

tile_set = set()
k_set = set()

results = {}
with open(args.infile, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    for row in csv_reader:
        k = int(row[KEY_K])
        tiles = int(row[KEY_TILES])
        speedup = float(row[KEY_SPEEDUP])

        if args.limit_tiles is not None and tiles >= args.limit_tiles:
            # Skip entries with a number of tiles above the limit
            continue

        tile_set.add(tiles)
        k_set.add(k)
        results[f"{k},{tiles}"] = speedup

result_matrix = [[None for _ in range(len(k_set))] for _ in range(len(tile_set))]
k_vals = sorted(list(k_set))
tile_vals = sorted(list(tile_set))
for key, speedup in results.items():
    k, tiles = [int(x) for x in key.split(',')]
    assert k in k_set
    assert tiles in tile_set
    k_idx = k_vals.index(k)
    tile_idx = tile_vals.index(tiles)
    assert result_matrix[tile_idx][k_idx] is None
    result_matrix[tile_idx][k_idx] = speedup

# Verify that all values in matrix are set
for k in range(len(k_vals)):
    for i in range(len(tile_set)):
        assert result_matrix[i][k] is not None

# Plot
fig, ax = plt.subplots(figsize=(20, 20))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
im = ax.imshow(result_matrix, cmap=cmap, vmin=-100, vmax=100)

ax.set_xticks([x for x in range(len(k_vals))], labels=k_vals)
ax.set_yticks([x for x in range(len(tile_set))], labels=sorted(list(tile_set)))

# Optional: Add text values to each box with value
#for i in range(len(tile_vals)):
#    for j in range(len(k_vals)):
#        text = ax.text(i, j, npvals[i, j],
#                ha="center", va="center", color="w")

ax.set_xlabel("K")
ax.set_ylabel("Output tiles")
ax.set_aspect(aspect=0.5)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Speedup (%)", rotation=-90, va="bottom")
plt.savefig(args.outfile, bbox_inches='tight')
