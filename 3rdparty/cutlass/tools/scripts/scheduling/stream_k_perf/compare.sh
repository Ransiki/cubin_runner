#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <first_wave_idx> <last_wave_idx> <outfile>"
  exit 1
fi

first_wave_idx=$1
last_wave_idx=$2
outfile=$3

if [ $first_wave_idx -gt $last_wave_idx ]; then
  echo "Last wave must be greater than or equal to first wave."
  echo "Passed in first_wave=${first_wave_idx} and last_wave=${last_wave_idx}"
  exit 1
fi

# Use the deviceQuery binary from the CTK's demo suite to get SM count
ctk_bin_dir=$(which nvcc | awk -F'nvcc' '{print $1}')
if [ ! -d $ctk_bin_dir ]; then
  echo "Directory containing NVCC not found"
  exit 1
fi
sms=$($ctk_bin_dir/../demo_suite/deviceQuery | grep "Multiprocessors" | awk -F'(' '{print $2}' | awk -F')' '{print $1}')

profiler_bin=./tools/profiler/cutlass_profiler

# Get stream-k and data-parallel kernels. There must be exactly two kernels available, one of which supports
# stream-k and the other of which does not.
tmpfile=/tmp/1
$profiler_bin --mode=enumerate | grep Operation: | awk -F' ' '{print $NF}' > $tmpfile
if [ $(wc -l $tmpfile | awk '{print $1}') -ne 2 ] || [ $(grep "stream_k" $tmpfile | wc -l) -ne 1 ] || [ $(grep -v "stream_k" $tmpfile | wc -l) -ne 1 ]; then
  echo "Incorrect library configuration"
  echo "There must be exactly two kernels available, one of which supports stream-k and the other of which does not."
  echo "Kernels obtained via ./tools/profiler/cutlass_profiler --mode=enumerate | grep Operation: | awk -F' ' '{print $NF}'"
  cat $tmpfile
  exit 1
fi
sk_kernel=$(grep    "stream_k" $tmpfile)
dp_kernel=$(grep -v "stream_k" $tmpfile)

find_value() {
  field=$1
 
  # Get the index of the field in the comma-delimited list
  index=$($profiler_bin --mode=enumerate | grep "Problem," | awk -v RS=',' -v str=${field} '{if ($0 == str) print NR}')

  # Get values at index in CSV
  $profiler_bin --mode=enumerate | grep "CUTLASS," | awk -v idx=${index} -F',' '{print $idx}' | uniq > $tmpfile

  if [ $(wc -l $tmpfile | awk '{print $1}') -ne 1 ]; then
    echo "Found more than one value for ${field}. Each kernel must use the same value for ${field}"
    exit 1
  fi

  cat $tmpfile
}

tile_m=$(find_value cta_m)
tile_n=$(find_value cta_n)
cluster_m=$(find_value cluster_m)
cluster_n=$(find_value cluster_n)
cluster_size=$(($cluster_m * $cluster_n))

# Find a square-ish problem size that matches the number of tiles and cluster requirements.
# Assumes that `tiles` is already a multiple of `cluster_m`
find_squarish() {
  tiles=$1
  python3 -c "\
x = [n for n in range(1,${tiles}+1) if ${tiles} % n == 0 and n % ${cluster_n} == 0];\
dists = [abs(a - (${tiles} // a)) for a in x];\
mindist = dists.index(min(dists));\
m = x[mindist];\
n = ${tiles} // m;\
print(m*${tile_m}, n*${tile_n})"
}

# Profiles the provided kernel and returns its GFLOPs/s
profile() {
  m=$1
  n=$2
  k=$3
  kernel=$4
  verify=$5
  $profiler_bin --m=$m --n=$n --k=$k --verification-enabled=$verify --kernels="${kernel}" | grep "   Math:" | awk -F' ' '{print $2}'
}

min_tiles=$(( $first_wave_idx * $sms + 1 ))
max_tiles=$(( ($last_wave_idx + 1) * $sms - 1 ))

verify=true
header="K,M,N,Tiles,Baseline-GFLOPs/s,New-GFLOPs/s,Speedup"
echo $header
echo $header >> $outfile
for k in 512 1024 2048 4096 8192; do
  for tiles in $(seq $min_tiles $max_tiles); do
    if [ $(($tiles % $cluster_size)) -ne 0 ]; then
      continue;
    fi

    mn=$(find_squarish $tiles)
    m=$(echo $mn | awk -F' ' '{print $1}')
    n=$(echo $mn | awk -F' ' '{print $2}')

    # Run data-parallel kernel
    gflops_dp=$(profile $m $n $k $dp_kernel $verify)

    # Run stream-K kernel
    gflops_sk=$(profile $m $n $k $sk_kernel $verify)

    speedup=$(python -c "print('{:.2f}'.format(100 * (${gflops_sk} - ${gflops_dp}) / ${gflops_dp}))")

    line="${k},${m},${n},${tiles},${gflops_dp},${gflops_sk},${speedup}"
    echo $line
    echo $line >> $outfile
  done
done
