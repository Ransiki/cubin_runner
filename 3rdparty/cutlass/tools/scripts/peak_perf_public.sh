# Gemm public performance runs

python3 ../tools/scripts/peak_perf_gemm_public.py --chip="2080Ti" --phases=profile --clock=1335 --sm-count=68 --providers=cutlass
python3 ../tools/scripts/peak_perf_gemm_public.py --chip="2080Ti" --phases=profile --clock=1335 --sm-count=68 --providers=cublas --profiler-path=../cutlass-2.x-read-only/build/tools/profiler/cutlass_profiler
python3 ../tools/scripts/peak_perf_gemm_public.py --chip="2080Ti" --phases=process --clock=1335 --sm-count=68 --providers=cutlass,cublas --output=cutlass_2.0_peak_perf_gemm_public.csv

#python3 ../tools/scripts/peak_perf_public.py --chip="TitanV" --phases=profile --clock=1380 --sm-count=80 --providers=cutlass
#python3 ../tools/scripts/peak_perf_public.py --chip="TitanV" --phases=profile --clock=1380 --sm-count=80 --providers=cublas --profiler-path=../cutlass-2.x-read-only/build/tools/profiler/cutlass_profiler
#python3 ../tools/scripts/peak_perf_public.py --chip="TitanV" --phases=process --clock=1380 --sm-count=80 --providers=cutlass,cublas --output=cutlass_2.0_peak_perf_gemm_public.csv --append=true

# Conv public performance runs
