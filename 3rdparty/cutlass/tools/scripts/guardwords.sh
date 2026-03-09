#!/bin/bash -v
#
#  {$nv-release-never file}
#
# detect leaks for internal release (cuDNN/Fort)

# Exclude EULA.txt, the license used for CUTLASS DSLs
export GREP="grep --ignore-case -r -n --exclude-dir=docs --exclude=*.png --exclude=EULA.txt"
export CASE_SENSITIVE_GREP="grep -r -n --exclude-dir=docs --exclude=*.png --exclude=EULA.txt"

# save all the file names to a temporary file, then treat it as a regular file to parse line by line
touch filename_tmp.txt
echo "#  {$nv-release-never file} " >> filename_tmp.txt
find . -not -name "*.png" >> filename_tmp.txt

# Red alerts - These MUST NOT BE EXPOSED.
${GREP} e8m7 *
${GREP} e8m10 *
${GREP} e6m9 *
${GREP} e0m3 *
${GREP} sm82 *
${GREP} ghmma *
${GREP} supermma *
${GREP} "super mma" *
${GREP} 2xacc *
${GREP} DoubleAccum *

${GREP} xmma *

# General guards relevant to dkg
${GREP} dkg * | grep -v tDKgDK
${GREP} "cutlass[_|-|\s]ir" * | grep -v cutlass_irrelevant | grep -v CUTLASS_irrelevant
${GREP} "collective[_|-|\s]ir" *
${GREP} "nv[_|-|\s]tile" *
${GREP} "cuda[_|-|\s]tile" *
${GREP} "tile[_|-|\s]ir" *
${GREP} "tileaa" *
${GREP} "tileas" *
${GREP} cutegen *
${GREP} "NV_CONTRIB" *

# Guards for sensitive internal environment variables used by CUTLASS Python
# i.e. those that we don't want to advertise explicitly to users
${GREP} "_print_ir_after_all"
${GREP} "_keepir_after_passes"

${GREP} gitlab *
${GREP} jira *
${GREP} perfsim *
${GREP} jenkins *
${GREP} nvbugs *
${GREP} cfk *
${GREP} ocg *
${GREP} mods * | grep -v "div/mod" | grep -v "divmod"

# Host names and people
${GREP} computelab *
${GREP} powerstroke *
${GREP} akerr *
${GREP} dumerrill *
${GREP} haichengw *
${GREP} manigupta *
${GREP} nfarooqui *
${GREP} ashivam *
${GREP} mfh *
${GREP} computearch *
${GREP} compute_arch *
${GREP} perforce *
${GREP} p4 * | grep -v dp4a | grep -v DP4A | grep -v FP4 | grep -v fp4 | grep -v Fp4 | grep -v MSP430 | grep -v msp430 | grep -v Cp4x
${GREP} amodel *
${GREP} splinter *
${GREP} dvs *

# Guards itself (in case mistyped)
${GREP} nv- * | grep -v nv-nsight-cu-cli

${CASE_SENSITIVE_GREP} "\<_mma\>" *
${GREP} dmma * | grep -v "ThreadMma" | grep -v "TiledMMA" | grep -v "TiledMma" | grep -v -i "blockscaledMma" | grep -v "SpecializedMma" | grep -v "LoadMma"
${GREP} __nvvm_ldsm *
grep --ignore-case -r -n __nvvm | grep -v __nvvm_get_smem_pointer

${GREP} nv_p2r *
${GREP} nv_r2p *
${GREP} p2r *
${GREP} r2p *
${GREP} regoffset *
${GREP} reg_offset * | grep -v "int reg_offset = " | grep -v "int reg_idx = "
${GREP} genmetadata *

# Avoid disclosing mention of internal tools
${GREP} cask *
${GREP} jetfire *
${GREP} fenceinterference *
${GREP} warp_switch *
${GREP} extended_ptx *
${GREP} internal_nvvm *
${GREP} nvopt *
${GREP} knob *
${CASE_SENSITIVE_GREP} "\.pragma" *
${CASE_SENSITIVE_GREP} evo * | grep -v "Trevor"
${CASE_SENSITIVE_GREP} INTERNAL_L2_PREFETCH *

# We shouldn't really be naming things after SASS instructions.
${CASE_SENSITIVE_GREP} HMMA *
${CASE_SENSITIVE_GREP} IMMA *
${CASE_SENSITIVE_GREP} DMMA *
${CASE_SENSITIVE_GREP} QMMA *
${CASE_SENSITIVE_GREP} OMMA * | grep -v "COMMAND"
${CASE_SENSITIVE_GREP} UIMMA *
${CASE_SENSITIVE_GREP} UHMMA *
${CASE_SENSITIVE_GREP} UQMMA *
${CASE_SENSITIVE_GREP} UOMMA *
${GREP} MMA\\.1 * # used for ampere MMA.16<> and blackwell MMA.128<>
${GREP} MMA\\.8 * # used for volta and ampere MMA.8<>
${GREP} MMA\\.6 * # used for hopper MMA.64<>
${GREP} LDG * # will also cover LDGSTS
# STSM can be removed if we ever decide to remove all its mentions from CUTE.
${CASE_SENSITIVE_GREP} STS * | grep -v "STSM" | grep -v "EXISTS" | grep -v "LISTS" | grep -v "TESTS" # case sensistive bcoz otherwise all 'tests' will show up too



# Avoid leaking Grafia features
${GREP} grafia *

# Avoid leaking Hopper SASS instructions
${GREP} HGMMA *
${GREP} IGMMA *
${GREP} QGMMA *
${GREP} BGMMA *
#${GREP} STSM *
${GREP} group_mma * | grep -v gpu.subgroup_ | grep -v subgroup_mma | grep -v warpgroup_mma
${GREP} tmaldg *
${GREP} tmastg *
${GREP} UBLKCP *
${GREP} cga * | grep -v "SymmetricGaussian" | grep -v "cACgAC" | grep -v "tCgA" | grep -v "cgA_mk"
${GREP} STAS *
${GREP} REDAS *
${GREP} USET * | grep -v "usetikzlibrary" | grep -v "ReuseTmem"
${GREP} ACQBULK *
${GREP} USETSHMSZ *
${GREP} FDL *

# Avoid leaking Blackwell/Rubin SASS and features
${GREP} carrot *
${GREP} rubin *
${GREP} sm102
${GREP} sm104 * # covers sm104
${GREP} sm107 * # covers sm107
${GREP} gb10 * | grep -v FK_Compiler_perf_testlist_GB100_SM100_cutlass3x_gemm_public # covers gb100/102/10b
${GREP} gb11 *  # covers gb110
${GREP} gb12 *  # covers gb120
${GREP} gb20 * | grep -v "GB200" # covers gb20x
${GREP} gr100 * # covers gr100
${GREP} gr102 * # covers gr102
${GREP} utcmma *
${GREP} LDTM * | grep -v FieldTma | grep -v fieldtma
${GREP} STTM * | grep -v CastTMA
${GREP} workid * | grep -v workidx
${GREP} work_id * | grep -v work_idx
${GREP} "work id" * | grep -v "work idx"
${GREP} "work-id" * | grep -v "work-idx"
${GREP} "flexible_cluster" *
${GREP} "flexible cluster" *
${GREP} "flexiblecluster" *
${GREP} cluster_mma *
${GREP} convmma * | grep -v "DepthwiseDirectConvMma"
${GREP} UTCHMMA *
${GREP} UTCIMMA *
${GREP} UTCBMMA *
${GREP} UTCQMMA *
${GREP} UTCMXQMMA *
${GREP} UTCOMMA *
${GREP} UTCBAR *
${GREP} UTC.MMA *
${GREP} nq_2d_tiled *
${GREP} Nq2dTiled
${GREP} activation_stationary *
${GREP} ActivationStationary *
${GREP} tma_load_w * # Also covers tma_load_w128
${GREP} TmaLoadPrefetch *
${GREP} tma_loadpf *
${GREP} tmaload_prefetch *

# CAGE (Cache-Aware GEMM) related terminlogy
${GREP} cage *
${GREP} cacheaware *
${GREP} cache-aware *
${GREP} "cache aware" *
${GREP} "alternating k" *
${GREP} "continuous raster" *

# Avoid leaking Blackwell new types
${GREP} e2m5 *
# e3m4 was upstreamed
${GREP} e3m4fn *
${GREP} s2m6 *

# Avoid leaking Feynman
${GREP} feynman *
${GREP} DLCMMA *
${GREP} DLCBAR *
${GREP} DLCCP *
${GREP} DLCFLUSH *
${GREP} DPCTMA *
${GREP} DPCBLKCP *
${GREP} DPCRM *
${GREP} LDSPM *
${GREP} STASPM *
${GREP} TMALDSP *
${GREP} TMASTSP *
${GREP} SM140 *
${GREP} FN100 *
${GREP} tcgen06 *

# Avoid leaking comments with bad words
${GREP} hack *
${GREP} ugly *
${GREP} stupid *
${GREP} dumb *
${GREP} suck *
${GREP} shit *
${CASE_SENSITIVE_GREP} XXX * | grep -v "_XXX_"
${GREP} broken *
${GREP} fixme *

# Avoid TODOs (that are not properly mentioned) and use of WARs in code
${GREP} todo * | grep -v "TODO" | grep -v "autodoc"
${CASE_SENSITIVE_GREP} WAR * | grep -v "WARP" | grep -v "WARRANTIES" | grep -v "SOFTWARE" | grep -v "WARN" | grep -v "TOWARD" | grep -v "WARM"

# Avoid leaking internal website
${GREP} confluence.nvidia.com *
${GREP} jirasw.nvidia.com *
${GREP} gitlab-master.nvidia.com *
${GREP} nvbugspro.nvidia.com *

#remove the temporary filename file
rm filename_tmp.txt
