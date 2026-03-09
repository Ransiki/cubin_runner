#
# This is a short script to design the CUTLASS HMMA.884 GEMM
#

def get_lane_indices(tid):
  lane_in_quad = (tid & 3)
  quad = (tid >> 2)
  quad_pair = ((tid & 0xf) >> 3)

  return (quad_pair, quad, lane_in_quad)


#
def hmma_nt_A_coordinate(tid, WarpDelta):
  ''' WarpDelta is the difference in each dimension (row, column) between warp tiles'''

  quad_pair, quad, lane_in_quad = get_lane_indices(tid)

  # row_idx = q0_q2
  row_idx = (((quad >> 2) & 1) | ((quad & 1) << 1))
  col = lane_in_quad

  row_idx *= WarpDelta[0]

  return row_idx, col

#
def hmma_nt_B_coordinate(tid, WarpDelta):
  ''' WarpDelta is the difference in each dimension (row, column) between warp tiles'''

  quad_pair, quad, lane_in_quad = get_lane_indices(tid)

  # col_idx = q1_q2
  col_idx = ((quad & 2) | ((quad >> 2) & 1))
  row = lane_in_quad

  col_idx *= WarpDelta[1]

  return col_idx, row

#
# sanity check
#
if False:
  print("A:")
  for tid in range(32):
    contig_idx, strided = hmma_nt_A_coordinate(tid, [2, 2])
    print("T%d - row: %d, col: %d" %( tid, 8 * contig_idx, strided))


  print("B:")
  for tid in range(32):
    contig_idx, strided = hmma_nt_B_coordinate(tid, [2, 2])
    print("T%d - row: %d, col: %d" %( tid, strided, 8 * contig_idx))

#
# Swizzling
#

def is_conflict_free(addr, is_store = True, size=128):

  beat_count = (size >> 5) * (len(addr) >> 5)
  bank_count = int(32 / beat_count)

  for beat in range(beat_count):

    # thread ID accessing each bank
    banks = [-1 for x in range(bank_count)]

    # address being accessed
    accesses = [-1 for x in range(bank_count)]

    for tid in range(bank_count):
      address = addr[tid + beat * bank_count]
      bank = (address % bank_count)
      if banks[bank] >= 0 and banks[bank] != tid and (is_store or accesses[bank] != address):
        # conflict
        print("Bank conflict! thread %d accessing bank %d (address %d) concurrently accessed by thread %d who accessed address %d" % (tid, bank, address, banks[bank], accesses[bank]))
        return False
      else:
        banks[bank] = tid
        accesses[bank] = address
  return True

#
# Sanity check
#
if False:
  free = is_conflict_free([x for x in range(32)], 128)
  print("addr[tid]=tid %s conflict free" % ("is" if free else "is NOT"))

#
# Swizzling operator
#

def sts_swizzle(tid, addr):
  return addr

def lds_swizzle(tid, addr):
  return addr

#
# Conflict free on store to shared?
#

#
def LDGSTS_conflict_free():
  ''' 8x4x128b loaded from consecutive memory needs to be conflict free on storing'''

  # model a warp-wide LDG128
  ldg128_addr = range(32)

  # compute store address
  addr = [sts_swizzle(tid, ldg128_addr[tid]) for tid in range(32)]

  return is_conflict_free(addr)

#
if False:
  free = LDGSTS_conflict_free()
  print("LDG=>STS %s conflict free" % ("is" if free else "is NOT"))

#
# Conflict free on load from shared?
#

def hmma_nt_A_column_sts_offset(tid):

  _addr = (tid & 4) | ((tid & 1) << 1) | ((tid >> 1) & 1)

  t2t0 = ((tid >> 1) & 2) | (tid & 1)
  t4t3 = ((tid >> 3) & 3)
  return (_addr << 2) | (t2t0 ^ t4t3)

#
def LDS128_conflict_free():

  # logical address
  logical_addr = [0 for x in range(32)]
  addr = [0 for x in range(32)]
  for tid in range(32):
    addr[tid] = hmma_nt_A_column_sts_offset(tid)


  print("A.column - loading from shared memory")
  for tid, address in enumerate(addr):
    print("T%d - LDS.128 [%d]" % (tid, address))

  free = is_conflict_free(addr)

  return free

#
if False:
  free = LDS128_conflict_free()
  print("A.column - LDS => RF %s conflict free" % ("is" if free else "is NOT"))

#
# Generalizing
#

#
def ldg128_hmma_A_column(tid, lda, ldg_idx):
  ''' Computes the vector loaded by the CTA '''
  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  return ldg_idx * 8 + (lane_id & 0x07) + lda * (lane_id >> 3)

#
def sts128_hmma_A_column(tid, WarpCount, ldg_idx):
  ''' Computes addresses to write to SMEM'''

  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  column = (ldg_idx << 3) | (lane_id & 0x7)

  WarpOffsetIdx = (column % WarpCount)
  qp_column = int(column / WarpCount)

  qp_row = ((qp_column >> 1) & 2) | (qp_column & 1)
  qp_col = ((qp_column >> 1) & 1)
  
  t4t3 = ((tid >> 3) & 3)
  if WarpCount == 1:
    col_rotate = ((tid >> 1) & 2) | (tid & 1) # t[2,0] for warpcount=1
  else:
    col_rotate = (tid & 3) # t[1, 0] for warpcount={2,4} 
  
  t_col = (qp_col << 2) | (col_rotate ^ t4t3)

  WarpOffsetDelta = 4
  offset = 8 * (WarpOffsetIdx * WarpOffsetDelta + qp_row) + t_col
  return offset

#
def lds128_hmma_A_column(tid, WarpCount, tile_idx = 0):

  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)
  quad_id = (lane_id >> 2)

  offset = warp_id * 32 + tile_idx * 16

  evenodd_quad_id = (quad_id & 1)
  halfwarp_hilo = (lane_id >> 4)

  smem_col_idx = (evenodd_quad_id << 2)

  if WarpCount == 1:
    # WarpCount == 1 case requires two SMEM pointers
    if tile_idx == 0:
      # lo 32x32x4 tile
      smem_col_idx = smem_col_idx | ((lane_id & 3) ^ (halfwarp_hilo))
    else:
      # hi 32x32x4 tile
      smem_col_idx = smem_col_idx | ((lane_id & 3) ^ (2 | halfwarp_hilo))
  elif WarpCount == 2:
    t4_0 = ((lane_id >> 3) & 2)
    smem_col_idx = smem_col_idx | ((lane_id & 3) ^ t4_0 ^ warp_id)
  elif WarpCount == 4:
    smem_col_idx = smem_col_idx | ((lane_id & 3) ^ warp_id)

  smem_row_idx = halfwarp_hilo
  offset += smem_col_idx + smem_row_idx * 8

  return offset

###################################################################################################
#
# B.row-major
#

#
def ldg128_hmma_B_row(tid, ldb, ldg_idx):
  ''' Computes the vector loaded by the CTA '''
  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  return ldg_idx * 8 + (lane_id & 0x07) + ldb * (lane_id >> 3)

#
def sts128_hmma_B_row(tid, WarpCount, ldg_idx):
  ''' Computes addresses to write to SMEM'''

  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  column = (ldg_idx << 3) | (lane_id & 0x7)

  hmma_op = (column % WarpCount)

  hmma_row = hmma_op * 4 + (int(column / WarpCount) & 3)
  hmma_col = (int(column / WarpCount) >> 2)

  t4t3 = ((tid >> 3) & 3)
  col_rotate = (tid & 3)

  offset = (hmma_row << 3) | (hmma_col << 2) | (col_rotate ^ t4t3)

  return offset

#
def lds128_hmma_B_row(tid, WarpCount, tile_idx = 0):

  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)
  quad_id = (lane_id >> 2)

  quads = (quad_id >> 1)

  if WarpCount == 1:
    t3t4 = ((tid >> 2) & 2) | ((tid >> 4) & 1)
    col_rotate = (tid & 3) ^ t3t4
  elif WarpCount == 2:
    t4_0 = ((tid >> 3) & 2)
    col_rotate = (tid & 3) ^ t4_0 ^ warp_id
  elif WarpCount == 4:
    t4_0 = ((tid >> 3) & 2)
    col_rotate = (tid & 3) ^ warp_id

  smem_row_idx = warp_id * 4 + ((quads & 1) << 1) | ((quads >> 1) & 1)
  smem_col_idx = tile_idx * 4 | col_rotate
  offset = smem_row_idx * 8 + smem_col_idx

  return offset

###################################################################################################

#
# Verification
#

class Vector:
  def __init__(self, operand, outer_start = -1, outer_end = -1, inner = -1):
    self.operand = operand
    self.start = outer_start
    self.end = outer_end
    self.inner = inner

  def __str__(self):
    rng = "%d..%d" %(self.start, self.end-1)
    if self.inner < 0:
      return "          "
    if self.operand == 'A':
      x = "A[% 4s, %d]" % (rng, self.inner)
    elif self.operand == 'B':
      x = "B[%d, % 4s]" % (self.inner, rng)
    else:
      x = "Undef"
    return x

#
# Test program
#

#
def test_congruous_loading(operand, ldg128_operation, sts128_operation, lds128_operation, CtaTileOuter, warp_count = 1):
  ''' Prints global memory indices, shared memory indices, and register file indices'''
  gmem = []

  CtaTileK = 4

  SmemStride = 8
  GmemStride = CtaTileOuter

  for k_idx in range(CtaTileK):
    for outer_idx in range(CtaTileOuter):
      gmem.append(Vector(operand, outer_idx * SmemStride, (outer_idx + 1) * SmemStride, k_idx))

  print("\nWarp shape: %d" % warp_count)
  print("GMEM:")
  for ldg_idx in range(CtaTileOuter >> 3):
    print("\nLDG.128 [%d] (Warp 0)" % ldg_idx)
    for warp_row in range(4):
      ldg_data = []
      for lane in range(8):
        tid = warp_row * 8 + lane
        ldg_offset = ldg128_operation(tid, GmemStride, ldg_idx)
        ldg_data.append(str(gmem[ldg_offset]))
      print(";  ".join(ldg_data))


  smem = [Vector('Undefined') for x in range(len(gmem))]
  sts128_addr = []
  for ldg_idx in range(CtaTileOuter >> 3):
    for tid in range(32):
      ldg_offset = ldg128_operation(tid, GmemStride, ldg_idx)
      sts_offset = sts128_operation(tid, warp_count, ldg_idx)
      smem[sts_offset] = gmem[ldg_offset]
      sts128_addr.append(sts_offset)

  print("\nSMEM:")
  for row in range(len(smem) >> 3):
    print(";  ".join([str(smem[row * 8 + x]) for x in range(8)]))

  if is_conflict_free(sts128_addr):
    print("(conflict free)")
  else:
    print("NOT CONFLICT FREE")

  for lds128 in range(2):
    lds128_addr = []
    RF = []
    for tid in range(warp_count * 32):
      offset = lds128_operation(tid, warp_count, lds128)
      lds128_addr.append(offset)
      RF.append(smem[offset])

    print("\nLDS.128[%d]" % lds128)
    for warp_idx in range(warp_count):
      print("\nWarp %d" % warp_idx)
      for quadpair in range(4):
        print(";  ".join([str(RF[warp_idx * 32 + quadpair * 8 + lane]) for lane in range(8)]))

    if is_conflict_free(lds128_addr, False):
      print("(conflict free)")
    else:
      print("NOT CONFLICT FREE")

  print("\n")

if True:
  #for log_warp_count in range(3):
  if True:
    log_warp_count = 0
    outer_dim = (8 << log_warp_count)
    print("=====================================================================================================")
    test_congruous_loading('A', ldg128_hmma_A_column, sts128_hmma_A_column, lds128_hmma_A_column, outer_dim, 1 << log_warp_count)
    print("-----------------------------------------------------------------------------------------------------")
    test_congruous_loading('B', ldg128_hmma_B_row, sts128_hmma_B_row, lds128_hmma_B_row, outer_dim, 1 << log_warp_count)

###################################################################################################
#
# Crosswise loading
#
###################################################################################################

#
class Matrix:
  def __init__(self, tid = -1, operand = '', row_start = -1, rows = -1, col_start = -1, cols = -1):
    self.tid = tid
    self.operand = operand
    self.row_start = row_start
    self.rows = rows
    self.col_start = col_start
    self.cols = cols

  def str_w_tid(self, _tid):
    if self.tid < 0 or self.rows < 0 or self.cols < 0:
      x =  "                  "
    else:
      row_rng = "%d..%d" %(self.row_start, self.row_start + self.rows-1) if self.rows > 1 else str(self.row_start)
      col_rng = "%d..%d" %(self.col_start, self.col_start + self.cols-1) if self.cols > 1 else str(self.col_start)
      x = "T%d: %s[%s, %s]" % (_tid, self.operand, row_rng, col_rng)
    return x

  def __str__(self):
    if self.tid < 0 or self.rows < 0 or self.cols < 0:
      x =  "                  "
    else:
      row_rng = "%d..%d" %(self.row_start, self.row_start + self.rows-1) if self.rows > 1 else str(self.row_start)
      col_rng = "%d..%d" %(self.col_start, self.col_start + self.cols-1) if self.cols > 1 else str(self.col_start)
      x = "T%d: %s[%s, %s]" % (self.tid, self.operand, row_rng, col_rng)
    return x


#
def ldg128_hmma_A_row(tid, ldm, ldg128_idx = 0):
  ''' Computes the vector loaded by the CTA '''
  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  return (ldg128_idx * 8 + (lane_id >> 2)) * ldm + (lane_id & 3)

#
def sts64_hmma_A_row(tid, WarpCount, ldg128_idx = 0, sts64_idx = 0):
  ''' Computes the store address (in units of 64b offsets)'''

  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  k_idx = ((lane_id & 0x03) << 1) | sts64_idx
  smem_row = (ldg128_idx >> 1)

  # Two store pointers are needed
  if (ldg128_idx in [0, 3]):
    smem_col = (lane_id & 0x0f) ^ ((lane_id >> 4) & 1)
  else:
    smem_col = (lane_id & 0x0f) ^ 2 ^ ((lane_id >> 4) & 1)

  kstart = k_idx * 4
  kend = (k_idx + 1) * 4 - 1

  print("T%d - LDG.128[%d], STS.64[%d] => [k=%d..%d, bank: %d]" % (tid, ldg128_idx, sts64_idx, kstart, kend, smem_col))

  kidx_stride = 32
  offset = k_idx * kidx_stride + smem_row * 16 + smem_col
  return offset

#
def lds128_hmma_A_row(tid, WarpCount, k_idx = 0, lds128_idx = 0):
  ''' Computes an address (in units of 128b offsets) loaded by each thread for each k_index'''
  
  warp_id = (tid >> 5)
  lane_id = (tid & 0x1f)

  quad_idx = (lane_id >> 2)
  quadpair_idx = ((quad_idx & 1) << 1) | ((quad_idx >> 2) & 1)

  smem_row = ((quadpair_idx >> 1) & 1)
  smem_col = (quadpair_idx & 1) ^ smem_row

  offset = k_idx * 16 + smem_row * 8 + ((lane_id & 3) << 1) + smem_col 

  return offset

#
def PrintSMEM(SMEM, WarpCount):
  for k_idx in range(8):
    for row in range(2):
      print("; ".join([str(SMEM[k_idx * 32 + row * 16 + col]) for col in range(16)]))

#
def SimulateArow():
  ''' Simple simulator for loading a tile of data and storing to SMEM '''
  WarpCount = 1

  SMEM = [Vector('A', -1) for x in range(8 * 32)]

  for ldg128_idx in range(4):
    LDG_RF = [Matrix(-1, 'A', -1) for x in range(64)]
    for tid in range(32):
      offset = ldg128_hmma_A_row(tid, 4, ldg128_idx) * 8
      k_idx = offset % 32
      outer_idx = (offset >> 5)
      LDG_RF[tid*2] = Matrix(tid, 'A', outer_idx, 1, k_idx, 4)
      LDG_RF[tid*2+1] = Matrix(tid, 'A', outer_idx, 1, k_idx + 4, 4)

    for sts64_idx in range(2):
      addr = [-1 for x in range(32)]
      for tid in range(32 * WarpCount):
        offset = sts64_hmma_A_row(tid, WarpCount, ldg128_idx, sts64_idx)
        SMEM[offset] = LDG_RF[tid * 2 + sts64_idx]
        addr[tid] = offset
      if is_conflict_free(addr, True, 64):
        print("(conflict free)")
      else:
        print("STS.64 BANK CONFLICTS")

  PrintSMEM(SMEM, WarpCount)

  for k_idx in range(1):
    LDS_RF = [Matrix(-1, 'A', -1) for x in range(64)]
    addr = [-1 for x in range(32)]
    for tid in range(32):
      offset = lds128_hmma_A_row(tid, WarpCount, k_idx)
      addr[tid] = offset
      LDS_RF[tid*2] = SMEM[offset * 2]
      LDS_RF[tid*2+1] = SMEM[offset * 2 + 1]

    if is_conflict_free(addr, False):
      print("(conflict free)")
    else:
      print("LDS.128 BANK CONFLICTS")

    for quad in range(8):
      print("; ".join([LDS_RF[quad * 8 + x].str_w_tid((quad << 2) | (x >> 1)) for x in range(8)]))


#SimulateArow()


if True:
  pass
