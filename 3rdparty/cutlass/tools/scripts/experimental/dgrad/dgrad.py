#
#
import sys, re, os
from collections import OrderedDict
from enum import Enum
from problem_size import *

######################################################################################
#                                     Helper functions
######################################################################################
#
def InitialPQRS(problem, row):
  ''' computes the initial (n, p, q, r, s) coordinate '''
  n, h, w = problem.div_mod_nhw(row)

  if problem.mode == Mode.xCross:
    filter_r = ((h + problem.pad_h) % problem.stride_h)
    filter_s = ((w + problem.pad_w) % problem.stride_w)
    p = (h + problem.pad_h - filter_r)
    q = (w + problem.pad_w - filter_s)

  else:
    filter_r = ((h + problem.pad_h - problem.R + 1) % problem.stride_h)
    filter_s = ((w + problem.pad_w - problem.S + 1) % problem.stride_w)
    p = (h + problem.pad_h - problem.R + 1 + filter_r)
    q = (w + problem.pad_w - problem.S + 1 + filter_s)


  '''
  print("Initial(m: %d => (n: %d, h: %d, w: %d), Starting (p: %d, q: %d)," \
  "Starting filter position (r: %d, s: %d)"" % (row, n, h, w, p, q, filter_r, filter_s))
  '''

  return (n, p, q, filter_r, filter_s)

#
def PQtoHW(problem, p, q, filter_r, filter_s):
  h = p * problem.stride_h - problem.pad_h + filter_r
  w = q * problem.stride_w - problem.pad_w + filter_s
  return h, w

#
def LinearizeNHW(problem, n, h, w):
  return n * (problem.H * problem.W) + h * (problem.W) + w


#
def InitialHW4FilterRS(problem, filter_r, filter_s):
  r_ = filter_r
  s_ = filter_s
  if problem.mode == Mode.Conv:
    r_ = problem.R - 1 - r_
    s_ = problem.S - 1 - s_

  h_mapped = (problem.pad_h - r_) % problem.stride_h
  w_mapped = (problem.pad_w - s_) % problem.stride_w
  
  
  if h_mapped >= 0 and h_mapped < problem.H and\
  w_mapped >= 0 and w_mapped < problem.W:
    return (h_mapped, w_mapped)
  else:
    print("FirstHW returns negative starting h and w")
    sys.exit(0)


#
def AssertDgrad(problem, coord):
  n, p, q, r, s = coord
  return ((p % problem.stride_h) == 0) and ((q % problem.stride_w) == 0)

#
def coord_string(coord):
  return "(n: %d, p: %d, q: %d) * (r: %d, s: %d)" % (coord)

# 
def num_start_r_positions(problem):
  if problem.stride_h >= problem.R:
    return problem.R
  else:
    #return problem.R - problem.stride_h + 1
    return problem.stride_h

#
def num_start_s_positions(problem):
  if problem.stride_w >= problem.S:
    return problem.S
  else:
    #return problem.S - problem.stride_w + 1
    return problem.stride_w
######################################################################################

######################################################################################
#                  Simulating analytic mapping for NHW rows
######################################################################################
def MapDgradNHW(problem):
  # return map: key: (start_r, start_s), value: [mapped rows]
  nhw_map = OrderedDict()

  # number of starting filter positions
  num_starting_r_positions = num_start_r_positions(problem)
  num_starting_s_positions = num_start_s_positions(problem)

  print("num_starting_r_positions (%d), num_starting_s_positions (%d)" % \
    (num_starting_r_positions, num_starting_s_positions))

  # for all possible starting filter positions
  for filter_r in range(0, num_starting_r_positions):
    for filter_s in range(0, num_starting_s_positions):

      nhw_map[(filter_r, filter_s)] = []

      # starting row (h, w) position for filter position = (filter_r, filter_s)
      start_h, start_w = InitialHW4FilterRS(problem, filter_r, filter_s)

      print("\n** Mapping NHW rows [start (h: %d, w: %d)]"\
      " for starting filter (r, s) : (%d, %d) **" % (start_h, start_w, filter_r, filter_s))

      # map each valid row for starting filter position = (filter_r, filter_s)
      for row in range(0, problem.NHW):
        n, p, q = problem.npq(row, start_h, start_w)
        #n, p, q = problem.npq_fix_1(row, filter_r, filter_s)
        #n, p, q = problem.npq_fix_2(row, start_h, start_w)
        #n, p, q = problem.npq_fix_3(row, start_h, start_w, filter_r, filter_s)


        r_ = filter_r
        s_ = filter_s

        if problem.mode == Mode.Conv:
          r_ = problem.R - 1 - r_
          s_ = problem.S - 1 - s_

        # mapped Dx row (h, w) for the starting filter position
        # note that mapped_(h, w) are valid rows and divisible by stride_h, and stride_w
        mapped_h = start_h + p * problem.stride_h
        mapped_w = start_w + q * problem.stride_w

        # mapped Dy (p, q) to be accessed for the starting filter position and row (n, h_mapped, w_mapped)
        mapped_p = (mapped_h + problem.pad_h - r_) // problem.stride_h
        mapped_q = (mapped_w + problem.pad_w - s_) // problem.stride_w

        # linearized nhw row index this thread is working on
        mapped_row = LinearizeNHW(problem, n, mapped_h, mapped_w)
        
        # for a CTA larger than the number of valid rows for a filter position will have 
        # threads in a CTA working on mapped_rows which are OOB for a filter position = (filter_r, filter_s)
        pred = mapped_row < problem.NHW and \
        mapped_h < problem.H and mapped_w < problem.W and \
        mapped_p >= 0 and mapped_q >= 0 and \
        mapped_p < problem.P and mapped_q < problem.Q
        valid = "true" if pred else "false"

        # cache and return mapped rows for verification
        if mapped_row < problem.NHW:
          nhw_map[(filter_r, filter_s)].append(mapped_row)

        print("row: %d => mapped_row: %d - (n, h, w): (%d, %d, %d)"\
          " - gemm_k=0 access (p, q): (%d, %d) * (r: %d, s: %d)"\
          " - valid [%s]"\
          % (row, mapped_row, \
            n, mapped_h, mapped_w, \
            mapped_p, mapped_q, \
            filter_r, filter_s, \
            valid))

  return nhw_map

######################################################################################
def VerifyMapping(problem, nhw_map):
  mapped_rows = []
  for start_filter in nhw_map:
    mapped_rows += nhw_map[start_filter]
  mapped_rows.sort()

  '''
  # this check is not true for all cases. For 1x1 filter 2x2 stride some nhw will not get
  # mapped to any thread
  # check mapping size
  if len(mapped_rows) != problem.NHW:
    print("NHW rows in mapping %s" % mapped_rows)
    print("Incomplete mapping (total rows:%d , mapped rows: %d)\n" % (problem.NHW, len(mapped_rows)))
    return False
  '''

  # check if mapping is one-to-one
  prev = -1
  i = 0
  while i < len(mapped_rows):
    if prev == mapped_rows[i]:
      print("NHW rows in mapping %s" % mapped_rows)
      print("Not a one-to-one mapping. Repeated row: (%d)\n" % (prev))
      return False
    
    prev = mapped_rows[i]
    i = i + 1

  #print(mapped_rows)
  # valid mapping
  return True

######################################################################################
##                          Reference check all access
######################################################################################
def ReferenceCheck(problem):
  print("\n** Reference check **")
  # check access with reference
  for row in range(0, problem.NHW):

    initial_coord = InitialPQRS(problem, row)

    print("row: %d Initial Mma: %s" % (row, coord_string(initial_coord)))
    accesses = []
    total_accesses = []
    # simulate GEMM K iteration
    n, p, q, r, s = initial_coord

    while r < problem.R:
      q = initial_coord[2]
      s = initial_coord[4]
      while s < problem.S:
        
        access_coord = (n,p//problem.stride_h, q//problem.stride_w,r,s)
        print("      Access "+coord_string(access_coord))
        total_accesses.append(access_coord)

        if (p >= 0) and (p // problem.stride_h) < problem.P and \
           (q >= 0) and (q // problem.stride_w) < problem.Q:
          # valid access into dy n,p,q coordinate
          accesses.append(access_coord)
          #print("Valid access "+coord_string(coord))

        next_q = q - problem.stride_w * problem.sign
        next_s = s + problem.stride_w
        
        q = next_q
        s = next_s

      next_p = p - problem.stride_h * problem.sign
      next_r = r + problem.stride_h
      p = next_p
      r = next_r


    # verify these accesses result in the proper sum
    reference_accesses = []
    n, h, w = problem.div_mod_nhw(row)
    for r in range(problem.R):
      for s in range(problem.S):
        r_ = r
        s_ = s
        if problem.mode == Mode.Conv:
          r_ = problem.R - 1 - r_
          s_ = problem.S - 1 - s_

        p = h + problem.pad_h - r_
        q = w + problem.pad_w - s_

        if (p >= 0) and (p % problem.stride_h) == 0 and \
           (q >= 0) and (q % problem.stride_w) == 0:

           p = p // problem.stride_h
           q = q // problem.stride_w

           if p < problem.P and q < problem.Q:
            coord = (n, p, q, r, s)
            reference_accesses.append(coord)

    # did we hit every access?
    if len(reference_accesses) == len(accesses):
      for ref_coord, coord in zip(reference_accesses, accesses):
        if ref_coord != coord:
          print("ERROR: access miss-match")
          sys.exit(0)
      print("PASS m: %d - n,h,w (%d, %d, %d) - (num accesses: total: %d, ref: %d, valid: %d)" %\
        (row, n, h, w, len(total_accesses), len(reference_accesses), len(accesses)))
      for access in total_accesses:
        valid = "true" if access in accesses else "false" 
        print("Mma: %s [%s]" % (coord_string(access), valid))
    else:
      print("ERROR: reference: %d, accesses: %d" %(len(reference_accesses), len(accesses)))
      print("reference_accesses: " + str(reference_accesses))
      print("accesses          : " + str(accesses))
      sys.exit(0)
######################################################################################

#
def Main():

  problems = []

  # test hand-written strided layers
  for layer in StridedLayers.layers:
    problems.append(ProblemSize(layer))


  # test strided layers from standard network
  # add all strided Resnet50 layers to problem set
  #for layer in Resnet50.layers:
  #  if layer.is_strided():
  #    problems.append(ProblemSize(layer))

  # process problem in problem set
  for problem in problems:
    print("\n\n** Map and verify problem **\n%s" % problem)

    nhw_map = MapDgradNHW(problem)

    if not VerifyMapping(problem, nhw_map):
      print("Failed NHW mapping for problem size:\n%s" % problem)
      sys.exit(0)

    ReferenceCheck(problem)


#
if __name__ == '__main__':
  sys.exit(Main())
