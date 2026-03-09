/******************************************************************************
 * Copyright (C) 2023 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************/
/*! \file
    \brief Mods SM_ID, time stamps, and static scheduling verification on the host-side.
*/
//
// {$nv-release-never file}
//
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/util/host_tensor.h"
#include <algorithm>
#include "cutlass/arch/mods.h" // {$nv-release-never}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {


struct ModsVerify {
static bool static_scheduling_verify(cutlass::HostTensor<cutlass::mods::ModsInstrumentation::OutputTileInfo, cutlass::layout::PackedVectorLayout> mods_output, int MaxSmIdx) {
  std::vector<std::pair<int32_t, int32_t>> tile_smid(mods_output.size());
  for (uint32_t i = 0 ; i < mods_output.size(); i++) {
    cutlass::Coord<1> index(i);
    cutlass::mods::ModsInstrumentation::OutputTileInfo tmp = mods_output.host_view().at({index});

    // Verify SMID
    if (tmp.sm_id < 0 || tmp.sm_id > MaxSmIdx) {
      assert(0 && "Error: SM_ID is out of range! \n" );
      return false;
    }
    // Verify mainloop timing
    if (tmp.mainloop_start_ts < 0 || tmp.mainloop_end_ts < tmp.mainloop_start_ts) {
      assert(0 && "Error: incorrect time stamps!");
      return false;
    }
    tile_smid[i].first = i;
    tile_smid[i].second = tmp.sm_id;
  }

  // sort the <tile_id, sm_id> pairs ascending based on smid
  stable_sort(tile_smid.begin(), tile_smid.end(),
        [](const std::pair<int32_t, int32_t> &left, const std::pair<int32_t, int32_t>  &right) {return left.second < right.second;});

#if CUTLASS_DEBUG_TRACE_LEVEL
  printf("-----------------------------\n");
  for (uint32_t i = 0; i < tile_smid.size(); i++) {
    printf("tile_id %d, sm_id %d \n", tile_smid[i].first, tile_smid[i].second);
  } 
  printf("-----------------------------\n");
#endif

  // calculate tile steps on each SM
  std::vector<int32_t> tile_step(mods_output.size(), 0);
  for (uint32_t i = 1; i < tile_smid.size(); i++) {
    if (tile_smid[i-1].second == tile_smid[i].second) { // tiles processed by the same SM
      tile_step[i] = tile_smid[i].first - tile_smid[i-1].first;
    }
  }
  // verify that steps are the same for all the SMs
  int32_t curr_step, pre_step = 0;
  for (uint32_t i = 0; i < tile_step.size(); i++) {
    if (tile_step[i] != 0 ) {
      curr_step = tile_step[i];
      if (pre_step != 0 && curr_step != pre_step) {
        assert(0 && "Error: Not Static Scheduling ! \n");
        return false;
      } else {
        pre_step = curr_step;
      }
    }
  }
  CUTLASS_TRACE_HOST("  verified MODS output with " << mods_output.size() << " output tiles" << "\n");
  return true;
}

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
