/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// {$nv-internal-release file}
/*! \file
    \brief Matrix multiply
*/

#pragma once

#include <cassert>
#include "wmma_ptx.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// This file contains WMMA recipe implentations using CUDA C++ provided to the PTXAS team. 
// Once the wmma SM80 implentations are avialable in PTXAS they should be verified by
// enabling them in wmma_ptx.h file. Set/Unset below flag in the file `cutlass/arch/wmma.h` 
// CUTLASS_USE_SM80_WMMA_CUDA_EMULATION=1 (CUTLASS provided CUDA emulation for WMMA SM80)
// CUTLASS_USE_SM80_WMMA_CUDA_EMULATION=0 (verifies ptxas wmma instructions for SM80)
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {
namespace emu {

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////// structures to enclose CUTLASS implentations wmma instructions ///////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA load for A, B, and C matrices 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                           ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                         ///< Data type of elements 
  typename Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared    ///< Data resides in shared or global memory
>
struct WmmaLoadA;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <  
  typename Shape_,                           ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                         ///< Data type of elements 
  typename Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared    ///< Data resides in shared or global memory
>
struct WmmaLoadB;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <  
  typename Shape_,                           ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                         ///< Data type of elements 
  typename Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared    ///< Data resides in shared or global memory
>
struct WmmaLoadC;
/////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA.MMA with Wmma::Shape targeting an MMA instruction with Wmma::TargetInstructionShape
//////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                                   ///< Size of the wmma matrix product (concept: GemmShape)
  typename ElementA_,                                ///< Data type of A elements 
  typename LayoutA_,                                 ///< Layout of A matrix (concept: MatrixLayout)  
  typename ElementB_,                                ///< Data type of B elements
  typename LayoutB_,                                 ///< Layout of B matrix (concept: MatrixLayout)  
  typename ElementC_,                                ///< Element type of C matrix  
  typename LayoutC_,                                 ///< Layout of C matrix (concept: MatrixLayout)
  typename Operator_ = cutlass::arch::OpMultiplyAdd  ///< Inner product operator (multiply-add, xor.popc)
>
struct Wmma;
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA store for matrix D
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                           ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                         ///< Data type of elements 
  typename Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared    ///< Data resides in shared or global memory
>
struct WmmaStoreD;
/////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// Partial specializations for cutlass::arch::emu::wmma templates for wmma.mma.8m8n4k.f64
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//
// WMMALoad[A|B|C] global memory loads for matrix size of 8x8x4 targeting DMMA.884
//
//////////////////////////////////////////////////////////////////////////////////////////
// Wmma load matrix A. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadA<  
  gemm::GemmShape<8,8,4>,           ///< Size of the matrix product (concept: GemmShape)
  double,                           ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<8,8,4>;
  using Element = double;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  using Fragment = Array<double, 1>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(      
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
    // USE CUTLASS WMMA CUDA C++ implementation
    int lane_id = cutlass::arch::LaneId();
    int quad_id = lane_id >> 2;
    int lane_in_quad = lane_id & 0x3;

    uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
    uint64_t const *data = reinterpret_cast<uint64_t const *>(ptr);

    Layout layout(ldm);

    reg[0] = data[layout(MatrixCoord{quad_id, lane_in_quad})];
#else
    // USE PTX Instruction
    PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
    ptx(frag, ptr, ldm);
#endif //CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. Supported layouts (ColumnMajor, RowMajor)
template <typename Layout_> 
struct WmmaLoadB<  
  gemm::GemmShape<8,8,4>,           ///< Size of the matrix product (concept: GemmShape)
  double,                           ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<8,8,4>;
  using Element = double;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  using Fragment = Array<double, 1>;

  // load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
   // Use CUTLASS WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;

  uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
  uint64_t const *data = reinterpret_cast<uint64_t const *>(ptr);

  Layout layout(ldm);

  reg[0] = data[layout(MatrixCoord{lane_in_quad, quad_id})];
#else
    // USE PTX Instruction
    PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
    ptx(frag, ptr, ldm);
#endif //CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix C. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadC<  
  gemm::GemmShape<8,8,4>,           ///< Size of the matrix product (concept: GemmShape)
  double,                           ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<8,8,4>;
  using Element = double;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  using Fragment = Array<double, 2>;

  // load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
    // Use CUTLASS WMMA CUDA C++ implementation
    int lane_id = cutlass::arch::LaneId();
    int quad_id = lane_id >> 2;
    int lane_in_quad = lane_id & 0x3;
    
    uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
    uint64_t const *data = reinterpret_cast<uint64_t const *>(ptr);
    
    Layout layout(ldm);
    
    reg[0] = data[layout(MatrixCoord{quad_id, (lane_in_quad*2 + 0)})];
    reg[1] = data[layout(MatrixCoord{quad_id, (lane_in_quad*2 + 1)})];

#else
    // USE PTX Instruction
    PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
    ptx(frag, ptr, ldm);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// WMMA for matrix size of 8x8x4 targeting DMMA.884 (F64)
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: F64 = F64 * F64 + F64

/*********************************************************************************
 NOTE: Wmma DMMA CUDA C++ API encloses _mma.m8n8k4.atype.btype.f64.f64.f64.f64 
 Currently cutlass::arch::Mma only implements atype.btype == row.col
 When additional atype.btype _mma are avialable we can add those specializations
 to WMMA DMMA API. 
*********************************************************************************/
template <>
struct Wmma<
  gemm::GemmShape<8,8,4>,                   ///< Size of the matrix product (concept: GemmShape)
  double,                                   ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  double,                                   ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  double,                                   ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add)
> {

  using WmmaShape = gemm::GemmShape<8,8,4>;
  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using Operator = cutlass::arch::OpMultiplyAdd;

  // MMA cutlass::arch::Mma instance to be used
  using Mma = Mma<
    gemm::GemmShape<8,8,4>,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Operator>;

  // Fragment per thread
  using FragmentA = Array<double, 1>;
  using FragmentB = Array<double, 1>;
  using FragmentC = Array<double, 2>;

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA const &a, 
    FragmentB const &b,
    FragmentC const &c) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
    // Use CUTLASS WMMA CUDA C++ implementation
    Mma mma;
    mma(d, a, b, c);
#else
    // Use PTX instruction
    PtxWmma<
      WmmaShape,                  
      ElementA,                                 
      LayoutA,                        
      ElementB,                                  
      LayoutB,                  
      ElementC,                                   
      LayoutC,                         
      Operator
    > ptx;

    ptx(d, a, b, c);
#endif //CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

////////////////////////////////////////////////////////////////////////////////////
//
// WMMAStoreD global memory loads for matrix size of 8x8x4 targeting DMMA.884 (F64)
//
////////////////////////////////////////////////////////////////////////////////////

// Wmma store matrix D. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaStoreD<  
  gemm::GemmShape<8,8,4>,           ///< Size of the matrix product (concept: GemmShape)
  double,                           ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<8,8,4>;
  using Element = double;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  using Fragment = Array<double, 2>;

  // Store from fragment to matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment const &frag,
    Element * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
   // Use CUTLASS WMMA CUDA C++ implementation
   int lane_id = cutlass::arch::LaneId();
   int quad_id = lane_id >> 2;
   int lane_in_quad = lane_id & 0x3;

   Layout layout(ldm);
    
   uint64_t const *reg = reinterpret_cast<uint64_t const *>(&frag);
   uint64_t *data = reinterpret_cast<uint64_t *>(ptr);
    
   data[layout(MatrixCoord{quad_id, (lane_in_quad*2 + 0)})] = reg[0];
   data[layout(MatrixCoord{quad_id, (lane_in_quad*2 + 1)})] = reg[1];
#else
   // USE PTX Instruction
   PtxWmmaStoreD<WmmaShape, Element, Layout, kMemory> ptx;
   ptx(frag, ptr, ldm);
#endif //CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////
// Partial specializations for cutlass::arch::emu::wmma templates for wmma.mma.16m8n4k.TF32
///////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMALoad[A|B|C] global memory loads for matrix size of 16x8x4 targeting HMMA.1684.F32.TF32
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Wmma load matrix A. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadA<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  //static int const kNumElementsPerTile = cutlass::platform::is_same<Element, tfloat32_t>::value ? 1 : 2;
  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kK/kTileSizeY;
  
  // Fragment per thread
  using Fragment = Array<Element, 2>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }
#else
  // USE PTX Instruction
  PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. Supported layouts (ColumnMajor, RowMajor)
template <typename Layout_> 
struct WmmaLoadB<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  //static int const kNumElementsPerTile = cutlass::platform::is_same<Element, tfloat32_t>::value ? 1 : 2;
  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kQuadsPerWarp;                          // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kK/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 1>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + lane_in_quad;
        int col = (c * kTileSizeY) + (quad_id * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Matrix C RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadC<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 4>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// WMMA for matrix size of 16x8x4 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: F32.row = TF32.row * TF32.col + F32.row

template <>
struct Wmma<
  gemm::GemmShape<16,8,4>,                  ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                            ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  tfloat32_t,                            ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  float,                                    ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using Operator = cutlass::arch::OpMultiplyAdd;

  // MMA cutlass::arch::Mma instance to be used
  using Mma = Mma<
    WmmaShape,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Operator>;

  // Fragment per thread
  using FragmentA = Array<ElementA, 2>;
  using FragmentB = Array<ElementB, 1>;
  using FragmentC = Array<ElementC, 4>;

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA const &a, 
    FragmentB const &b,
    FragmentC const &c) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  Mma mma;
  mma(d, a, b, c);

#else

  // Use PTX instruction
  PtxWmma<
    WmmaShape,                  
    ElementA,                                 
    LayoutA,                        
    ElementB,                                  
    LayoutB,                  
    ElementC,                                   
    LayoutC,                         
    Operator
  > ptx;

  ptx(d, a, b, c);

#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION

  }
};

/////////////////////////////////////////////////////////////////////////////////////////
//
// WMMAStoreD global memory loads for matrix size of 16x8x4 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////////////

// Matrix D RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaStoreD<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                            // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;    // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 4>;
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t const *reg = reinterpret_cast<uint32_t const *>(&frag);
  uint32_t *data = reinterpret_cast<uint32_t *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        data[layout(MatrixCoord{row, col})] = reg[reg_idx++];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Partial specializations for cutlass::arch::emu::wmma templates for wmma.mma.16m16n4k.TF32
////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMALoad[A|B|C] global memory loads for matrix size of 16x16x4 targeting HMMA.1684.F32.TF32
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Wmma load matrix A. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadA<  
  gemm::GemmShape<16,16,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,4>;
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  //static int const kNumElementsPerTile = cutlass::platform::is_same<Element, tfloat32_t>::value ? 1 : 2;
  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kK/kTileSizeY;
  
  // Fragment per thread
  using Fragment = Array<Element, 2>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];      
      }
    }
  }
#else
  // USE PTX Instruction
  PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. Supported layouts (ColumnMajor, RowMajor)
template <typename Layout_> 
struct WmmaLoadB<  
  gemm::GemmShape<16,16,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,4>;
  
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  //static int const kNumElementsPerTile = cutlass::platform::is_same<Element, tfloat32_t>::value ? 1 : 2;
  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kQuadsPerWarp;                          // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kK/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 2>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + lane_in_quad;
        int col = (c * kTileSizeY) + (quad_id * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];        
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Matrix C RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadC<  
  gemm::GemmShape<16,16,4>,         ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,4>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 8>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// WMMA for matrix size of 16x16x4 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: F32.row = TF32.row * TF32.col + F32.row

template <>
struct Wmma<
  gemm::GemmShape<16,16,4>,                 ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                            ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  tfloat32_t,                            ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  float,                                    ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {

  using WmmaShape = gemm::GemmShape<16,16,4>;
  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using Operator = cutlass::arch::OpMultiplyAdd;
  
  // Wmma Fragments per thread
  using FragmentA = Array<ElementA, 2>;
  using FragmentB = Array<ElementB, 2>;
  using FragmentC = Array<ElementC, 8>;
  
  // Wmma 16164 TF32 targeting HMMA.1684.F32.TF32
  using InstructionShape = gemm::GemmShape<16,8,4>;
    
  // Create HMMA.1684.F32.TF32 instruction type using cutlass::arch::Mma 
  using Mma = cutlass::arch::Mma<
    InstructionShape,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Operator>;

  // Mma Fragments per thread
  using MmaFragmentA = Array<ElementA, 2>;
  using MmaFragmentB = Array<ElementB, 1>;
  using MmaFragmentC = Array<ElementC, 4>;

  //static int const kMmaTileX = WmmaShape::kM/InstructionShape::kM;
  static int const kMmaTileY = WmmaShape::kN/InstructionShape::kN; 

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA const &a, 
    FragmentB const &b,
    FragmentC const &c) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  Mma mma;

  // Get MmaFragments from Wmma Fragments
  MmaFragmentA const *A = reinterpret_cast<MmaFragmentA const*>(&a);
  MmaFragmentB const *B = reinterpret_cast<MmaFragmentB const*>(&b);
  MmaFragmentC const *C = reinterpret_cast<MmaFragmentC const*>(&c);
  MmaFragmentC *D = reinterpret_cast<MmaFragmentC *>(&d); 

  for(int col=0; col<kMmaTileY; col++) {

    mma(D[col], A[0], B[col], C[col]);

  }

#else

  // Use PTX instruction
  PtxWmma<
    WmmaShape,                  
    ElementA,                                 
    LayoutA,                        
    ElementB,                                  
    LayoutB,                  
    ElementC,                                   
    LayoutC,                         
    Operator
  > ptx;

  ptx(d, a, b, c);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION

  }
};

/////////////////////////////////////////////////////////////////////////////////////////
//
// WMMAStoreD global memory loads for matrix size of 16x16x4 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////////////

// Matrix D RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaStoreD<  
  gemm::GemmShape<16,16,4>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,4>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                            // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;    // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 8>;
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t const *reg = reinterpret_cast<uint32_t const *>(&frag);
  uint32_t *data = reinterpret_cast<uint32_t *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        data[layout(MatrixCoord{row, col})] = reg[reg_idx++];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Partial specializations for cutlass::arch::emu::wmma templates for wmma.mma.16m16n8k.TF32
////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMALoad[A|B|C] global memory loads for matrix size of 16x16x8 targeting HMMA.1684.F32.TF32
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Wmma load matrix A. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadA<  
  gemm::GemmShape<16,16,8>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,8>;
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kK/kTileSizeY;
  
  // Fragment per thread
  using Fragment = Array<Element, 4>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }
#else
  // USE PTX Instruction
  PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. Supported layouts (ColumnMajor, RowMajor)
template <typename Layout_> 
struct WmmaLoadB<  
  gemm::GemmShape<16,16,8>,         ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,8>;
  
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 1;
  static int const kTileSizeX = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kQuadsPerWarp;                          // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kK/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 4>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + lane_in_quad;
        int col = (c * kTileSizeY) + (quad_id * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Matrix C RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaLoadC<  
  gemm::GemmShape<16,16,8>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,8>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 8>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// WMMA for matrix size of 16x16x8 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: F32.row = TF32.row * TF32.col + F32.row

template <>
struct Wmma<
  gemm::GemmShape<16,16,8>,                 ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                            ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  tfloat32_t,                            ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  float,                                    ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {

  using WmmaShape = gemm::GemmShape<16,16,8>;
  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using Operator = cutlass::arch::OpMultiplyAdd;
  
  // Wmma Fragments per thread
  using FragmentA = Array<ElementA, 4>;
  using FragmentB = Array<ElementB, 4>;
  using FragmentC = Array<ElementC, 8>;

  // Wmma 16168 TF32 targeting HMMA.1684.F32.TF32
  using InstructionShape = gemm::GemmShape<16,8,4>;
  
  // Create HMMA.1684.F32.TF32 instruction type using cutlass::arch::Mma 
  using Mma = cutlass::arch::Mma<
    InstructionShape,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Operator>;

  // Mma Fragments per thread
  using MmaFragmentA = Array<ElementA, 2>;
  using MmaFragmentB = Array<ElementB, 1>;
  using MmaFragmentC = Array<ElementC, 4>;

  static int const kMmaTileM = WmmaShape::kM/InstructionShape::kM;
  static int const kMmaTileN = WmmaShape::kN/InstructionShape::kN; 
  static int const kMmaTileK = WmmaShape::kK/InstructionShape::kK; 

  // Check kMmaTileM
  static_assert((kMmaTileM==1), "CUDA C++ emulation code for WMMA 16168 targeting HMM 1684 needs kMmaTileM value equal to 1");

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA const &a, 
    FragmentB const &b,
    FragmentC const &c) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  Mma mma;

  // Accmulate output in d <= a * b + c
  d = c;

  // Get MmaFragments from Wmma Fragments
  MmaFragmentA const *A = reinterpret_cast<MmaFragmentA const*>(&a);
  MmaFragmentB const *B = reinterpret_cast<MmaFragmentB const*>(&b);
  MmaFragmentC *D = reinterpret_cast<MmaFragmentC *>(&d); 

  CUTLASS_PRAGMA_UNROLL
  for(int k=0; k<kMmaTileK; k++) {
    CUTLASS_PRAGMA_UNROLL
    for(int n=0; n<kMmaTileN; n++) {
      mma(D[n], A[k], B[n * kMmaTileN + k], D[n]);
    }
  }

#else

  // Use PTX instruction
  PtxWmma<
    WmmaShape,                  
    ElementA,                                 
    LayoutA,                        
    ElementB,                                  
    LayoutB,                  
    ElementC,                                   
    LayoutC,                         
    Operator
  > ptx;

  ptx(d, a, b, c);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION

  }
};

////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMAStoreD global memory loads for matrix size of 16x16x8 targeting HMMA.1684.F32.TF32
//
////////////////////////////////////////////////////////////////////////////////////////////

// Matrix D RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct WmmaStoreD<  
  gemm::GemmShape<16,16,8>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                           ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,16,8>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                            // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;    // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, 8>;
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t const *reg = reinterpret_cast<uint32_t const *>(&frag);
  uint32_t *data = reinterpret_cast<uint32_t *>(ptr);

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        data[layout(MatrixCoord{row, col})] = reg[reg_idx++];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Partial specializations for cutlass::arch::emu::wmma templates for wmma.mma.16m16n16k.s8 and wmma.mma.16m16n16k.s8
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMALoad[A|B|C] global memory loads for matrix size of 16x16x16 and 32x8x16 targeting IMMA.16816.S8.S8
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Wmma load matrix A. row-major layout  int8 loads can be vectorized as 32-bit loads
template <typename WmmaShape_> 
struct WmmaLoadA<  
  WmmaShape_,        ///< Size of the matrix product (concept: GemmShape)
  int8_t,                           ///< Data type of elements 
  layout::RowMajor,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  using Element = int8_t;
  using Layout = layout::RowMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  static int const kNumElementsPerTile = 4;
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kK/kTileSizeY;
  
  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;


  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm/4);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile/4; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col/4})];
      }
    }
  }
#else
  // USE PTX Instruction
  PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};


// Wmma load matrix A. column-major layout int8 loads
template <typename WmmaShape_> 
struct WmmaLoadA<  
  WmmaShape_,        ///< Size of the matrix product (concept: GemmShape)
  int8_t,                           ///< Data type of elements 
  layout::ColumnMajor,              ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  using Element = int8_t;
  using Layout = layout::ColumnMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  static int const kNumElementsPerTile = 4;
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kK/kTileSizeY;
  
  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        frag[reg_idx++] = ptr[layout(MatrixCoord{row, col})];
      }
    }
  }
#else
  // USE PTX Instruction
  PtxWmmaLoadA<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. column-major layout int8 loads can be vectorized as 32-bit loads
template <typename WmmaShape_> 
struct WmmaLoadB<  
  WmmaShape_,                       ///< Size of the matrix product (concept: GemmShape)
  int8_t,                           ///< Data type of elements 
  layout::ColumnMajor,              ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int8_t;
  using Layout = layout::ColumnMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 4;
  static int const kTileSizeX = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kQuadsPerWarp;                          // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kK/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  uint32_t const *data = reinterpret_cast<uint32_t const *>(ptr);

  Layout layout(ldm/4);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile/4; i++) {
        int row = (r * kTileSizeX) + (lane_in_quad * kNumElementsPerTile) + i;
        int col = (c * kTileSizeY) + quad_id;
        reg[reg_idx++] = data[layout(MatrixCoord{row/4, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Wmma load matrix B. row-major layout int8 loads
template <typename WmmaShape_> 
struct WmmaLoadB<  
  WmmaShape_,                    ///< Size of the matrix product (concept: GemmShape)
  int8_t,                        ///< Data type of elements 
  layout::RowMajor,              ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal            ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int8_t;
  using Layout = layout::RowMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 4;
  static int const kTileSizeX = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kQuadsPerWarp;                          // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kK/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + (lane_in_quad * kNumElementsPerTile) + i;
        int col = (c * kTileSizeY) + quad_id;
        frag[reg_idx++] = ptr[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadB<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

//  Wmma load matrix C. row-major layout int 32-bit loads can be vectorized as 64-bit loads
template <typename WmmaShape_> 
struct WmmaLoadC<  
  WmmaShape_,                       ///< Size of the matrix product (concept: GemmShape)
  int,                              ///< Data type of elements 
  layout::RowMajor,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int;
  using Layout = layout::RowMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
  uint64_t const *data = reinterpret_cast<uint64_t const *>(ptr);

  Layout layout(ldm/2);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile/2; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        reg[reg_idx++] = data[layout(MatrixCoord{row, col/2})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

//  Wmma load matrix C. column-major layout int loads
template <typename WmmaShape_> 
struct WmmaLoadC<  
  WmmaShape_,        ///< Size of the matrix product (concept: GemmShape)
  int,                              ///< Data type of elements 
  layout::ColumnMajor,              ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int;
  using Layout = layout::ColumnMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                          // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;  // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        frag[reg_idx++] = ptr[layout(MatrixCoord{row, col})];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};


////////////////////////////////////////////////////////////////////////////////
//
// WMMA for matrix size of 16x16x16 targeting IMMA.16816.S8.S8
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: S32.row = S8.row * S8.col + S32.row

template <typename WmmaShape_>
struct Wmma<
  WmmaShape_,                               ///< Size of the matrix product (concept: GemmShape)
  int8_t,                                   ///< ElementA
  layout::RowMajor,                         ///< LayoutA
  int8_t,                                   ///< ElementB
  layout::ColumnMajor,                      ///< LayoutB
  int,                                      ///< ElementC
  layout::RowMajor,                         ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {

  using WmmaShape = WmmaShape_;
  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;
  

  // Wmma 161616 S8 targeting IMMA.16816.S8.S8
  using InstructionShape = gemm::GemmShape<16,8,16>;
  
  // Create IMMA.16816.S8.S8 instruction type using cutlass::arch::Mma 
  using Mma = cutlass::arch::Mma<
    InstructionShape,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Operator>;

  static int const kMmaTileM = WmmaShape::kM/InstructionShape::kM;
  static int const kMmaTileN = WmmaShape::kN/InstructionShape::kN; 
  static int const kMmaTileK = WmmaShape::kK/InstructionShape::kK;

  // Mma Fragments per thread
  using MmaFragmentA = Array<ElementA, 8>;
  using MmaFragmentB = Array<ElementB, 4>;
  using MmaFragmentC = Array<ElementC, 4>; 

  // Wmma Fragments per thread
  using FragmentA = Array<ElementA, int(MmaFragmentA::kElements * kMmaTileM * kMmaTileK)>;
  using FragmentB = Array<ElementB, int(MmaFragmentB::kElements * kMmaTileK * kMmaTileN)>;
  using FragmentC = Array<ElementC, int(MmaFragmentC::kElements * kMmaTileM * kMmaTileN)>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA const &a, 
    FragmentB const &b,
    FragmentC const &c) const {

#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  Mma mma;

  // Accmulate output in d <= a * b + c
  d = c;

  // Get MmaFragments from Wmma Fragments
  MmaFragmentA const *A = reinterpret_cast<MmaFragmentA const*>(&a);
  MmaFragmentB const *B = reinterpret_cast<MmaFragmentB const*>(&b);
  MmaFragmentC *D = reinterpret_cast<MmaFragmentC *>(&d); 

  CUTLASS_PRAGMA_UNROLL
  for(int m=0; m<kMmaTileM; m++) {
    CUTLASS_PRAGMA_UNROLL
    for(int n=0; n<kMmaTileN; n++) {
      mma(D[n * kMmaTileM + m], A[m], B[n], D[n * kMmaTileM + m]);
    }
  }

#else

  // Use PTX instruction
  PtxWmma<
    WmmaShape,                  
    ElementA,                                 
    LayoutA,                        
    ElementB,                                  
    LayoutB,                  
    ElementC,                                   
    LayoutC,                         
    Operator
  > ptx;

  ptx(d, a, b, c);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION

  }
};

////////////////////////////////////////////////////////////////////////////////////////////
//
// WMMAStoreD global memory loads for matrix size of 16x16x16 targeting IMMA.16816.S8.S8
//
////////////////////////////////////////////////////////////////////////////////////////////

// Matrix D RowMajor. row-major layout int 32-bit stores can be vectorized as 64-bit loads
template <typename WmmaShape_> 
struct WmmaStoreD<  
  WmmaShape_,                       ///< Size of the matrix product (concept: GemmShape)
  int,                              ///< Data type of elements 
  layout::RowMajor,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int;
  using Layout = layout::RowMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                            // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;    // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  
  
  uint64_t const *reg = reinterpret_cast<uint64_t const *>(&frag);
  uint64_t *data = reinterpret_cast<uint64_t *>(ptr);

  Layout layout(ldm/2);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile/2; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        data[layout(MatrixCoord{row, col/2})] = reg[reg_idx++];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

// Matrix D ColumnMajor. column-major layout int 32-bit stores
template <typename WmmaShape_> 
struct WmmaStoreD<  
  WmmaShape_,        ///< Size of the matrix product (concept: GemmShape)
  int,                              ///< Data type of elements 
  layout::ColumnMajor,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = WmmaShape_;
  
  using Element = int;
  using Layout = layout::ColumnMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  static int const kNumElementsPerTile = 2; 
  static int const kTileSizeX = WarpParams::kQuadsPerWarp;                            // Number of tiles in the M dimension for matrix
  static int const kTileSizeY = WarpParams::kThreadsPerQuad * kNumElementsPerTile;    // Number of tiles in the N dimension for matrix
  static int const kNumTilesX = WmmaShape::kM/kTileSizeX;
  static int const kNumTilesY = WmmaShape::kN/kTileSizeY;

  // Fragment per thread
  using Fragment = Array<Element, kNumElementsPerTile * kNumTilesX * kNumTilesY>;

  // check supported wmma shape for the given multiplicand data types and target mma instruction
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, WmmaShape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 32, 8, 16>, WmmaShape>::value,
    "Supported list of wmma shape for int8_t multiplicands targeting IMMA.16816: 16x16x16 and 32x8x16");
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element * ptr,
    int ldm) const {


#if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  // WMMA CUDA C++ implementation
  int lane_id = cutlass::arch::LaneId();
  int quad_id = lane_id >> 2;
  int lane_in_quad = lane_id & 0x3;  

  Layout layout(ldm);

  int reg_idx = 0;
  CUTLASS_PRAGMA_UNROLL
  for(int c=0; c<kNumTilesY; c++) {
    CUTLASS_PRAGMA_UNROLL
    for(int r=0; r<kNumTilesX; r++) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i<kNumElementsPerTile; i++) {
        int row = (r * kTileSizeX) + quad_id;
        int col = (c * kTileSizeY) + (lane_in_quad * kNumElementsPerTile) + i;
        ptr[layout(MatrixCoord{row, col})] = frag[reg_idx++];
      }
    }
  }

#else
  // USE PTX Instruction
  PtxWmmaLoadC<WmmaShape, Element, Layout, kMemory> ptx;
  ptx(frag, ptr, ldm);
#endif // #if CUTLASS_USE_SM80_WMMA_CUDA_EMULATION
  }
};

} // namespace emu
} // namespace arch
} // namespace cutlass
