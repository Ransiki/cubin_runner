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
/*! \file
    \brief Templates exposing warp matrix multiply-add (WMMA) operations
*/
#pragma once
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"

namespace cutlass {
namespace arch {
namespace emu {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// MemoryKind class (Shared vs. Global memory)
/////////////////////////////////////////////////////////////////////////////////////////////////
enum class MemoryKind {
  kShared,  // Data resides in shared memory
  kGlobal   // Data resides in global memory
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// WarpParams holds architecture-specific constants 
/////////////////////////////////////////////////////////////////////////////////////////////////
struct WarpParams {
  static int const kThreadsPerWarp = 32;
  static int const kQuadsPerWarp = 8;
  static int const kThreadsPerQuad = 4;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
///
///  WMMA structures to enclose * PTX * instruction string 
///
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA PTX string load for A, B, and C matrices 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                          ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                        ///< Data type of elements 
  typename Layout_,                         ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared   ///< Data resides in shared or global memory
>
struct PtxWmmaLoadA;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <  
  typename Shape_,                          ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                        ///< Data type of elements 
  typename Layout_,                         ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared   ///< Data resides in shared or global memory
>
struct PtxWmmaLoadB;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <  
  typename Shape_,                          ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                        ///< Data type of elements 
  typename Layout_,                         ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared   ///< Data resides in shared or global memory
>
struct PtxWmmaLoadC;
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA Matrix multiply-add operation
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
  typename ElementA_,                                ///< Data type of A elements 
  typename LayoutA_,                                 ///< Layout of A matrix (concept: MatrixLayout)  
  typename ElementB_,                                ///< Data type of B elements
  typename LayoutB_,                                 ///< Layout of B matrix (concept: MatrixLayout)  
  typename ElementC_,                                ///< Element type of C matrix  
  typename LayoutC_,                                 /// Layout of C matrix (concept: MatrixLayout)
  typename Operator = cutlass::arch::OpMultiplyAdd   ///< Inner product operator (multiply-add, xor.popc)
>
struct PtxWmma;
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// WMMA store for matrix D
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                          ///< Size of the matrix product (concept: GemmShape)
  typename Element_,                        ///< Data type of elements 
  typename Layout_,                         ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind Memory = MemoryKind::kShared   ///< Data resides in shared or global memory
>
struct PtxWmmaStoreD;
/////////////////////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}

/// Specialization of PTX WMMA structures targeting DMMA
// Wmma load matrix A Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaLoadA<  
  gemm::GemmShape<8,8,4>,           ///< Size of the matrix product (concept: GemmShape)
  double,                           ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using Shape = gemm::GemmShape<8,8,4>;
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

    // USE PTX
    uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);

    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          "wmma.load.a.aligned.global.sync.row.m8n8k4.f64  {%0}, [%1], %2;\n"
          : "=l"(reg[0])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if (cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          "wmma.load.a.aligned.global.sync.col.m8n8k4.f64  {%0}, [%1], %2;\n"
          : "=l"(reg[0])
          : "l"(ptr), "r"(ldm)
      );
    }
    else {
      assert(false);
    }
  }

};

template <typename Layout_> 
struct PtxWmmaLoadB<  
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

    uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
  
    if(cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          "wmma.load.b.aligned.global.sync.col.m8n8k4.f64  {%0}, [%1], %2;\n"
          : "=l"(reg[0])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          "wmma.load.b.aligned.global.sync.col.m8n8k4.f64  {%0}, [%1], %2;\n"
          : "=l"(reg[0])
          : "l"(ptr), "r"(ldm)
      );
    }
    else {
      assert(false);
    }
  }
};


// Wmma load matrix C Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaLoadC<  
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

    uint64_t *reg = reinterpret_cast<uint64_t *>(&frag);
    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          "wmma.load.c.aligned.global.sync.row.m8n8k4.f64  {%0, %1}, [%2], %3;\n"
          : "=l"(reg[0]), "=l"(reg[1])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if (cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          "wmma.load.c.aligned.global.sync.col.m8n8k4.f64  {%0}, [%1], %2;\n"
          : "=l"(reg[0])
          : "l"(ptr), "r"(ldm)
      );
    }
    else {
      assert(false);
    }

  }
};

template <>
struct PtxWmma<
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

    // Use PTX instruction
    uint64_t const *A = reinterpret_cast<uint64_t const *>(&a);
    uint64_t const *B = reinterpret_cast<uint64_t const *>(&b);
    uint64_t const *C = reinterpret_cast<uint64_t const *>(&c);
    uint64_t *D = reinterpret_cast<uint64_t *>(&d);
  
    asm(
      "wmma.mma.aligned.sync.row.col.m8n8k4.f64.f64  {%0, %1}, {%2}, {%3}, {%4, %5}; \n" 
      : "=l"(D[0]), "=l"(D[1])
      : "l"(A[0]), "l"(B[0]), "l"(C[0]), "l"(C[1])
       );
    }
};

// Wmma store matrix D Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaStoreD<  
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

    uint64_t const *reg = reinterpret_cast<uint64_t const *>(&frag);
    
    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          "wmma.store.d.sync.aligned.row.m8n8k4.f64  [%0], {%1, %2}, %3;"
          : 
          : "l"(ptr), "l"(reg[0]), "l"(reg[1]), "r"(ldm)
      );
    }
    else if(cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          "wmma.store.d.sync.aligned.col.m8n8k4.f64  [%0], {%1, %2}, %3;"
          : 
          : "l"(ptr), "l"(reg[0]), "l"(reg[1]), "r"(ldm)
      );
    }
    else {
      assert(0);
    }
  }

};

//
/// Specialization of PTX WMMA structures targeting HMMA.1684.F32.TF32

// Wmma load matrix A Supported layouts (RowMajor, ColumnMajor)template <typename Layout_> 
template <typename Layout_> 
struct PtxWmmaLoadA<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                       ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 
  
  
  // Fragment per thread
  using Fragment = Array<Element, 2>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

    uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);

    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.load.a.aligned.global.sync.row.m16n8k4.tf32  {%0, %1}, [%2], %3;\n"
          #else
          "wmma.load.a.aligned.global.sync.row.m16n8k4.fe8m10  {%0, %1}, [%2], %3;\n"
          #endif
          : "=r"(reg[0]), "=r"(reg[1])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if (cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.load.a.aligned.global.sync.col.m16n8k4.tf32  {%0, %1}, [%2], %3;\n"
          #else
          "wmma.load.a.aligned.global.sync.col.m16n8k4.fe8m10  {%0, %1}, [%2], %3;\n"
          #endif
          : "=r"(reg[0]), "=r"(reg[1])
          : "l"(ptr), "r"(ldm)
      );
    }
   else {
      assert(0);
    }
  }
};

// Wmma load matrix B Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaLoadB<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  tfloat32_t,                    ///< Data type of elements 
  Layout_,                          ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = tfloat32_t;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 

  // Fragment per thread
  using Fragment = Array<Element, 1>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {


    uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
    
    if (cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.load.b.aligned.global.sync.col.m16n8k4.tf32  {%0, %1}, [%2], %3;\n"
          #else
          "wmma.load.b.aligned.global.sync.col.m16n8k4.fe8m10  {%0, %1}, [%2], %3;\n"
          #endif
          : "=r"(reg[0]), "=r"(reg[1])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value){
        asm volatile(
        #if __CUDACC_VER_MAJOR__ >= 11
        "wmma.load.b.aligned.global.sync.row.m16n8k4.tf32  {%0, %1}, [%2], %3;\n"
        #else
        "wmma.load.b.aligned.global.sync.row.m16n8k4.fe8m10  {%0, %1}, [%2], %3;\n"
        #endif
        : "=r"(reg[0]), "=r"(reg[1])
        : "l"(ptr), "r"(ldm)
      );
    }
    else {
      assert(0);
    }
  }
};

// Wmma load matrix C Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaLoadC<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = float;
  using Layout = Layout_;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  // Fragment per thread
  using Fragment = Array<Element, 4>;

  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment &frag,
    Element const * ptr,
    int ldm) const {

    uint32_t *reg = reinterpret_cast<uint32_t *>(&frag);
  
    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.load.c.aligned.global.sync.row.m16n8k4.tf32  {%0, %1, %2, %3}, [%4], %5;\n"
          #else
          "wmma.load.c.aligned.global.sync.row.m16n8k4.fe8m10  {%0, %1, %2, %3}, [%4], %5;\n"
          #endif
          : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
          : "l"(ptr), "r"(ldm)
      );
    }
    else if(cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.load.c.aligned.global.sync.col.m16n8k4.tf32  {%0, %1, %2, %3}, [%4], %5;\n"
          #else
          "wmma.load.c.aligned.global.sync.col.m16n8k4.fe8m10  {%0, %1, %2, %3}, [%4], %5;\n"
          #endif
          : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
          : "l"(ptr), "r"(ldm)
      );
    }
    else {
      assert(0);
    }
  }
};

// Matrix D RowMajor. Supported layouts (RowMajor, ColumnMajor)
template <typename Layout_> 
struct PtxWmmaStoreD<  
  gemm::GemmShape<16,8,4>,          ///< Size of the matrix product (concept: GemmShape)
  float,                            ///< Data type of elements 
  Layout_,                 ///< Layout of matrix (concept: MatrixLayout)
  MemoryKind::kGlobal               ///< Data resides in shared or global memory
> {

  using WmmaShape = gemm::GemmShape<16,8,4>;
  
  using Element = float;
  using Layout = layout::RowMajor;
  static MemoryKind const kMemory = MemoryKind::kGlobal; 


  // Fragment per thread
  using Fragment = Array<Element, 4>;
  
  // Load into fragment from matrix in memory
  CUTLASS_DEVICE
  void operator()(    
    Fragment const &frag,
    Element * ptr,
    int ldm) const {


    uint32_t const *reg = reinterpret_cast<uint32_t const *>(&frag);

    if(cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.store.d.aligned.global.sync.row.m16n8k4.tf32  [%0], {%1, %2 %3 %4}, %5;\n"
          #else
          "wmma.store.d.aligned.global.sync.row.m16n8k4.fe8m10  [%0], {%1, %2 %3 %4}, %5;\n"
          #endif
          : 
          : "l"(ptr), "r"(reg[0]), "r"(reg[1]), "r"(reg[2]), "r"(reg[3]),  "r"(ldm)
      );
    }
    else if(cutlass::platform::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
      asm volatile(
          #if __CUDACC_VER_MAJOR__ >= 11
          "wmma.store.d.aligned.global.sync.col.m16n8k4.tf32  [%0], {%1, %2 %3 %4}, %5;\n"
          #else
          "wmma.store.d.aligned.global.sync.col.m16n8k4.fe8m10  [%0], {%1, %2 %3 %4}, %5;\n"
          #endif
          : 
          : "l"(ptr), "r"(reg[0]), "r"(reg[1]), "r"(reg[2]), "r"(reg[3]),  "r"(ldm)
      );
    }
    else {
      assert(0);
    }
  }
};

// {$nv-internal-release end}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace emu
} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
