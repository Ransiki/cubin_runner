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
    \brief Describes convolution-specific parameters as compile-time constants.
*/

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace contraction {
    struct GettCoord : public Coord<6, uint32_t> {
        static const int kRank = 6;

      CUTLASS_HOST_DEVICE
      GettCoord() { }

      CUTLASS_HOST_DEVICE
      GettCoord( Index blockedM,
                 Index unblockedM,
                 Index blockedN,
                 Index unblockedN,
                 Index k,
                 Index batched)
      {
          this->at(kBlockedM) = blockedM;
          this->at(kUnblockedM) = unblockedM;
          this->at(kBlockedN) = blockedN;
          this->at(kUnblockedN) = unblockedN;
          this->at(kK) = k;
          this->at(kBatched) = batched;
      }

      CUTLASS_HOST_DEVICE
      GettCoord(const GettCoord &other) : Coord<kRank, Index>(other) { 
          this->at(kBlockedM)   = other.at(kBlockedM);
          this->at(kUnblockedM) = other.at(kUnblockedM);
          this->at(kBlockedN)   = other.at(kBlockedN);
          this->at(kUnblockedN) = other.at(kUnblockedN);
          this->at(kK)          = other.at(kK);
          this->at(kBatched)    = other.at(kBatched);
      }

      /// Assignment
      CUTLASS_HOST_DEVICE
      GettCoord& operator=(GettCoord const& other) {
          this->at(kBlockedM)   = other.at(kBlockedM);
          this->at(kUnblockedM) = other.at(kUnblockedM);
          this->at(kBlockedN)   = other.at(kBlockedN);
          this->at(kUnblockedN) = other.at(kUnblockedN);
          this->at(kK)          = other.at(kK);
          this->at(kBatched)    = other.at(kBatched);
          return *this;
      }

      static int const kBlockedM = 0;
      static int const kUnblockedM = 1;
      static int const kBlockedN = 2;
      static int const kUnblockedN = 3;

      /// GEMM K dimension - inner dimension of the GEMM problem
      static int const kK = 4;

      /// batched dimension
      static int const kBatched = 5;

      CUTLASS_HOST_DEVICE
      Index const& blockedM() const { return this->at(kBlockedM); }

      CUTLASS_HOST_DEVICE
      Index const& unblockedM() const { return this->at(kUnblockedM); }

      CUTLASS_HOST_DEVICE
      Index const& blockedN() const { return this->at(kBlockedN); }

      CUTLASS_HOST_DEVICE
      Index const& unblockedN() const { return this->at(kUnblockedN); }

      CUTLASS_HOST_DEVICE
      Index const& k() const { return this->at(kK); }

      CUTLASS_HOST_DEVICE
      Index const& batched() const { return this->at(kBatched); }


      CUTLASS_HOST_DEVICE
      Index & blockedM() { return this->at(kBlockedM); }

      CUTLASS_HOST_DEVICE
      Index & unblockedM() { return this->at(kUnblockedM); }

      CUTLASS_HOST_DEVICE
      Index & blockedN() { return this->at(kBlockedN); }

      CUTLASS_HOST_DEVICE
      Index & unblockedN() { return this->at(kUnblockedN); }

      CUTLASS_HOST_DEVICE
      Index & k() { return this->at(kK); }

      CUTLASS_HOST_DEVICE
      Index & batched() { return this->at(kBatched); }

      CUTLASS_HOST_DEVICE
      Index product() const
      {
          Index total = 1;
          total *= this->blockedM();
          total *= this->unblockedM();
          total *= this->blockedN();
          total *= this->unblockedN();
          total *= this->k();
          total *= this->batched();
          return total;
      }

   };

} // namespace contraction
} // namespace cutlass
