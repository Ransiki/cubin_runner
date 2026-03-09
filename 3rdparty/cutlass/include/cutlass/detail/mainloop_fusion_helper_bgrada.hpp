/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief SM100 Mainloop Fusion configs specific for BgradA
*/

#pragma once

#include <cute/util/type_traits.hpp> // cute::void_t

namespace cutlass::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CollectiveMainloop, typename = void>
struct Sm100HasMainloopFusion{
  static constexpr bool value = false;
};

template <typename CollectiveMainloop>
struct Sm100HasMainloopFusion<CollectiveMainloop, cute::void_t<decltype(CollectiveMainloop::HasMainloopFusion)>> {
  static constexpr bool value = true;
};

template <typename CollectiveMainloop, typename Default, typename = void>
struct Sm100TileShapeUpdateNType {
  using type = Default;
};

template <typename CollectiveMainloop, typename Default>
struct Sm100TileShapeUpdateNType<CollectiveMainloop, Default, cute::void_t<typename CollectiveMainloop::TileShapeBgradA>> {
  using type = typename CollectiveMainloop::TileShapeBgradA;
};

template <typename CollectiveMainloop, typename Default, typename = void>
struct Sm100TiledMmaBgradAType {
  using type = Default;
};

template <typename CollectiveMainloop, typename Default>
struct Sm100TiledMmaBgradAType<CollectiveMainloop, Default, cute::void_t<typename CollectiveMainloop::TiledMma_BgradA>> {
  using type = typename CollectiveMainloop::TiledMma_BgradA;
};

template <typename CollectiveMainloop, typename Default, typename = void>
struct Sm100CtaShapeMNKUpdateNType {
  using type = Default;
};

template <typename CollectiveMainloop, typename Default>
struct Sm100CtaShapeMNKUpdateNType<CollectiveMainloop, Default, cute::void_t<typename CollectiveMainloop::CtaShape_MNK_BgradA>> {
  using type = typename CollectiveMainloop::CtaShape_MNK_BgradA;
};


template <typename CollectiveMainloop, typename Default, typename = void>
struct Sm100ElementMainloopFusionType {
  using type = Default;
};

template <typename CollectiveMainloop, typename Default>
struct Sm100ElementMainloopFusionType<CollectiveMainloop, Default, cute::void_t<typename CollectiveMainloop::ElementBgradA>> {
  using type = typename CollectiveMainloop::ElementBgradA;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail