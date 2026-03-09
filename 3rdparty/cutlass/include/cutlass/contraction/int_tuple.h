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
#pragma once

namespace cutlass {
namespace contraction {

template<int v>
struct Int
{
    static const int value = v;
};

template<typename... elements>
struct IntTuple
{
    static const int kRank = sizeof...(elements);
};

template<typename First, typename... Rest> struct Prepend;
template<typename First, typename... Rest>
struct Prepend<First, IntTuple<Rest...>>
{
    using Type = IntTuple<First, Rest...>;
};

/*
 * Counts all values of an IntTuple that are larger than 1
 */
template<typename Tuple, typename enable = void> struct Count;
template<typename First, typename... Rest>
struct Count<IntTuple<First, Rest...>, typename platform::enable_if<IntTuple<Rest...>::kRank != 0>::type>
{
    static const int value = (First::value > 1 ? 1 : 0) + Count<IntTuple<Rest...>>::value;
};

template<typename First, typename... Rest>
struct Count<IntTuple<First, Rest...>, typename platform::enable_if<IntTuple<Rest...>::kRank == 0>::type>
{
    static const int value = First::value > 1 ? 1 : 0;
};



/*
 * Multiplies all values from an IntTuple `Tuple`
 */
template<typename Tuple, typename enable = void> struct Product;

template<typename First, typename... Rest>
struct Product<IntTuple<First, Rest...>, typename platform::enable_if<IntTuple<Rest...>::kRank != 0>::type>
{
    static const int value = First::value * Product<IntTuple<Rest...>>::value;
};

template<typename First, typename... Rest>
struct Product<IntTuple<First, Rest...>, typename platform::enable_if<IntTuple<Rest...>::kRank == 0>::type>
{
    static const int value = First::value;
};

/*
 * At<i>: exctracted value at position i from the provided IntTuple `Tuple`
 */
template<int i, typename Tuple, typename enable = void> struct At;

template<typename First, typename... Rest>
struct At<0, IntTuple<First, Rest...>, void>
{
    static const int value = First::value;
};

template<int i, typename First, typename... Rest>
struct At<i, IntTuple<First, Rest...>, typename platform::enable_if<i!=0>::type>
{
    static const int value = At<i-1, IntTuple<Rest...>>::value;
};

/*
 * Multiplies all values from an IntTuple `Tuple` starting from 0 to end (inclusive)
 */
template<int idx, int end, typename Tuple, int multiplier, typename enable = void> struct ProductRange_;

template<int end, typename Tuple, int multiplier = 1> struct ProductRange
{
    static const int value = platform::conditional< (end < 0), Int<1>, ProductRange_<0, end, Tuple, multiplier>>::type::value;
};

template<int idx, int end, typename Tuple, int multiplier>
struct ProductRange_<idx, end, Tuple, multiplier, typename platform::enable_if<idx != end>::type>
{
    static const int value = (idx == 0 ? multiplier : 1 ) * At<idx, Tuple>::value * ProductRange_<idx+1, end, Tuple, multiplier>::value;
};
template<int idx, int end, typename Tuple, int multiplier>
struct ProductRange_<idx, end, Tuple, multiplier, typename platform::enable_if<idx == end>::type>
{
    static const int value = (idx == 0 ? multiplier : 1 ) * At<idx, Tuple>::value;
};

template<int i, int kMaxRank, typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup);

template<int i, int kMaxRank, typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup, platform::true_type)
{
    tup[i] = cutlass::contraction::At<i, Tup>::value;
    initTuple<i+1, kMaxRank, Tup>(tup);
}

template<int i, int kMaxRank, typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup, platform::false_type) {}

template<int i, int kMaxRank, typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup)
{
    initTuple<i, kMaxRank, Tup>(tup, platform::integral_constant<bool, i != kMaxRank>{});
}

/// initializes the runtime tuple `tup` from a compiletime IntTuple `Tup`
template<int kMaxRank, typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup)
{
    initTuple<0, kMaxRank <= Tup::kRank ? kMaxRank : Tup::kRank, Tup>(tup);
}

/// initializes the runtime tuple `tup` from a compiletime IntTuple `Tup`
template<typename Tup>
CUTLASS_HOST_DEVICE
void initTuple(int* tup)
{
    initTuple<0, Tup::kRank, Tup>(tup);
}

/// Runtime tuple
template<typename T> // IntTuple
class Tuple
{
    public:
    CUTLASS_HOST_DEVICE
    Tuple()
    {
        initTuple<T>(data);
    }

    CUTLASS_HOST_DEVICE
    int operator[](const int i) const
    {
        return data[i];
    }

    private:
    int data[T::kRank];
};

}
}
