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

#include <cmath>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
  
struct {
  char const *comment;
  char const *name;
  double value_f64;
}
constant_definitions[] = {
  {"Returns 1, the multiplicative identity element", "one", double(1.0)},
  {"Returns 0, the additive identity element", "zero", double(0.0)},
  {"Returns 2", "two", double(2.0)},
  {"Returns pi, approximately 3.141", "pi", double(M_PI)},
  {"Returns 2 * pi", "two_pi", double(M_PI * 2.0)},
  {"Returns pi / 2", "half_pi", double(M_PI_2)},
  {"Returns sqrt(pi)", "root_pi", double(std::sqrt(M_PI))},
  {"Returns sqrt(pi / 2)", "root_half_pi", double(std::sqrt(M_PI_2))},
  {"Returns sqrt(2 * pi)", "root_two_pi", double(std::sqrt(M_PI * 2))},
  {"Returns sqrt(ln(4))", "root_ln_four", double(std::sqrt(std::log(4)))},
  {"Returns e, approximately 2.718...", "e", double(std::exp(1.0))},
  {"Returns (1/2)", "half", double(0.5)},
  {"Returns sqrt(2), approximately 1.414...", "root_two", double(std::sqrt(2.0))},
  {"Returns sqrt(2)/2, approximately 0.707...", "half_root_two", double(std::sqrt(2.0) * 0.5)},
  {"Returns ln(2), approximately 0.693...", "ln_two", double(std::log(2.0))},
  {"Returns ln(ln(2)), approximately -0.3665...", "ln_ln_two", double(std::log(std::log(2.0)))},
  {"Returns 1/3, approximately 0.333...", "third", double(1.0 / 3.0)},
  {"Returns 2/3, approximately 0.666...", "twothirds", double(2.0 / 3.0)},
  {"Returns pi - 3, approximately 0.1416...", "pi_minus_three", double(M_PI - 3.0)},
  {"Returns 4 - pi, approximately 0.858...", "four_minus_pi", double(4.0 - M_PI)},
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int size> struct bitset;

template <> struct bitset<8> {
  using type = uint64_t;
};

template <> struct bitset<4> {
  using type = uint32_t;
};

template <> struct bitset<2> {
  using type = uint16_t;
};

template <> struct bitset<1> {
  using type = uint8_t;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void emit_specialization(char const *type_name) {

  std::cout << "\n/////////////////////////////////////////////////////////////////////////////////////\n";

  std::cout << "\n// Specialization for " << type_name << "\n";

  for (auto const &def : constant_definitions) {

    std::cout << "\n/// " << def.comment << "  (specialization for " << type_name << ")\n"
      << "template <> CUTLASS_HOST_DEVICE " << type_name << " " << def.name << "<" << type_name << ">() {\n";

    // Type conversion from double
    T value = static_cast<T>(static_cast<T>(def.value_f64));

    using Bitset = typename bitset<sizeof(T)>::type;
    int const kSizeInBits = sizeof(Bitset) * 8;

    // Bitcast to 32-bit literal
    Bitset bits = reinterpret_cast<Bitset const &>(value);

    std::cout << "  uint" << kSizeInBits << "_t bits = 0x" << std::hex << bits << "u" << std::dec << ";\n";

    std::cout << "  return reinterpret_cast<" << type_name << " const &>(bits);\n";

    std::cout << "}\n";

    std::cout << "\n/// " << def.comment << "  (specialization for complex<" << type_name << ">)\n"
      << "template <> CUTLASS_HOST_DEVICE complex<" << type_name << "> " << def.name << "< complex<" << type_name << "> >() {\n";

    std::cout << "  return complex<" << type_name << ">(" << def.name << "<" << type_name << ">(), " << type_name << "());\n";

    std::cout << "}\n";
  }
}

template <>
void emit_specialization<double>(char const *) {
  std::cout << "\n/////////////////////////////////////////////////////////////////////////////////////\n";

  std::cout << "\n// Specialization for double\n";

  for (auto const &def : constant_definitions) {

    std::cout << "\n/// " << def.comment << "  (specialization for double)\n"
      << "template <> CUTLASS_HOST_DEVICE double " << def.name << "<double>() {\n";

    // Bitcast to 64-bit literal
    uint64_t bits = reinterpret_cast<uint64_t const &>(def.value_f64);

    std::cout << "  uint64_t bits = 0x" << std::hex << bits << "ull" << std::dec << ";\n";

    std::cout << "  return reinterpret_cast<double const &>(bits);\n";

    std::cout << "}\n";

    std::cout << "\n/// " << def.comment << "  (specialization for complex<double>)\n"
      << "template <> CUTLASS_HOST_DEVICE complex<double> " << def.name << "< complex<double> >() {\n";

    std::cout << "  return complex<double>(" << def.name << "<double>(), double());\n";

    std::cout << "}\n";
  }
}

int main() {

  using namespace cutlass;

  char const *license_text[] = {
    "/***************************************************************************************************",
    " * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.                  ",
    " * SPDX-License-Identifier: BSD-3-Clause                                                            ",
    " *                                                                                                  ",
    " * Redistribution and use in source and binary forms, with or without                               ",
    " * modification, are permitted provided that the following conditions are met:                      ",
    " *                                                                                                  ",
    " * 1. Redistributions of source code must retain the above copyright notice, this                   ",
    " * list of conditions and the following disclaimer.                                                 ",
    " *                                                                                                  ",
    " * 2. Redistributions in binary form must reproduce the above copyright notice,                     ",
    " * this list of conditions and the following disclaimer in the documentation                        ",
    " * and/or other materials provided with the distribution.                                           ",
    " *                                                                                                  ",
    " * 3. Neither the name of the copyright holder nor the names of its                                 ",
    " * contributors may be used to endorse or promote products derived from                             ",
    " * this software without specific prior written permission.                                         ",
    " *                                                                                                  ",
    " * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"                    ",
    " * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE                        ",
    " * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                   ",
    " * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE                     ",
    " * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL                       ",
    " * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR                       ",
    " * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER                       ",
    " * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,                    ",
    " * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE                    ",
    " * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                             ",
    " *                                                                                                  ",
    " **************************************************************************************************/",
  };

  // Emit license
  for (char const *line : license_text) {
    std::cout << line << "\n";
  }

  // Includes for the generator program
  std::cout << "\n/* \\file \n  \\brief Boost-style constant definitions for floating-point types.\n";
  std::cout << "*/\n\n";

  std::cout << "#pragma once\n\n";

  std::cout << "#include \"cutlass/cutlass.h\"\n";
  std::cout << "#include \"cutlass/numeric_types.h\"\n\n";
  std::cout << "#include \"cutlass/complex.h\"\n\n";

  std::cout << "///////////////////////////////////////////////////////////////////////////////////\n\n";
  std::cout << "namespace cutlass {\n";
  std::cout << "namespace constants {\n\n";
  std::cout << "///////////////////////////////////////////////////////////////////////////////////\n\n";
  
  std::cout << "//\n// Primary templates\n//\n\n";

  for (auto const &def : constant_definitions) {

    std::cout << "/// " << def.comment << "\n"
      << "template <typename T> CUTLASS_HOST_DEVICE T " << def.name << "();\n\n";
  }

  emit_specialization<double>("double"); 
  emit_specialization<float>("float"); 
  emit_specialization<cutlass::tfloat32_t>("tfloat32_t"); 
  emit_specialization<cutlass::half_t>("half_t"); 
  emit_specialization<cutlass::bfloat16_t>("bfloat16_t"); 

  std::cout << "///////////////////////////////////////////////////////////////////////////////////\n\n";
  std::cout << "} // namespace constants\n";
  std::cout << "} // namespace cutlass\n\n";
  std::cout << "///////////////////////////////////////////////////////////////////////////////////\n";
   
  return 0;
}

