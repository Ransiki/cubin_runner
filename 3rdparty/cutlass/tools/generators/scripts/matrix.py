#
# \file matrix.py
#
# \brief Procedural generator for matrix classes with value semantics. Covers all sizes up to 4x4
#        except 1x1.
#
# Usage:
#
#  $ cd tools/generators/scripts 
#
#  $ python matrix.py > ../../../include/cutlass/matrix.h
#

import re
import sys

#################################################################################################

matrix_class_template = """
/// ${Rows}-by-${Columns} matrix template class definition
template <typename Element_>
struct Matrix<Element_, ${Rows}, ${Columns}> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = ${Rows};

  /// Number of columns in matrix
  static int const kColumns = ${Columns};

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = ${Count};

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  ${Body}
};

/// Template alias for ${Rows}-by-${Columns} matrix
template <typename Element>
using Matrix${Rows}x${Columns} = Matrix<Element, ${Rows}, ${Columns}>;

${FreeConstructor}

/////////////////////////////////////////////////////////////////////////////////////////////////"""

#################################################################################################

#
def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

#################################################################################################

#
def EmitMatrixConstructorCopy(rows, columns):
  return SubstituteTemplate(
    """
  /// Copy constructor for a ${Rows}-by-${Columns} matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    """, {
      "Rows": str(rows),
      "Columns": str(columns)
    }
  )

#
def EmitMatrixConstructorElementsHelper(rows, columns):
  declarations = ""
  for row in range(rows):
    declarations += "    "
    for col in range(columns):
      declarations += "Element _%d_%d" % (row, col)
      if col + 1 < columns or row + 1 < rows:
        declarations += ", "
    if row + 1 < rows:
      declarations += "\n"

  ctor_assignments = ""
  free_assignments = ""
  for row in range(rows):
    ctor_assignments += "\n  "
    free_assignments += "\n  "
    for col in range(columns):
      idx = row * columns + col
      ctor_assignments += "  data[%d] = _%d_%d;" % (idx, row, col)
      comma = ", " if idx + 1 < (rows * columns) else " "
      free_assignments += "_%d_%d%s" % (row, col, comma)

  return (declarations, ctor_assignments, free_assignments)

#
def EmitMatrixConstructorElements(rows, columns):

  declarations, assignments, ignore = EmitMatrixConstructorElementsHelper(rows, columns)

  return SubstituteTemplate(
    """
  /// Constucts a ${Rows}-by-${Columns} matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
${ElementDeclarations}
  ) {
${ElementAssignments}
  }
    """, {
      "Rows": str(rows),
      "Columns": str(columns),
      "ElementDeclarations": declarations,
      "ElementAssignments": assignments
    }
  )

#
def EmitMatrixFreeConstructor(rows, columns):
  declarations, ignore, assignments = EmitMatrixConstructorElementsHelper(rows, columns)

  return SubstituteTemplate("""
/// Free funciton to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix${Rows}x${Columns}<Element> make_Matrix${Rows}x${Columns}(
${ElementDeclarations}
) {
  return Matrix${Rows}x${Columns}<Element>(${ElementAssignments}
  );
}
""", { 
    "Rows": str(rows), 
    "Columns": str(columns), 
    "ElementDeclarations": declarations, 
    "ElementAssignments": assignments 
  })

#
def EmitMatrixConstructorRows(rows, columns):
  declarations = ""
  assignments = ""

  for row in range(rows):
    if row:
      declarations += ",\n"
    declarations += "    Matrix<Element, 1, %d> const &row_%d" % (columns, row)

  for row in range(rows):
    for col in range(columns):
      idx = row * columns + col
      assignments += "\n    data[%d] = row_%d.data[%d];" % (idx, row, col)

  return SubstituteTemplate(
    """
  /// Constucts a ${Rows}-by-${Columns} matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
${ElementDeclarations}
  ) { ${ElementAssignments}
  }
    """, {
      "Rows": str(rows),
      "Columns": str(columns),
      "ElementDeclarations": declarations,
      "ElementAssignments": assignments
    }
  )

#
def EmitMatrixConstructorColumns(rows, columns):
  declarations = ""
  assignments = ""

  for col in range(columns):
    if col:
      declarations += ",\n"
    declarations += "    Matrix<Element, %d, 1> const &column_%d" % (columns, col)

  for row in range(rows):
    for col in range(columns):
      idx = row * columns + col
      assignments += "\n    result.data[%d] = column_%d.data[%d];" % (idx, col, row)

  return SubstituteTemplate(
    """
  /// Static method to construct a ${Rows}-by-${Columns} matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
${ElementDeclarations}
  ) { 
    Matrix result;
    ${ElementAssignments}
    return result;
  }
    """, {
      "Rows": str(rows),
      "Columns": str(columns),
      "ElementDeclarations": declarations,
      "ElementAssignments": assignments
    }
  )

#
def EmitMatrixConstructorIdentity(rows):
  body = ""
  for i in range(rows):
    idx = i * rows + i
    body += "\n    m.data[%d] = Element(1);" % idx
  return SubstituteTemplate(
    """
  /// Constructs an identity matrix
  CUTLASS_HOST_DEVICE
  static Matrix identity() {
    Matrix m;
    ${body}

    return m;
  }
    """, {"body": body})

#
def EmitMatrixConstructorUniform(rows, columns):

  body = ""
  for row in range(rows):
    for col in range(columns):
      idx = row * columns + col
      body += "\n    m.data[%d] = s;" % idx

  return SubstituteTemplate("""
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    ${body}

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  """, { "body": body})

#
def EmitMatrixConstructorDiagonal(rows, columns):
  ctor_body = ""
  get_diag_body = ""

  for i in range(rows):
    idx = i * rows + i
    ctor_body += "\n    m.data[%d] = diag.data[%d];" % (idx, i)
    get_diag_body += "\n    diag.data[%d] = data[%d];" % (i, idx)

  diagonal_count = min(rows, columns)

  return SubstituteTemplate(
    """
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, ${DiagonalCount}, 1> const &diag) {
    Matrix m;
    ${ctor_body}

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, ${DiagonalCount}> const &diag) {
    Matrix m;
    ${ctor_body}

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${DiagonalCount}, 1> diagonal() const {
    Matrix<Element, ${DiagonalCount}, 1> diag;
    ${get_diag_body}

    return diag;
  }
    """, {
    "DiagonalCount": str(diagonal_count),
    "ctor_body": ctor_body,
    "get_diag_body": get_diag_body
  })


#
def EmitMatrixConstructors(rows, columns):

  text = EmitMatrixConstructorCopy(rows, columns)

  text += EmitMatrixConstructorElements(rows, columns)

  if columns > 1 and rows > 1:
    text += EmitMatrixConstructorRows(rows, columns)

  if rows > 1 and columns > 1:
    text += EmitMatrixConstructorColumns(rows, columns)

  if rows == columns:
    text += EmitMatrixConstructorIdentity(rows)

  text += EmitMatrixConstructorUniform(rows, columns)

  if min(rows, columns) > 1:
    text += EmitMatrixConstructorDiagonal(rows, columns)

  return text

#################################################################################################

#
def EmitMatrixTranspose(rows, columns):

  body = ""
  for row in range(rows):
    for col in range(columns):
      dst_idx = col * rows + row
      src_idx = row * columns + col
      body += "\n    mt.data[%d] = data[%d];" % (dst_idx, src_idx)

  return SubstituteTemplate("""
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${Columns}, ${Rows}> transpose() const {
    Matrix<Element, ${Columns}, ${Rows}> mt;
    ${body}

    return mt;
  }
    """, {
    "Rows": str(rows),
    "Columns": str(columns),
    "body": body,
    })

#################################################################################################

#
def EmitMatrixAccessors(rows, columns):
  return SubstituteTemplate("""
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * ${Rows} + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * ${Rows} + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  """, {
    "Rows": str(rows)
  })

#################################################################################################

#
def EmitMatrixSliceAccessor(rows, columns, slice_rows, slice_cols):

  get_slice_body = ""
  set_slice_body = ""

  for row in range(slice_rows):
    for col in range(slice_cols):
      slice_idx = row * slice_cols + col
      idx = row * columns + col
      get_slice_body += "\n    m.data[%d] = data[i * %d + j + %d];" % (slice_idx, columns, idx)
      set_slice_body += "\n    data[i * %d + j + %d] = m.data[%d];" % (columns, idx, slice_idx)

  text = SubstituteTemplate(
    """
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${SliceRows}, ${SliceColumns}> slice_${SliceRows}x${SliceColumns}(int i = 0, int j = 0) const {
    Matrix<Element, ${SliceRows}, ${SliceColumns}> m;
    ${get_slice_body}

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_${SliceRows}x${SliceColumns}(Matrix<Element, ${SliceRows}, ${SliceColumns}> const &m, int i = 0, int j = 0) {
    ${set_slice_body}

    return *this;
  }
    """, {
      "SliceRows": str(slice_rows),
      "SliceColumns": str(slice_cols),
      "get_slice_body": get_slice_body,
      "set_slice_body": set_slice_body
    })

  if slice_rows == 1 and slice_cols == columns:
    text += SubstituteTemplate("""
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, ${Columns}> row(int i) const {
    return slice_1x${Columns}(i, 0);
  }

  Matrix &set_row(Matrix<Element, 1, ${Columns}> const &v, int i = 0) {
    return set_slice_1x${Columns}(v, i, 0);
  }
    """, { "Columns": str(columns)})
  elif slice_cols == 1 and slice_rows == rows:
    text += SubstituteTemplate("""
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${Rows}, 1> column(int j) const {
    return slice_${Rows}x1(0, j);
  }

  Matrix &set_column(Matrix<Element, ${Rows}, 1> const &v, int j =0) {
    return set_slice_${Rows}x1(v, 0, j);
  }
    """, { "Rows": str(rows)})

  return text

#
def EmitMatrixSliceAccessors(rows, columns):
  text = ""
  for slice_rows in range(1, rows + 1):
    for slice_cols in range(1, columns + 1):
      if (slice_rows > 1 or slice_cols > 1):
        text += EmitMatrixSliceAccessor(rows, columns, slice_rows, slice_cols)
  return text

#################################################################################################

#
def EmitMatrixHcat(rows, columns, left_cols):
  body = ""

  if left_cols == 1 and rows == 1:
    left_operand = "Element"
    left_type = "an Element"
  else:
    left_operand = SubstituteTemplate("Matrix<Element, ${Rows}, ${Columns}> const &", {
      "Rows": str(rows), "Columns": str(left_cols) })
    left_type = "a %d-by-%d matrix" % (rows, left_cols)

  right_cols  = columns - left_cols
  if right_cols == 1 and rows == 1:
    right_operand  = "Element"
    right_type = "an Element"
  else:
    right_operand = SubstituteTemplate("Matrix<Element, ${Rows}, ${Columns}> const &", {
      "Rows": str(rows), "Columns": str(right_cols) })
    right_type = "a %d-by-%d matrix" % (rows, columns - left_cols)

  body = ""
  idx = 0
  for i in range(rows):
    body += "\n      "
    for j in range(columns):
      if idx > 0:
        body += ", "
      idx += 1
      if j < left_cols:
        if rows == 1 and left_cols == 1:
          at = "lhs"
        else:
          at = "lhs.at(%d, %d)" % (i, j)
      else:
        if rows == 1 and (right_cols == 1):
          at = "rhs"
        else:
          at = "rhs.at(%d, %d)" % (i, j - left_cols)
      body += at

  return SubstituteTemplate("""
  /// Forms a ${Rows}-by-${Columns} matrix by horizontally concatenating ${LeftType} with ${RightType}
  CUTLASS_HOST_DEVICE
  static Matrix hcat(${LeftOperand} lhs, ${RightOperand} rhs) {
    return Matrix(${body});
  }
  """, { 
    "LeftOperand": left_operand, 
    "LeftType": left_type,
    "RightOperand": right_operand, 
    "RightType": right_type, 
    "body": body})

#
def EmitMatrixHcatMethod(rows, columns, right_columns):
  if columns + right_columns <= 4:
    total_columns = columns + right_columns
    if right_columns == 1 and rows == 1:
      right_type = "an Element"
      right_operand = "Element"
    else:
      right_type = "a %d-by-%d matrix" % (rows, right_columns)
      right_operand = "Matrix<Element, %d, %d> const &" % (rows, right_columns)
    return SubstituteTemplate("""
  /// Concatenates this matrix with a ${RightType} to form a ${Rows}-by-${TotalColumns} matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${Rows}, ${TotalColumns}> hcat(${RightOperand} rhs) const {
    return Matrix<Element, ${Rows}, ${TotalColumns}>::hcat(*this, rhs);
  }
    """, { 
    "Rows": str(rows), 
    "Columns": str(columns), 
    "RightColumns": str(right_columns), 
    "RightType": right_type,
    "RightOperand": right_operand,
    "TotalColumns": str(total_columns) 
  })
  else:
    return ""

#
def EmitMatrixVcat(rows, columns, upper_rows):
  body = ""

  if upper_rows == 1 and columns == 1:
    upper_operand = "Element"
    upper_type = "an Element"
  else:
    upper_operand = SubstituteTemplate("Matrix<Element, ${Rows}, ${Columns}> const &", {
      "Rows": str(upper_rows), "Columns": str(columns) })
    upper_type = "a %d-by-%d matrix" % (upper_rows, columns)

  lower_rows  = rows - upper_rows
  if lower_rows == 1 and columns == 1:
    lower_operand  = "Element"
    lower_type = "an Element"
  else:
    lower_operand = SubstituteTemplate("Matrix<Element, ${Rows}, ${Columns}> const &", {
      "Rows": str(lower_rows), "Columns": str(columns) })
    lower_type = "a %d-by-%d matrix" % (lower_rows, columns)

  body = ""
  idx = 0
  for i in range(rows):
    body += "\n      "
    for j in range(columns):
      if idx > 0:
        body += ", "
      idx += 1
      if i < upper_rows:
        if upper_rows == 1 and columns == 1:
          at = "upper"
        else:
          at = "upper.at(%d, %d)" % (i, j)
      else:
        if lower_rows == 1 and columns == 1:
          at = "lower"
        else:
          at = "lower.at(%d, %d)" % (i - upper_rows, j)
      body += at

  return SubstituteTemplate("""
  /// Forms a ${Rows}-by-${Columns} matrix by vertically concatenating ${LeftType} with ${RightType}
  CUTLASS_HOST_DEVICE
  static Matrix vcat(${LeftOperand} upper, ${RightOperand} lower) {
    return Matrix(${body});
  }
  """, { 
    "LeftOperand": upper_operand, 
    "LeftType": upper_type,
    "RightOperand": lower_operand, 
    "RightType": lower_type, 
    "body": body})

#
def EmitMatrixVcatMethod(rows, columns, lower_rows):
  total_rows = rows + lower_rows
  if total_rows <= 4:
    if lower_rows == 1 and columns == 1:
      lower_type = "an Element"
      lower_operand = "Element"
    else:
      lower_type = "a %d-by-%d matrix" % (lower_rows, columns)
      lower_operand = "Matrix<Element, %d, %d> const &" % (lower_rows, columns)
    return SubstituteTemplate("""
  /// Concatenates this matrix with a ${LowerType} to form a ${TotalRows}-by-${Columns} matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${TotalRows}, ${Columns}> vcat(${LowerOperand} rhs) const {
    return Matrix<Element, ${TotalRows}, ${Columns}>::vcat(*this, rhs);
  }
    """, { 
    "Rows": str(rows), 
    "Columns": str(columns), 
    "LowerRows": str(lower_rows), 
    "LowerType": lower_type,
    "LowerOperand": lower_operand,
    "TotalRows": str(total_rows) 
  })
  else:
    return ""

#
def EmitMatrixBlockConstructor(rows, columns, mi, mj):

  def get_matrix_type(block_rows, block_columns):
    if block_rows == 1 and block_columns == 1:
      return "Element                        "
    return   "Matrix<Element, %d, %d> const &" % (block_rows, block_columns)

  def get_matrix_element(mat, block_rows, block_columns, i, j):
    text = mat
    if block_rows > 1 or block_columns > 1:
      text += ".at(%d, %d)" % (i, j)
    return text 

  right_columns = columns - mj
  lower_rows = rows - mi

  upper_left = get_matrix_type(mi, mj)
  upper_right = get_matrix_type(mi, right_columns)
  lower_left = get_matrix_type(lower_rows, mj)
  lower_right = get_matrix_type(lower_rows, right_columns)

  body = ""
  for i in range(rows):
    body += "\n      "
    for j in range(columns):
      if i * columns + j > 0:
        body += ", "
      if i < mi:
        if j < mj:
          body += get_matrix_element("A", mi, mj, i, j)
        else:
          body += get_matrix_element("B", mi, right_columns, i, j - mj)
      else:
        if j < mj:
          body += get_matrix_element("C", lower_rows, mj, i - mi, j)
        else:
          body += get_matrix_element("D", lower_rows, right_columns, i - mi, j - mj)

  return SubstituteTemplate("""
  /// Forms a ${Rows}-by-${Columns} matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    ${UpperLeftOperand} A, ${UpperRightOperand} B,
    ${LowerLeftOperand} C, ${LowerRightOperand} D) {
    return Matrix(${body}
    );
  }
  """, {
    "Rows": str(rows),
    "Columns": str(columns),
    "UpperLeftOperand": upper_left,
    "UpperRightOperand": upper_right,
    "LowerLeftOperand": lower_left,
    "LowerRightOperand": lower_right,
    "body" : body
  })

#
def EmitMatrixBlockConstructors(rows, columns):
  text = ""
  for j in range(1, columns):
    text += EmitMatrixHcat(rows, columns, j)

  for j in range(1, 5):
    text += EmitMatrixHcatMethod(rows, columns, j)

  for i in range(1, rows):
    text += EmitMatrixVcat(rows, columns, i)

  for i in range(1, 5):
    text += EmitMatrixVcatMethod(rows, columns, i)

  for i in range(1, rows):
    for j in range(1, columns):
      text += EmitMatrixBlockConstructor(rows, columns, i, j)

  return text

#################################################################################################

#
def EmitMatrixMethodsElementwiseUnary(rows, columns):
  body = ""
  for row in range(rows):
    for col in range(columns):
      idx = row * columns + col
      body += "\n    m.data[%d] = -m.data[%d];" % (idx, idx)
  
  return SubstituteTemplate("""
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    ${body}

    return m;
  }
  """, { "body": body })

#
def EmitMatrixMethodsElementwiseBinary(rows, columns):

  # (name, C++ operator overload, has_operator_overload, has_scalar_version)
  operations = [
    ('add', '+', True, False),
    ('subtract', '-', True, False),
    ('multiply', '*', False, True),
    ('divide', '/', True, True),
  ]

  text = ""

  for op_name, operator, has_operator_overload, has_scalar_version in operations:

    const_body = ""
    assign_body = ""
    scalar_body = ""
    scalar_assign = ""

    for row in range(rows):
      const_body += "\n"
      assign_body += "\n"
      scalar_body += "\n"
      scalar_assign += "\n"
      for col in range(columns):
        idx = row * columns + col
        const_body  += "    result.data[%d] = data[%d] %s rhs.data[%d];\n" % (idx, idx, operator, idx)
        assign_body += "    data[%d] %s= rhs.data[%d];\n" % (idx, operator, idx)
        scalar_body += "    result.data[%d] = data[%d] %s s;\n" % (idx, idx, operator)
        scalar_assign += "    data[%d] %s= s;\n" % (idx, operator)

    text += SubstituteTemplate(
      """
  /// Elementwise ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix ${OpName}(Matrix const &rhs) const {

    Matrix result;
    ${body}
    return result;
  }
      """, {
        "OpName": op_name,
        "operator": operator,
        "body": const_body,
        "assign_body": assign_body,
        "Rows": str(rows),
        "Columns": str(columns),
      })

    if has_scalar_version:
      text += SubstituteTemplate(
        """
  /// Scalar ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix ${OpName}(Element const &s) const {

    Matrix result;
    ${body}
    return result;
  }

  /// Scalar ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix operator ${operator}(Element const &s) const {
    return ${OpName}(s);
  }

  /// Scalar ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix & operator ${operator}=(Element const &s) {
    ${scalar_assign}
    return *this;
  }
        """, {
          "OpName": op_name,
          "operator": operator,
          "body": scalar_body,
          "scalar_assign": scalar_assign,
          "Rows": str(rows),
          "Columns": str(columns),
        })

    if has_operator_overload:
      text += SubstituteTemplate(
        """
  /// Elementwise ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix operator ${operator}(Matrix const &rhs) const {
    return ${OpName}(rhs);
  }

  /// Elementwise ${OpName} operator (${Rows}-by-${Columns})
  CUTLASS_HOST_DEVICE
  Matrix & operator ${operator}=(Matrix const &rhs) {
    ${assign_body}
    return *this;
  }
        """, {
          "OpName": op_name,
          "operator": operator,
          "body": const_body,
          "assign_body": assign_body,
          "Rows": str(rows),
          "Columns": str(columns)
        })

  return text

#
def EmitMatrixProductGeneral(rows, columns, rhs_columns):

  body = ""
  for k in range(columns):
    body += "\n    // k=%d\n" % k
    for i in range(rows):
      for j in range(rhs_columns):
        result_idx = i * rhs_columns + j
        lhs_idx = i * columns + k
        rhs_idx = k * rhs_columns + j
        body += "    accum.data[%d] += data[%d] * rhs.data[%d];\n" % (result_idx, lhs_idx, rhs_idx)

  text = SubstituteTemplate("""
  /// Matrix product of size ${Rows}-by-${RHSColumns}-by-${Columns}
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${Rows}, ${RHSColumns}> product(
    Matrix<Element, ${Columns}, ${RHSColumns}> const &rhs,
    Matrix<Element, ${Rows}, ${RHSColumns}> accum = Matrix<Element, ${Rows}, ${RHSColumns}>()
  ) const {
    ${body}
    return accum;
  }

  /// Matrix product of size ${Rows}-by-${RHSColumns}-by-${Columns}
  CUTLASS_HOST_DEVICE
  Matrix<Element, ${Rows}, ${RHSColumns}> operator*(Matrix<Element, ${Columns}, ${RHSColumns}> const &rhs) const {
    return product(rhs);
  }
  """, {
    "Rows": str(rows),
    "Columns": str(columns),
    "RHSColumns": str(rhs_columns),
    "body": body
  })

  if columns == rhs_columns:
    text += SubstituteTemplate("""
  /// Matrix product of size ${Rows}-by-${RHSColumns}-by-${Columns}
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, ${Columns}, ${RHSColumns}> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    """, {
      "Rows": str(rows),
      "Columns": str(columns),
      "RHSColumns": str(rhs_columns),
    })

  return text

#
def EmitMatrixProductDot(rows, columns, rhs_columns):

  body = ""
  for k in range(columns):
    body += "\n    // k=%d\n" % k
    for i in range(rows):
      for j in range(rhs_columns):
        lhs_idx = i * columns + k
        rhs_idx = k * rhs_columns + j
        body += "    accum += data[%d] * rhs.data[%d];\n" % (lhs_idx, rhs_idx)

  return SubstituteTemplate("""
  /// Matrix product of size ${Rows}-by-${RHSColumns}-by-${Columns}
  CUTLASS_HOST_DEVICE
  Element product(Matrix<Element, ${Columns}, ${RHSColumns}> const &rhs, Element accum = Element()) const {
    ${body}
    return accum;
  }

  /// Matrix product of size ${Rows}-by-${RHSColumns}-by-${Columns}
  CUTLASS_HOST_DEVICE
  Element operator*(Matrix<Element, ${Columns}, ${RHSColumns}> const &rhs) const {
    return product(rhs);
  }
  """, {
    "Rows": str(rows),
    "Columns": str(columns),
    "RHSColumns": str(rhs_columns),
    "body": body
  })

#
def EmitMatrixDot(rows, columns):
  extent = max(rows, columns)

  body = ""
  for idx in range(extent):
    body += "\n    accum += data[%d] * rhs.data[%d];" % (idx, idx)

  return SubstituteTemplate("""
  /// Dot product of vectors with extent ${Extent}
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, ${Extent}, 1> const &rhs, Element accum = Element()) const {
    ${body}
    return accum;
  }

  /// Dot product of vectors with extent ${Extent}
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, ${Extent}> const &rhs, Element accum = Element()) const {
    ${body}
    return accum;
  }
  """, {
    "Extent": str(extent),
    "body": body
  })


#
def EmitMatrixProducts(rows, columns):
  text = ""
  for rhs_columns in range(1, 5):
    if rows == 1 and rhs_columns == 1:
      text += EmitMatrixProductDot(rows, columns, rhs_columns)
    else:
      text += EmitMatrixProductGeneral(rows, columns, rhs_columns)

  if rows == 1 or columns == 1:
    text += EmitMatrixDot(rows, columns)

  return text

#################################################################################################

#
def EmitMatrixNorm(rows, columns):
  sum_body = ""
  norm_body = ""
  trace_body = ""
  for row in range(rows):
    for col in range(columns):
      idx = row * columns + col
      sum_body  += "\n    accum += data[%d];" % (idx)
      norm_body += "\n    accum += data[%d] * data[%d];" % (idx, idx)
      if row == col:
        trace_body += "\n    accum += data[%d];" % (idx)

  return SubstituteTemplate("""
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    ${SumBody}

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    ${NormBody}

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    ${TraceBody}

    return accum;
  }
    """, 
  { 
    "SumBody": sum_body, 
    "NormBody": norm_body,
    "TraceBody": trace_body
  })

#################################################################################################

#
def EmitMatrixRotations(rows, columns):
  if rows != columns:
    return ""
  if rows == 2:
    return """
  /// Returns 2-by-2 rotation matrix
  CUTLASS_HOST_DEVICE
  static Matrix rotation(Element theta) {
    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    return Matrix(
      c, -s,
      s,  c
    );
  }
    """
  if rows in (3, 4):
    return SubstituteTemplate("""
  /// Returns ${Rows}-by-${Columns} rotation matrix around the X axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_X(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(1, 1) = c;
    m.at(1, 2) = -s;
    m.at(2, 1) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns ${Rows}-by-${Columns} rotation matrix around the Y axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Y(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(2, 0) = -s;
    m.at(0, 2) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns ${Rows}-by-${Columns} rotation matrix around the Z axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Z(Element theta) {
    Matrix m = Matrix::identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(0, 1) = -s;
    m.at(1, 0) = s;
    m.at(1, 1) = c;

    return m;
  }

  /// Returns a ${Rows}-by-${Columns} rotation matrix around a unit-length axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation(Element theta, Matrix<Element, 3, 1> const &u) {
    Element x = u.data[0];
    Element y = u.data[1];
    Element z = u.data[2];

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    Element one_minus_cos = Element(1) - fast_cos(theta);

    Matrix m;

    m.set_slice_3x3({
      c + x * x * one_minus_cos, x * y * one_minus_cos - z * s, x * z * one_minus_cos + y * s,
      y * x * one_minus_cos * z * s, c + y * y * one_minus_cos, y * z * one_minus_cos - x * s,
      z * x * one_minus_cos - y * s, z * y * one_minus_cos + x * s, c + z * z * one_minus_cos
    });

    return m;
  }

  /// Returns a ${Rows}-by-${Columns} reflection about the plane specified by the 
  /// unit-length normal vector n_unit
  CUTLASS_HOST_DEVICE
  static Matrix reflection(Matrix<Element, 3, 1> const &n_unit) {

    Element a = n_unit.data[0];
    Element b = n_unit.data[1];
    Element c = n_unit.data[2];

    Matrix m = Matrix::identity();

    m.set_slice_3x3({
      Element(1) - Element(2) * a * a, Element(-2) * a * b, Element(-2) * a * c,
      Element(-2) * a * b, Element(1) - Element(2) * b * b, Element(-2) * b * c,
      Element(-2) * a * c, Element(-2) * b * c, Element(1) - Element(2) * c * c
    });

    return m;
  }
""", { "Rows": str(rows), "Columns": str(columns) } )

#
def EmitMatrixTransforms(rows, columns):
  if rows != 4 or columns != 4:
    return ""
  return """
  /// Returns a perspective projection matrix typical of OpenGL applications
  CUTLASS_HOST_DEVICE
  static Matrix perspective(Element near_plane, Element far_plane, Element fovH, Element fovV) {
    Element aspect = fovH / fovV;
    Element f = Element(cos(fovV)) / Element(fovH);
    Element Q = near_plane - far_plane;

    return Matrix(
      f / aspect, 0,                0,                           0,
      0,          f,                0,                           0,
      0,          0, (near_plane + far_plane) / Q, Element(2) * far_plane * near_plane / Q,
      0,          0,                -1,                          0
    );
  }

  CUTLASS_HOST_DEVICE
  static Matrix translation(Matrix<Element, 3, 1> const &v) {
    return Matrix(
      1, 0, 0, v.data[0],
      0, 1, 0, v.data[1],
      0, 0, 1, v.data[2],
      0, 0, 0, 1
    );
  }
  """  

#################################################################################################

#
def EmitMatrixCrossProduct(rows, columns):
  if (rows == 1 and columns == 3) or (rows == 3 and columns == 1):
    return """
  /// Cross product
  CUTLASS_HOST_DEVICE
  Matrix cross(Matrix const &rhs) const {
    return Matrix(
      data[1] * rhs.data[2] - data[2] * rhs.data[1],
      data[0] * rhs.data[2] - data[2] * rhs.data[1],
      data[0] * rhs.data[1] - data[1] * rhs.data[0]
    );
  }
  """
  return ""

#################################################################################################

#
def EmitMatrixDeterminant(rows, columns):
  if rows != columns:
    return ""

  if rows == 2:
    body = "    accum += data[0] * data[3] - data[1] * data[2];\n"
  else:
    body = "\n"
    for j in range(columns):
      idx = 0
      submatrix_body = "{ "
      for i in range(rows - 1):
        for k in range(columns):
          if k != j:
            if idx > 0:
              submatrix_body += ', '
            submatrix_body += "at(%d, %d)" % (i + 1, k)
            idx += 1
      submatrix_body += " }"
      sgn = "-" if (j % 2) else "+"
      body += "    accum %s= at(0, %d) * Matrix<Element, %d, %d>(%s).determinant();\n" % (sgn, j, rows - 1, rows - 1, submatrix_body)

  return SubstituteTemplate("""
  /// Computes the determinant of a ${Rows}-by-${Columns} matrix
  CUTLASS_HOST_DEVICE
  Element determinant(Element accum = Element()) const {
    ${body}
    return accum;
  }
  """, { "body": body, "Rows": str(rows), "Columns": str(columns)})

#
def EmitMatrixInverse(rows, columns):
  if rows == 2 and columns == 2:
    return """
  /// Computes the inverse of a 2-by-2 matrix given
  /// the matrix's determinant
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element det) const {
    return Matrix(
      data[3], -data[1],
      -data[2], data[0]
    ) * (Element(1) / det); 
  }

  /// Computes the inverse of a 2-by-2 matrix.
  CUTLASS_HOST_DEVICE
  Matrix inverse() const {
    return inverse(determinant());
  }
    """

  if rows == 3 and columns == 3:
    return """
  /// Computes the inverse of a 3-by-3 matrix given
  /// the matrix's determinant
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element det) const {
    return Matrix(
      at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1),
      at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2),
      at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1),

      at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2),
      at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0),
      at(0, 2) * at(1, 0) - at(0, 0) * at(1, 2),

      at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0),
      at(0, 1) * at(2, 0) - at(0, 0) * at(2, 1),
      at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)
    ) * (Element(1) / det);
  }
  /// Computes the inverse of a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix inverse() const {
    return inverse(determinant());
  }
    """

  if rows == 4 and columns == 4:
    return """
  /// Computes the inverse of a 4-by-4 matrix (ignores the optional argument)
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element ignore = 1) const {
    Matrix<Element, 2, 2> B = slice_2x2(0, 2);
    Matrix<Element, 2, 2> A = slice_2x2(0, 0);
    Matrix<Element, 2, 2> C = slice_2x2(2, 0);
    Matrix<Element, 2, 2> D = slice_2x2(2, 2);

    Matrix<Element, 2, 2> D_inv = D.inverse();

    Matrix<Element, 2, 2> E = (A - B * D_inv * C).inverse();

    return Matrix::block(
      E,              -E * B * D_inv,
      -D_inv * C * E, D_inv + D_inv * C * E * B * D_inv
    );
  }
    """
  return ""

#################################################################################################

#
def EmitMatrix(stream, rows, columns):
  print("EmitMatrix(%d, %d)" % (rows, columns), file=sys.stderr)

  body = EmitMatrixConstructors(rows, columns)

  body += EmitMatrixTranspose(rows, columns)

  body += EmitMatrixAccessors(rows, columns)

  body += EmitMatrixSliceAccessors(rows, columns)

  body += EmitMatrixBlockConstructors(rows, columns)

  body += EmitMatrixMethodsElementwiseBinary(rows, columns)

  body += EmitMatrixMethodsElementwiseUnary(rows, columns)

  body += EmitMatrixProducts(rows, columns)

  body += EmitMatrixNorm(rows, columns)

  body += EmitMatrixRotations(rows, columns)

  body += EmitMatrixTransforms(rows, columns)

  body += EmitMatrixCrossProduct(rows, columns)

  body += EmitMatrixDeterminant(rows, columns)

  body += EmitMatrixInverse(rows, columns)

  free_constructor = EmitMatrixFreeConstructor(rows, columns)

  print(SubstituteTemplate(matrix_class_template, {
    "Rows": str(rows),
    "Columns": str(columns),
    "Count": str(rows * columns),
    "Body": body,
    "FreeConstructor": free_constructor
    }),
    file = stream
  )
  pass

#################################################################################################

#
def EmitFilePrologue(stream):
  print(
"""/***************************************************************************************************
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
/*  
  \\file
  \\brief Matrix classes with value semantics.
*/

#pragma once

#include <iosfwd>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Primary template with partial specializations to follow
template <typename Element, int Rows, int Columns> struct Matrix;

/////////////////////////////////////////////////////////////////////////////////////////////////""",
  file = stream)

#
def EmitFileEpilogue(stream):
  print("""
/// Elementwise scalar multiplication
template <typename Element, int Rows, int Columns>
CUTLASS_HOST_DEVICE
Matrix<Element, Rows, Columns> operator*(Element s, Matrix<Element, Rows, Columns> const &rhs) {
  return rhs.multiply(s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////""",
  file = stream)

#################################################################################################

#
def EmitRank(stream, rank):
  for outer in range(1, 5):
    for inner in range(1, rank + 1):
      if outer == 1 and inner == 1:
        continue
      EmitMatrix(stream, inner, outer)

#
def Main():
  stream  = sys.stdout
  EmitFilePrologue(stream)
  for outer in range(1, 5):
    for inner in range(1, 5):
      if outer == 1 and inner == 1:
        continue
      EmitMatrix(stream, outer, inner)
  EmitFileEpilogue(stream)

#################################################################################################

#
if __name__ == "__main__":
  sys.exit(Main())

#################################################################################################
