package matrix

// Matrix represents a two-dimensional array
type Matrix [][]float64

// NewMatrix creates a new matrix
func NewMatrix(rows int, columns int, generator func() float64) Matrix {
	r := make(Matrix, rows)
	for row := range r {
		r[row] = make([]float64, columns)
		for column := range r[row] {
			r[row][column] = generator()
		}
	}
	return r
}

// Zeroes returns a float 0.0
func Zeroes() float64 {
	return 0.0
}

// Ones returns a float 1.0
func Ones() float64 {
	return 1.0
}

// Rows returns the number of rows
func (m Matrix) Rows() int {
	return len(m)
}

// Columns returns the number of columns
func (m Matrix) Columns() int {
	return len(m[0])
}

// Shape returns a 2-d array of the dimensions
func (m Matrix) Shape() [2]int {
	return [2]int{m.Rows(), m.Columns()}
}

// Multiply multiplies a matrix by a scalar
func (m Matrix) Multiply(f float64) Matrix {
	r := m
	for row := range m {
		for column := range m[row] {
			r[row][column] *= f
		}
	}
	return r
}

// ArrMult multiplies the values with the same index and then ads it up
func ArrMult(a []float64, b []float64) float64 {
	r := 0.0
	for i := range a {
		r += a[i] * b[i]
	}
	return r
}

// ArrSum sums 2 arrays and returns it
func ArrSum(a []float64, b[]float64) []float64{
	r := make([]float64, len(a))
	for i := range r{
		r[i] = a[i] + b[i]
	}
	return r
}

// GetRow returns a row from a matrix
func (m Matrix) GetRow(idx int) []float64 {
	return m[idx]
}

// GetColumns returns a column from a matrix
func (m Matrix) GetColumn(idx int) []float64 {
	r := make([]float64, len(m))
	for i := range m {
		r[i] = m[i][idx]
	}
	return r
}

// Dot returns the dot product of two matrices
func Dot(a Matrix, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), b.Columns(), Zeroes)

	for row := 0; row < a.Rows(); row++ {
		for column := 0; column < b.Columns(); column++ {
			r[row][column] = ArrMult(a.GetRow(row), b.GetColumn(column))
		}
	}
	return r
}

// MatMul multiplies two matrices of the same size
func MatMul(a Matrix, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), b.Columns(), Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = a[row][column] * b[row][column]
		}
	}

	return r
}

// Add adds two matrices
func Add(a Matrix, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), b.Columns(), Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = a[row][column] + b[row][column]
		}
	}

	return r
}

// Sub subtracts two matrices
func Sub(a Matrix, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), b.Columns(), Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = a[row][column] - b[row][column]
		}
	}

	return r
}

// Apply applies a function to a matrix
func Apply(m Matrix, f func(i float64) float64) Matrix {
	r := NewMatrix(m.Rows(), m.Columns(), Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = f(m[row][column])
		}
	}
	return r
}
