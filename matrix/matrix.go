package matrix

// Matrix is a matrix
type Matrix [][]float64

// NewMatrix makes a new Matrix given rows and columns
func NewMatrix(rows int, columns int, generator func() float64) *Matrix {
	r := make(Matrix, rows)
	for row := range r {
		r[row] = make([]float64, columns)
		for column := range r[row] {
			r[row][column] = generator()
		}
	}
	return &r
}

// NewCopy is because go slices are actually pointers and weird
func NewCopy(m Matrix) Matrix {
	return *NewMatrix(m.Rows(), m.Columns(), Zeroes)
}

// Zeroes returns 0
func Zeroes() float64 {
	return 0.0
}

// Ones returns one
func Ones() float64 {
	return 1.0
}

// Rows returns the number of rows
func (m *Matrix) Rows() int {
	return len(*m)
}

// Columns returns the number of Columns
func (m *Matrix) Columns() int {
	return len((*m)[0])
}

// shape returns the shape
func (m *Matrix) shape() []int {
	return []int{(*m).Rows(), (*m).Columns()}
}

// Multiply multiplies a Matrix by a scalar
func (m Matrix) Multiply(f float64) Matrix {
	r := m
	for row := range m {
		for column := range (m)[row] {
			(r)[row][column] *= f
		}
	}
	return r
}

// zipMultiply multiplies the vals with the same index
func zipMultiply(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		panic("Invalid zipMultiply Operation")
	}
	r := 0.0
	for i := range a {
		r += a[i] * b[i]
	}
	return r
}

// getRow returns the row
func (m Matrix) getRow(index int) []float64 {
	return m[index]
}

// getColumn returns the column
func (m Matrix) getColumn(index int) []float64 {
	r := make([]float64, len(m))
	for i := range m {
		r[i] = m[i][index]
	}
	return r
}

// Dot takes the dot product
func (m Matrix) Dot(i Matrix) Matrix {
	if m.Columns() != i.Rows() {
		panic("Invalid dot product")
	}
	r := NewMatrix(m.Rows(), i.Columns(), func() float64 {
		return 0
	})
	for row := 0; row < m.Rows(); row++ {
		for column := 0; column < i.Columns(); column++ {
			(*r)[row][column] = zipMultiply(m.getRow(row), i.getColumn(column))
		}
	}
	return *r
}

// WeirdMult weirdly multiplies 2 matrices
func (m Matrix) WeirdMult(i Matrix) Matrix {
	if (m.Rows() != i.Rows()) || (m.Columns() != i.Columns()) {
		panic("bad dimensions")
	}
	r := NewMatrix(m.Rows(), m.Columns(), func() float64 {
		return 0
	})
	for row := range *r {
		for column := range (*r)[row] {
			(*r)[row][column] = m[row][column] * i[row][column]
		}
	}
	return *r
}

// Add adds 2 matrices
func (m Matrix) Add(i Matrix) Matrix {
	if (m.Rows() != i.Rows()) || (m.Columns() != i.Columns()) {
		panic("bad dimensions")
	}
	r := NewMatrix(m.Rows(), m.Columns(), func() float64 {
		return 0
	})
	for row := range *r {
		for column := range (*r)[row] {
			(*r)[row][column] = m[row][column] + i[row][column]
		}
	}
	return *r
}

// Sub tracts 2 matrices
func (m Matrix) Sub(i Matrix) Matrix {
	if (m.Rows() != i.Rows()) || (m.Columns() != i.Columns()) {
		panic("bad dimensions")
	}
	r := NewMatrix(m.Rows(), m.Columns(), func() float64 {
		return 0
	})
	for row := range *r {
		for column := range (*r)[row] {
			(*r)[row][column] = m[row][column] - i[row][column]
		}
	}
	return *r
}

// Apply applies a function to a matrix
func (m Matrix) Apply(f func(i float64) float64) Matrix {
	r := NewCopy(m)
	for row := range r {
		for column := range (r)[row] {
			(r)[row][column] = f(m[row][column])
		}
	}
	return r
}

//AtoM is an array of floats to a 1,len(i) matrix
func AtoM(i []float64) Matrix {
	return Matrix{
		i,
	}
}

// Broadcast is a function that does stuff below
func Broadcast(i Matrix, s int) Matrix {
	r := NewMatrix(len(i[0]), s, Zeroes)
	for row := range *r {
		for column := range (*r)[row] {
			(*r)[row][column] = i[0][row]
		}
	}
	return *r
}

/*
last layer = [[a, b, c]] shape = 1x3
weight = [[d, e]
	  [f, g]
	  [h, j]]
next =   [[ad + bf + ch, ae + bg + cj]]
the dZ would be [[k, l]]
purpose of broadcast is to  turn dW into
[
	[a, a]
	[b, b]
	[c, c]
]

and the dCost/dLastLayer = dZcurr/dLast * dCost/dZcurr
The dZcurr/dL is just
	[[d+e, f+g, h+j]]
*/
// DZdAPrev is the above, takes in the weights
func DZdAPrev(weights Matrix) Matrix {
	r := NewMatrix(1, len(weights), Zeroes)
	for row := range weights {
		for column := range weights[row] {
			(*r)[0][row] += weights[row][column]
		}
	}
	return *r
}

// ASum sums an array
func ASum(i []float64) float64 {
	s := 0.0
	for x := range i {
		s += i[x]
	}
	return s
}

/*
[
	[a * ad + bf + ch, a * ae + bg + cj]
	[b * ad + bf + ch, b * ae + bg + cj]
	[c * ad + bf + ch, c * ae + bg + cj]
]
*/
// Func DZdW(dZ matrix) just turns the dZdW into dCdW
func (m Matrix) DZdW(dZ Matrix) Matrix {
	r := NewCopy(m)
	for row := range r {
		for column := range (r)[row] {
			r[row][column] = m[row][column] * dZ[0][column]
		}
	}
	return r
}
