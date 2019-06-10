package nn

import (
	"fmt"
	"go-mlp/matrix"
	"math"
	"math/rand"
)

// A Layer is a layer
type Layer struct {
	weights      matrix.Matrix
	dWeights     matrix.Matrix
	biases       matrix.Matrix
	dBiases      matrix.Matrix
	Z            matrix.Matrix
	Activations  matrix.Matrix
	dZ           matrix.Matrix
	dActivations matrix.Matrix
}

// NN is a nn
type NN struct {
	layers []Layer
}

//Print prints
func (n NN) Print() {
	for layer := range n.layers {
		fmt.Printf("Layer %v:\n", layer)
		fmt.Printf("	weights: %v \n", n.layers[layer].weights)
		fmt.Printf("	dWeights: %v \n", n.layers[layer].dWeights)
		fmt.Printf("	biases: %v \n", n.layers[layer].biases)
		fmt.Printf("	dBiases: %v \n", n.layers[layer].dBiases)
		fmt.Printf("	Z: %v \n", n.layers[layer].Z)
		fmt.Printf("	Activations: %v \n", n.layers[layer].Activations)
		fmt.Printf("	dZ: %v \n", n.layers[layer].dZ)
		fmt.Printf("	dActivations: %v \n", n.layers[layer].dActivations)
	}
}

// NewNN makes a nn
func NewNN(structure []int, inputs int) *NN {
	r := NN{}
	r.layers = make([]Layer, len(structure))
	for i := range structure {
		r.layers[i].biases = matrix.NewMatrix(1, structure[i], func() float64 { return 0.0 })
		s2 := append([]int{inputs}, structure...)
		r.layers[i].weights = matrix.NewMatrix(s2[i], structure[i], func() float64 { return genTest(s2[i], structure[i]) }) // Uses Xavier initialization

	}
	return &r
}

// genXav generates a weight based on Xavier initialization
func genXav(prev int, curr int) float64 {
	w := float64(prev) + (rand.Float64() * float64(curr-prev))
	return w * math.Sqrt(2/float64(prev))
}

// genTest makes the weights either 0.5 or 0.25, makes it easier to check backpropogation
func genTest(prev int, curr int) float64 {
	r := rand.Float64()
	if r >= 0.5 {
		return 0.25
	}
	return 0.5
}

// ResetActivations resets Activations and z and partials
func (n *NN) ResetActivations() {
	for layer := range (*n).layers {
		(*n).layers[layer].Activations = nil
		(*n).layers[layer].Z = nil
	}
}

// ResetPartials resets the partials
func (n *NN) ResetPartials() {
	for layer := range (*n).layers {
		(*n).layers[layer].dWeights = nil
		(*n).layers[layer].dBiases = nil
		(*n).layers[layer].dActivations = nil
		(*n).layers[layer].dZ = nil
	}
}

func (n *NN) Propogate(inputs []float64) {
	(*n).ResetActivations()
	(*n).layers[0].Z = matrix.Dot(matrix.Matrix{inputs}, (*n).layers[0].weights)
	(*n).layers[0].Activations = matrix.Matrix{ReLUL((*n).layers[0].Z[0])}
	for layer := 1; layer < len(n.layers); layer++ {
		(*n).layers[layer].Z = matrix.Dot(((*n).layers[layer-1].Activations), (*n).layers[layer].weights)
		(*n).layers[layer].Activations = matrix.Matrix{ReLUL((*n).layers[layer].Z[0])}
	}
}

func ReLUL(i []float64) []float64 {
	r := i
	for l := range r {
		r[l] = ReLU(i[l])
	}
	return r
}

func ReLU(i float64) float64 {
	return math.Max(0.0, i)
}

// Cost is Squared Error (After sigmoid))  Distance
func (n NN) GetCost(expected []float64) float64 {
	final := n.layers[len(n.layers)-1].Activations[0]
	r := 0.0
	for i := range expected {
		r += math.Pow((expected[i] - Sigmoid(final[i])), 2)
	}
	return r
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigDeriv(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func ReLUDeriv(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

func (n *NN) Backpropogate(expected []float64, inputs []float64) {
	(*n).ResetPartials()

	// Backpropogate the last layer first
	// Notes about how all of this was calculated in another document soon
	// probably
	cost := (*n).GetCost(expected)
	fmt.Println("Cost: ", cost)
	length := len((*n).layers)
	finalLayer := (*n).layers[length-1].Activations
	finalSigmoidedLayer := matrix.Apply(finalLayer, Sigmoid)
	finalSigmoidPrimeLayer := matrix.Apply(finalLayer, SigDeriv)
	(*n).layers[length-1].dActivations = matrix.MatMul(matrix.Sub(finalSigmoidedLayer, matrix.Matrix{expected}), finalSigmoidPrimeLayer).Multiply(2.0)
	(*n).layers[length-1].dZ = matrix.MatMul((*n).layers[length-1].dActivations, matrix.Apply((*n).layers[length-1].Activations, ReLUDeriv))
	(*n).layers[length-1].dBiases = (*n).layers[length-1].dZ
	(*n).layers[length-1].dWeights = DCDW((*n).layers[length-2].Activations, (*n).layers[length-1].dZ)
	// The last layer is done, now on to the "backpropogating"
	for lay := length - 2; lay >= 0; lay-- {
		// now for each layer do the backprop
		currentLayer := &(*n).layers[lay]
		nextLayer := &(*n).layers[lay+1]
		// first find the DCDA, or partial derivative of the cost with
		// respect to the layer's activations
		// while the activations arent changed during gradient descent,
		// its important to know the derivate for b a c k p r o p
		(*currentLayer).dActivations = DCDA((*nextLayer).weights, (*nextLayer).dZ)
		(*currentLayer).dZ = matrix.MatMul((*currentLayer).dActivations, matrix.Apply((*currentLayer).Activations, ReLUDeriv))
		(*currentLayer).dBiases = (*currentLayer).dZ
		// now for dWeights if its layer 0, you use inputs as
		// previous activations
		if lay == 0 {
			(*currentLayer).dWeights = DCDW(matrix.Matrix{inputs}, (*currentLayer).dZ)
		} else {
			(*currentLayer).dWeights = DCDW((*n).layers[lay-1].Activations, (*currentLayer).dZ)
		}
	}
}

// DCDW is the partial derivative of cost with respect to a set of weights, it
// depends on the previous activations, and the backpropd partials for DCDZ
func DCDW(previous matrix.Matrix, partials matrix.Matrix) matrix.Matrix {
	r := matrix.NewMatrix(previous.Columns(), partials.Columns(), matrix.Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = previous[0][row] * partials[0][column]
		}
	}
	return r
}

// DCDA is the partial derivative of the cost with respect fo the activations,
// it depends on weights its multiplied by and the partials of the
// activation*weights
func DCDA(weights matrix.Matrix, partials matrix.Matrix) matrix.Matrix {
	r := matrix.NewMatrix(1, weights.Rows(), matrix.Zeroes)
	for row := range r {
		for column := range r[row] {
			r[row][column] = matrix.ArrMult(partials[0], weights.GetRow(column))
		}
	}
	return r
}

// Train trains a neural network
func (n *NN) Train(inputs [][]float64, expecteds [][]float64, lr float64) {

	batchsize := len(inputs)
	if batchsize != len(expecteds) {
		panic("Error: length of inputs and outputs do not match")
	}

	avgCost := 0.0

	//Store the partials in an nn, add each time
	totalPartials := make([]Layer, len((*n).layers))
	for layer := range totalPartials {
		totalPartials[layer].dBiases = matrix.NewMatrix(1, len((*n).layers[layer].dBiases[0]), matrix.Zeroes)
		totalPartials[layer].dWeights = matrix.NewMatrix(len((*n).layers[layer].dBiases), len((*n).layers[layer].dBiases[0]), matrix.Zeroes)
	}
	for i := 0; i < batchsize; i++ {
		// reset
		(*n).ResetActivations()
		(*n).ResetPartials()
		// propogate
		(*n).Propogate(inputs[i])
		avgCost += (*n).GetCost(expecteds[i])
		(*n).Backpropogate(expecteds[i], inputs[i])
		// add the partials to the total
		for layer := range totalPartials {
			totalPartials[layer].dBiases = matrix.Add(totalPartials[layer].dBiases, (*n).layers[layer].dBiases)
			totalPartials[layer].dWeights = matrix.Add(totalPartials[layer].dWeights, (*n).layers[layer].dWeights)
		}
	}
	// average the partials(divide by batch size)
	for layer := range totalPartials {
		totalPartials[layer].dBiases = matrix.Apply(totalPartials[layer].dBiases, func(i float64) float64 {
			return i / float64(batchsize)
		})
		totalPartials[layer].dWeights = matrix.Apply(totalPartials[layer].dWeights, func(i float64) float64 {
			return i / float64(batchsize)
		})
	}
	avgCost /= float64(batchsize)

	// now adjust weights/biases based on dWeights/dBiases
	for layer := range (*n).layers {
		(*n).layers[layer].weights = matrix.Sub((*n).layers[layer].weights, totalPartials[layer].dWeights.Multiply(lr))
		(*n).layers[layer].dBiases = matrix.Sub((*n).layers[layer].biases, totalPartials[layer].dBiases.Multiply(lr))
	}
	fmt.Println("Avg cost: ", avgCost)
}
