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
		r.layers[i].biases = *matrix.NewMatrix(1, structure[i], func() float64 { return 0.01 })
		s2 := append([]int{inputs}, structure...)
		r.layers[i].weights = *matrix.NewMatrix(s2[i], structure[i], func() float64 { return genXav(s2[i], structure[i]) }) // Uses Xavier initialization

	}
	return &r
}

// genXav generates a weight based on Xavier initialization
func genXav(prev int, curr int) float64 {
	w := float64(prev) + (rand.Float64() * float64(curr-prev))
	return w * math.Sqrt(2/float64(prev))
}

func genBasic(prev int, curr int) float64 {
	return 1.1
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

// Propogate propogates the neural net by multiplying activations of prev layer by the weights then adding bias and then ReLUing it
func (n *NN) Propogate(inputs []float64) {
	(*n).ResetActivations()
	(*n).layers[0].Z = matrix.Matrix{inputs}.Dot((*n).layers[0].weights)
	(*n).layers[0].Activations = matrix.Matrix{reLUL((*n).layers[0].Z[0])}
	for layer := 1; layer < len(n.layers); layer++ {
		(*n).layers[layer].Z = ((*n).layers[layer-1].Activations).Dot((*n).layers[layer].weights)
		(*n).layers[layer].Activations = matrix.Matrix{reLUL((*n).layers[layer].Z[0])}
	}
}

// reLUL applies reLU to an array of floats
func reLUL(i []float64) []float64 {
	r := i
	for l := range r {
		r[l] = reLU(i[l])
	}
	return r
}

func reLU(i float64) float64 {
	return math.Max(0.0, i)
}

// GetCost returns the cost of a weird L2 distance
func (n NN) GetCost(expected []float64) float64 {
	lenght := len(n.layers)
	finallayer := n.layers[lenght-1].Activations[0]
	r := 0.0
	for i := range expected {
		r += math.Pow(Sigmoid(finallayer[i])-expected[i], 2)
	}
	return r
}

//Sigmoid takes the sigmoid
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

//SigDeriv returns deriv of sig of x
func SigDeriv(x float64) float64 {
	return Sigmoid(x) * (1.0 - Sigmoid(x))
}

//Backpropogate fills the partial derivatives and returns the loss
func (n *NN) Backpropogate(inputs []float64, expected []float64) float64 {
	(*n).ResetPartials()
	cost := (*n).GetCost(expected)
	fmt.Println(cost)
	//Do the last layer partial derivatives
	//Create dActi and dZ
	lenght := len(n.layers)
	expectedMatrix := matrix.AtoM(expected)
	expectedSigmoidMatrix := expectedMatrix.Apply(Sigmoid)
	(*n).layers[lenght-1].dActivations = ((*n).layers[lenght-1].Activations.Sub(expectedSigmoidMatrix)).Multiply(2.0).WeirdMult(expectedMatrix.Apply(SigDeriv))
	(*n).layers[lenght-1].dZ = ((n).layers[lenght-1].Z.Apply(DReLU)).WeirdMult((*n).layers[lenght-1].dActivations)
	(*n).layers[lenght-1].dBiases = (*n).layers[lenght-1].dZ
	if lenght != 1 {
		(*n).layers[lenght-1].dWeights = matrix.Broadcast((*n).layers[lenght-2].Activations, len((*n).layers[lenght-1].weights[0])).DZdW((*n).layers[lenght-1].dZ)
	}
	// now that the last layer is done, do the rest
	startlayer := lenght - 2
	for lay := startlayer; lay >= 0; lay-- {
		(*n).layers[lay].dActivations = matrix.DZdAPrev((*n).layers[lay+1].weights).Multiply(matrix.ASum((*n).layers[lay+1].dZ[0]))
		(*n).layers[lay].dZ = ((n).layers[lay].Z.Apply(DReLU)).WeirdMult((*n).layers[lay].dActivations)
		(*n).layers[lay].dBiases = (*n).layers[lay].dZ
		if lay != 0 {
			(*n).layers[lay].dWeights = matrix.Broadcast((*n).layers[lay-1].Activations, len((*n).layers[lay].weights[0])).DZdW((*n).layers[lay].dZ)
		} else if lay == 0 {
			(*n).layers[0].dWeights = matrix.Broadcast(matrix.AtoM(inputs), len((*n).layers[0].weights[0])).DZdW((*n).layers[0].dZ)
		}
	}
	return cost
}

// DReLU is the relu deriv
func DReLU(i float64) float64 {
	if i > 0 {
		return 1.0
	}
	return 0.0
}

// Train props then backprops and then updates gradients
func (n *NN) Train(inputs [][]float64, expecteds [][]float64) {
	//num_samples := len(inputs)

}
