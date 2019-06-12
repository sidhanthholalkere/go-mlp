package main

import (
	"go-mlp/nn"
	"math/rand"
)

func main() {
	rand.Seed(1)

	inputs := [][]float64{
		[]float64{1, 1},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 0},
	}
	outputs := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}
	mlp := nn.NewNN([]int{2, 1}, 2)
	mlp.Print()
	for i := 0; i < 100; i++ {
		mlp.Train(inputs, outputs, 2.0)
	}
	mlp.Print()
}
