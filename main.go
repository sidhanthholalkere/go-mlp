package main

import (
	"go-mlp/nn"
	"math/rand"
	"fmt"
)

func main() {
	rand.Seed(1)

	mlp := nn.NewNN([]int{3, 2}, 2)
	inputs := []float64{1.5, 2.5}
	outputs := []float64{1, 0}
	mlp.Propogate(inputs)
	mlp.Backpropogate(outputs, inputs)
	mlp.Print()
	cost := mlp.GetCost(outputs)
	fmt.Println("Cost: ", cost)
}
