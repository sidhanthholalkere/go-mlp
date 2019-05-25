package main

import (
	"go-mlp/nn"
	"math/rand"
	"fmt"
)

func main() {
	rand.Seed(1)

	Avery := nn.NewNN([]int{3, 2}, 2)
	inputs := []float64{1.5, 2.5}
	outputs := []float64{1, 0}
	Avery.Propogate(inputs)
	Avery.Backpropogate(outputs, inputs)
	Avery.Print()
	cost := Avery.GetCost(outputs)
	fmt.Println("Cost: ", cost)
}
