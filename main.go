package main

import (
	"go-mlp/nn"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	//Avery := matrix.NewMatrix(1, 3, matrix.Zeroes)
	//(*Avery)[0] = []float64{1.0, 2.0, 3.0}
	Joe := nn.NewNN([]int{2, 4, 3}, 3)
	Joe.Propogate([]float64{1.0, 2.0, 3.0})
	Joe.Backpropogate([]float64{1.0, 2.0, 3.0}, []float64{162.0, 160.0, 159.0})
	Joe.Print()
	//fmt.Println(matrix.Broadcast(*Avery, 4))
}
