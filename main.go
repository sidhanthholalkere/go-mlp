package main

import (
	"encoding/csv"
	"go-mlp/nn"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func main() {
	rand.Seed(1)
	hearts, _ := os.Open("heart.csv")
	defer hearts.Close()

	r := csv.NewReader(hearts)

	rows, _ := r.ReadAll()
	rows[0][0] = "63"

	Shuffle(rows)
	Shuffle(rows)

	totalLabels := make([][]float64, len(rows))
	for row := range totalLabels {
		totalLabels[row] = make([]float64, 2)
	}

	totalData := make([][]float64, len(rows))
	for row := range totalData {
		totalData[row] = make([]float64, 13)
	}

	for row := range rows {
		if rows[row][13] == "0" {
			totalLabels[row] = []float64{1.0, 0.0}
		} else {
			totalLabels[row] = []float64{0.0, 1.0}
		}
	}

	for row := range rows {
		for i := 0; i < 13; i++ {
			totalData[row][i], _ = strconv.ParseFloat(rows[row][i], 64)
		}
	}

	for row := range totalData {
		totalData[row][0] /= 77.0
		totalData[row][2] /= 3.0
		totalData[row][3] /= 200.0
		totalData[row][4] /= 564.0
		totalData[row][6] /= 2.0
		totalData[row][7] /= 202.0
		totalData[row][9] /= 6.2
		totalData[row][10] /= 2.0
		totalData[row][11] /= 4.0
		totalData[row][12] /= 3.0
	}

	Avery := nn.NewNN([]int{2}, 13)
	Avery.Print()
}

// Shuffle (taken from	https://www.calhoun.io/how-to-shuffle-arrays-and-slices-in-go/)
func Shuffle(vals [][]string) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for len(vals) > 0 {
		n := len(vals)
		randIndex := r.Intn(n)
		vals[n-1], vals[randIndex] = vals[randIndex], vals[n-1]
		vals = vals[:n-1]
	}
}
