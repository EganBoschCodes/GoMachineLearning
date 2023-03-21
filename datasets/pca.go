package datasets

import (
	"go-backprop/utils"

	"gonum.org/v1/gonum/mat"
)

func PCAReduce(data []DataPoint, dimensions int) {
	// Basic input checking
	input_dim := len(data[0].Input)
	if input_dim <= dimensions {
		return
	}

	// Center the Data
	mean := GetMean(data, 200)
	for _, datapoint := range data {
		for i := 0; i < input_dim; i++ {
			datapoint.Input[i] -= mean[i]
		}
	}

	// Compute the Sample Covariance Matrix
	s := make([]float64, input_dim*input_dim)
	for i := range s {
		s[i] = 0
	}
	for _, datapoint := range data {
		covariance_matrix := covariance(datapoint.Input)
		for i := range covariance_matrix {
			for j := range covariance_matrix[i] {
				s[input_dim*i+j] += float64(covariance_matrix[i][j])
			}
		}
	}

	S := mat.NewSymDense(input_dim, s)

	// Calculate the eigenvectors and their corresponding eigenvalues
	var eig mat.EigenSym
	eig.Factorize(S, true)
	eigenvectors := mat.NewDense(input_dim, input_dim, nil)
	eig.VectorsTo(eigenvectors)

	// Choose only the ones with the greatest eigenvalues (gonum automatically sorts the eigenvectors as corresponding from smallest to greatest)
	chosenEigenvectors := make([][]float32, 0)
	for col := input_dim - 1; col > input_dim-dimensions-1; col-- {
		chosenEigenvectors = append(chosenEigenvectors, utils.Float32(mat.Col(nil, col, eigenvectors)))
	}

	// Project inputs onto the chosen eigenvectors
	for index := range data {
		newInput := make([]float32, dimensions)
		for i, ev := range chosenEigenvectors {
			newInput[i] = utils.Dot(ev, data[index].Input)
		}
		data[index].Input = newInput
	}
}

func covariance(a []float32) [][]float32 {
	matrix := make([][]float32, len(a))
	for i := range matrix {
		matrix[i] = make([]float32, len(a))
		for j := range matrix[i] {
			matrix[i][j] = a[i] * a[j]
		}
	}
	return matrix
}
