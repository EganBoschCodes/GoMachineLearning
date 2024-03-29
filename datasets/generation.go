package datasets

import (
	"go-backprop/utils"
	"math"
	"math/rand"
)

type DataPoint struct {
	Input  []float32
	Output []float32
}

func GetSpiralDataset() []DataPoint {
	points := make([]DataPoint, 0)

	for r := 0.2; r < 3; r += 0.05 {
		p1 := DataPoint{Input: []float32{float32(r * math.Sin(r)), float32(r * math.Cos(r))}, Output: []float32{1, 0, 0}}
		p2 := DataPoint{Input: []float32{float32(r * math.Sin(r+2.049)), float32(r * math.Cos(r+2.049))}, Output: []float32{0, 1, 0}}
		p3 := DataPoint{Input: []float32{float32(r * math.Sin(r-2.049)), float32(r * math.Cos(r-2.049))}, Output: []float32{0, 0, 1}}

		points = append(points, p1, p2, p3)
	}

	rand.Shuffle(len(points), func(i, j int) { points[i], points[j] = points[j], points[i] })

	return points
}

func GetMean(dataset []DataPoint, sampleSize int) []float32 {
	numSamples := utils.Min(sampleSize, len(dataset))

	means := make([]float32, 0)
	numInputs := len(dataset[0].Input)
	for i := 0; i < numInputs; i++ {
		means = append(means, 0)
		for j := 0; j < numSamples; j++ {
			datapoint := dataset[j]
			means[i] += datapoint.Input[i]
		}
		means[i] /= float32(numSamples)
	}
	return means
}

func NormalizeInputs(dataset []DataPoint) {
	rand.Shuffle(len(dataset), func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })

	sampleSize := 50
	means := GetMean(dataset, sampleSize)

	numInputs := len(dataset[0].Input)
	stddevs := make([]float32, 0)
	for i := 0; i < numInputs; i++ {
		stddevs = append(stddevs, 0)
		for j := 0; j < utils.Min(sampleSize, len(dataset)); j++ {
			datapoint := dataset[j]
			diff := datapoint.Input[i] - means[i]
			stddevs[i] += diff * diff
		}
		stddevs[i] = float32(math.Sqrt(float64(stddevs[i]) / float64(utils.Min(sampleSize, len(dataset)))))
	}

	for i := 0; i < numInputs; i++ {
		for _, datapoint := range dataset {
			datapoint.Input[i] = (datapoint.Input[i] - means[i]) / stddevs[i]
		}
	}
}
