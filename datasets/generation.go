package datasets

import "math"

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

	return points
}
