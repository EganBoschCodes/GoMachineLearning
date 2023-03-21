package datasets

func ApplyPhi(dataset []DataPoint, phi func(...float32) []float32) {
	for i := range dataset {
		dataset[i].Input = phi(dataset[i].Input...)
	}
}
