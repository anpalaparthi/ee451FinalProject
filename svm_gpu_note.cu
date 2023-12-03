// satisfy Contstraint: x . w, lenW elements

// getGradients: 
// if constrain: scalar muliply lambda * w = dw, db = 0
// else
    // dw = lambda * w - (cls[idx] * x), db = -cls[idx]

// updateWeights
    // scalar multiplication