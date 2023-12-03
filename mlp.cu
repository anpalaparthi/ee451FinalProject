// Based on python tutorial for general MLPs
//  https://pub.towardsai.net/the-multilayer-perceptron-built-and-implemented-from-scratch-70d6b30f1964

#include <stdlib.h>
#include <stdio.h>
// #include <cublas.h>
#include <time.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class MLP {
    private:
        float lr;
        int inputLayer;
        int hiddenLayer;
        int outputLayer = 1;
        int iters;
        float* designMatrix;
        float* weightMatrix1;
        float* weightMatrix2;
        int numSamples;
        float* y;

    float* sigmoid(float* x, int size) {
        float* result = new float[size];

        for (int i = 0; i < size; i++) {
            result[i] = 1.0 / (1.0 + exp(-1.0 * x[i]));
        }

        // DELETE RETURNED RESULT
        return result;
    }    

    float* sigmoid_derivative(float* x, int size) {
        float* result = sigmoid(x, size);
        for (int i = 0; i < size; i++) {
            result[i] *= (1 - result[i]);
        }

        // DELETE RETURNED RESULT
        return result;
    }    

    void backProp(float* z1, float* activations, float* z2, float* pHat, float*& partialDeriv1, float*& partialDeriv2) {
        
        float* delta2_1 = new float[numSamples * outputLayer];
        //delta2_1 = pHat - y (y = numSamples x outputLayer)
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < outputLayer; j++) {
                delta2_1[(i * numSamples) + j] += pHat[(i * numSamples) + j] - y[(i * numSamples) + j];
            } 
        }

        //partialDeriv2 = activations.T * delta2_1
        //  (numSamples x hiddenLayer).T * (num_samples x output_layer)
        //  = (hiddenLayer x numSamples) * (num_samples x output_layer)
        float* partialDeriv2 = new float[hiddenLayer * outputLayer];
        
        // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
        matrixMultATranspose(activations, delta2_1, partialDeriv2, numSamples, hiddenLayer, outputLayer);
        
        float* delta1_1 = delta2_1;
        float* delta1_2 = new float[numSamples * hiddenLayer];
        multBroadcast(delta1_1, weightMatrix2, delta1_2, numSamples, hiddenLayer);

        float* sigDeriv = sigmoid_derivative(z1, numSamples * hiddenLayer);
        float* delta1_3 = new float[numSamples * hiddenLayer];
        hadamard(delta1_2, sigDeriv, delta1_3, numSamples, hiddenLayer);
        
        float* partialDeriv1 = new float[inputLayer * hiddenLayer];
        // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
        matrixMultATranspose(designMatrix, delta1_3, partialDeriv1, numSamples, inputLayer, hiddenLayer);

        delete[] delta2_1;
        delete[] delta1_2;
        delete[] sigDeriv;
        delete[] delta1_3;
    }

    // c = a x b ((px q) * (q x r))
    void matrixMult(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * p) + j] += a[(i * p) + k] * b[(k * q) + j];
                }
            }
        }
    }

    // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
    void matrixMultATranspose(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * p) + j] += a[(k * q) + i] * b[(k * q) + j];
                }
            }
        }
    }
    
    //DON'T REALLY NEED THIS - BONUS
    // c = a.T x b ((p x q) * (r x q).T = (p x q) * (q x r))
    void matrixMultBTranspose(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * p) + j] += a[(i * p) + k] * b[(j * r) + k];
                }
            }
        }
    }

    // a = p x 1, b = 1 x q --> c = p x q
    // c[:][i] = a * b[i]
    void multBroadcast(float* a, float* b, float* c, int p, int q) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                c[(i * p) + j] = a[i] * b[j];
            }
        }
    }

    // p x q element-wise multiplication (Hadamard product)
    void hadamard(float* a, float* b, float *c, int p, int q) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                c[(i * p) + j] = a[(i * p) + j] * b[(i * p) + j];
            }
        }
    }

    public:
        MLP(float lr_in, int inputLayer_in, int hiddenLayer_in, 
                int iters_in, float* designMatrix_in, 
                int numSamples_in, float* y_in) {
            lr = lr_in;
            inputLayer = inputLayer_in;
            hiddenLayer = hiddenLayer_in;
            iters = iters_in;
            designMatrix = designMatrix_in;

            weightMatrix1 = new float[inputLayer * hiddenLayer];
            weightMatrix2 = new float[hiddenLayer * outputLayer];     
            numSamples = numSamples_in;       
            y = y_in;
        }

        // DELETE THESE
        void forwardProp(float*& z, float*& activations, float*& zh, float*& pHat) {
            
            z = new float[numSamples * hiddenLayer];
            // multiply designMatrix * weightMatrix1 
            //  ( (numSamples x inputLayer) * (inputLayer x hiddenLayer) )
            matrixMult(designMatrix, weightMatrix1, z, numSamples, inputLayer, hiddenLayer);

            activations = sigmoid(z, numSamples * hiddenLayer);

            // zh = multiply activations * weightMatrix 2
            //  (numSamples x hiddenLayer) * (hiddenLayer x outputLayer)
            zh = new float[numSamples * outputLayer];
            matrixMult(activations, weightMatrix2, zh, numSamples, hiddenLayer, outputLayer);        
             
            pHat = sigmoid(zh, numSamples * hiddenLayer);
        }

        // DELETE THIS
        float* predict(float* testData, int num) { // testData = num x input
            
            float* z = new float[num * hiddenLayer];
            // multiply testData * weightMatrix1 
            //  ( (num x inputLayer) * (inputLayer x hiddenLayer) )
            matrixMult(testData, weightMatrix1, z, num, inputLayer, hiddenLayer);

            float* activations = sigmoid(z, num * hiddenLayer);

            // zh = multiply activations * weightMatrix 2
            //  (numSamples x hiddenLayer) * (hiddenLayer x outputLayer)
            float* zh = new float[num * outputLayer];
            matrixMult(activations, weightMatrix2, zh, num, hiddenLayer, outputLayer);        
             
            float* pHat = sigmoid(zh, num * hiddenLayer);
            delete[] z;
            delete[] zh;
            
            return pHat;
        }

        void updateWeights(float* partialDeriv1, float* partialDeriv2) {
            // update weight matrix1 = input x hidden
            for (int i = 0; i < inputLayer; i++) {
                for (int j = 0; j < hiddenLayer; j++) {
                    weightMatrix1[(i * inputLayer) + j] -= lr * partialDeriv1[(i * inputLayer) + j];
                }
            }

            // update weight matrix2 = hidden x output
            for (int i = 0; i < hiddenLayer; i++) {
                for (int j = 0; j < outputLayer; j++) {
                    weightMatrix2[(i * hiddenLayer) + j] -= lr * partialDeriv2[(i * hiddenLayer) + j];
                }
            }
        }

        void train() {
            for (int i = 0; i < iters; i++) {
                float* z1;
                float* activations;
                float* z2;
                float* pHat;
                forwardProp(z1, activations, z2, pHat);

                float* partialDeriv1;
                float* partialDeriv2;
                backProp(z1, activations, z2, pHat, partialDeriv1, partialDeriv2);

                updateWeights(partialDeriv1, partialDeriv2);

                delete[] z1;
                delete[] activations;
                delete[] z2;
                delete[] pHat;
                delete[] partialDeriv1;
                delete[] partialDeriv2;
            }
        }

        ~MLP() {
            delete[] weightMatrix1;
            delete[] weightMatrix2;
        }
};

// both yTrue and yPred have size elements
double accuracy(float* yTrue, float* yPred, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        int trueVal = (int) yTrue[i];
        int pred = (int) yPred[i];
        cout << "yTrue " << trueVal << " | " << "yPred " << pred << endl;
        if (trueVal == pred) {
            sum++;
        }
    }
    return (sum / (1.0 * size));
}

int main() {

    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 250;
    double testSize = 0.1;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 2;

    // SVM hyperparameters
    float learningRate = 0.001; //1e-3
    int iters = 1000;
    int numHidden = 3;
    
    cout << "defined params" << endl;

    //allocate memory for training and test data
    float* Xtrain = new float[numTrain * numFeatures];
    float* ytrain = new float[numTrain];
    // for (int i = 0; i < numTrain; i++) {
    //     Xtrain[i] = new float[numFeatures];
    // }
    
    float* Xtest = new float[numTest * numFeatures];
    float* ytest = new int[numTest];
    // for (int i = 0; i < numTest; i++) {
    //     Xtest[i] = new float[numFeatures];
    // }

    cout << "finished allocation" << endl;

    // read from csv: https://www.youtube.com/watch?v=NFvxA-57LLA
    ifstream inputFile;
    inputFile.open("blob_data.csv");

    cout << "open file" << endl;

    string line = "";
    int total = 0;
    bool flag = true;
    int idx = 0;
    while (getline(inputFile, line)) {
        if (flag) {
            flag = false;
            continue;
        }
        float xData1;
        float xData2;
        int cls;
        string temp = "";

        stringstream inputString(line);
        // ss >> xData1 >> xData2 >> cls;
        getline(inputString, temp, ',');
        xData1 = atof(temp.c_str());
        getline(inputString, temp, ',');
        xData2 = atof(temp.c_str());
        getline(inputString, temp, ',');
        cls = atoi(temp.c_str());

        if (total == numTrain) {
            idx = 0;
        }
        cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
        if (total < numTrain) {
            Xtrain[(idx * numTrain) + 0] = xData1;
            Xtrain[(idx * numTrain) + 1] = xData2;
            ytrain[idx] = 1.0 * cls;
        } else {
            Xtest[(idx * numTest) + 0] = xData1;
            Xtest[(idx * numTest) + 1] = xData2;
            ytest[idx] = 1.0 * cls;
        }

        line = "";
        total++;
        idx++;

    }
    
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

    // MLP(float lr_in, int inputLayer_in, int hiddenLayer_in, 
                // int iters_in, float* designMatrix_in, 
                // int numSamples_in, float* y_in)
        
    MLP classifier = MLP(learningRate, numFeatures, numHidden, iters, Xtrain, numTrain, ytrain);
    classifier.train();
    float* predictions = classifier.predict(Xtest, numTest);
    
    cout << "cassifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("SVM Accuracy: %f\n", acc);

    delete[] predictions;

    //free memory of training and test data
    // for (int i = 0; i < numTrain; i++) {
    //     delete[] Xtrain[i];
    // }
    delete[] Xtrain;
    delete[] ytrain;
    
    // for (int i = 0; i < numTest; i++) {
    //     delete[] Xtest[i];
    // }
    delete[] Xtest;
    delete[] ytest;

    cout << "done" << endl;
    
}