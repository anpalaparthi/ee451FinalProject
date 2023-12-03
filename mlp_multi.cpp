// Based on python tutorial for general MLPs
//  https://pub.towardsai.net/the-multilayer-perceptron-built-and-implemented-from-scratch-70d6b30f1964

#include <stdlib.h>
#include <stdio.h>
// #include <cublas.h>
#include <time.h>
#include <cmath>
#include <random>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

#define NUM_CLASSES 10
#define NUM_PIXELS 784

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
                delta2_1[(i * outputLayer) + j] = pHat[(i * outputLayer) + j] - y[(i * outputLayer) + j];
            } 
        }

        //partialDeriv2 = activations.T * delta2_1
        //  (numSamples x hiddenLayer).T * (num_samples x output_layer)
        //  = (hiddenLayer x numSamples) * (num_samples x output_layer)
        partialDeriv2 = new float[hiddenLayer * outputLayer];
        
        // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
        matrixMultATranspose(activations, delta2_1, partialDeriv2, hiddenLayer, numSamples, outputLayer);
        
        float* delta1_1 = delta2_1;
        float* delta1_2 = new float[numSamples * hiddenLayer];
        multBroadcast(delta1_1, weightMatrix2, delta1_2, numSamples, hiddenLayer);

        float* sigDeriv = sigmoid_derivative(z1, numSamples * hiddenLayer);
        float* delta1_3 = new float[numSamples * hiddenLayer];
        hadamard(delta1_2, sigDeriv, delta1_3, numSamples, hiddenLayer);
        
        partialDeriv1 = new float[inputLayer * hiddenLayer];
        // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
        matrixMultATranspose(designMatrix, delta1_3, partialDeriv1, inputLayer, numSamples, hiddenLayer);

        delete[] delta2_1;
        delete[] delta1_2;
        delete[] sigDeriv;
        delete[] delta1_3;
    }

    // c = a x b ((px q) * (q x r))
    void matrixMult(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                c[(i * r) + j] = 0;
            }
        }

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * r) + j] += a[(i * q) + k] * b[(k * r) + j];
                }
            }
        }
    }

    // c = a.T x b ((q x p).T * (q x r) = (p x q) * (q x r))
    ////partialDeriv2 = activations.T * delta2_1
        //  (numSamples x hiddenLayer).T * (num_samples x output_layer)
        //  = (hiddenLayer x numSamples) * (num_samples x output_layer)
        // matrixMultATranspose(activations, delta2_1, partialDeriv2, hiddenLayer, numSamples, outputLayer);
    void matrixMultATranspose(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                c[(i * r) + j] = 0;
            }
        }

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * r) + j] += a[(k * p) + i] * b[(k * r) + j];
                }
            }
        }
    }
    
    //DON'T REALLY NEED THIS - BONUS
    // c = a.T x b ((p x q) * (r x q).T = (p x q) * (q x r))
    void matrixMultBTranspose(float* a, float* b, float* c, int p, int q, int r) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                c[(i * r) + j] = 0;
            }
        }

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < r; j++) {
                for (int k = 0; k < q; k++) {
                    c[(i * r) + j] += a[(i * q) + k] * b[(j * q) + k];
                }
            }
        }
    }

    // a = p x 1, b = 1 x q --> c = p x q
    // c[:][i] = a * b[i]
    void multBroadcast(float* a, float* b, float* c, int p, int q) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                c[(i * q) + j] = a[i] * b[j];
            }
        }
    }

    // p x q element-wise multiplication (Hadamard product)
    void hadamard(float* a, float* b, float *c, int p, int q) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                c[(i * q) + j] = a[(i * q) + j] * b[(i * q) + j];
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

            // normal random number gerneation
            //  https://cplusplus.com/reference/random/normal_distribution/
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0,1.0);
            for (int i = 0; i < inputLayer * hiddenLayer; i++) {
                weightMatrix1[i] = distribution(generator);
            }
            
            // cout << "hidden layer = " << hiddenLayer << endl;
            // cout << "outer layer = " << outputLayer << endl;

            for (int i = 0; i < hiddenLayer * outputLayer; i++) {
                weightMatrix2[i] = distribution(generator);
            }
            
            // cout << "*******construct weight matrix 1******" << endl;
            // for (int i = 0; i < inputLayer * hiddenLayer; i++) {
            //     cout << "construct weight matrix 1: " << weightMatrix1[i] << endl;
            // }
            // cout << "*******construct weight matrix 1******" << endl;
            
            // cout << "*******construct weight matrix 2******" << endl;
            // for (int i = 0; i < hiddenLayer * outputLayer; i++) {
            //     cout << "construct weight matrix 2: " << weightMatrix2[i] << endl;
            // }
            // cout << "*******construct weight matrix 2******" << endl;

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
             
            pHat = sigmoid(zh, numSamples * outputLayer);
        }

        // DELETE THIS
        float* predict(float* testData, int num) { // testData = num x input
            
            /*
            cout << "*******test data******" << endl;
            for (int i = 0; i < num * inputLayer; i++) {
                cout << "testData: " << testData[i] << endl;
            }
            cout << "*******test data******" << endl;
            */
            
            // cout << "*******weight matrix 1******" << endl;
            // for (int i = 0; i < inputLayer * hiddenLayer; i++) {
            //     cout << "weight matrix 1: " << weightMatrix1[i] << endl;
            // }
            // cout << "*******weight matrix 1******" << endl;
            

            float* z = new float[num * hiddenLayer];
            // multiply testData * weightMatrix1 
            //  ( (num x inputLayer) * (inputLayer x hiddenLayer) )
            matrixMult(testData, weightMatrix1, z, num, inputLayer, hiddenLayer);
            
            /*
            cout << "*******z******" << endl;
            for (int i = 0; i < num * hiddenLayer; i++) {
                cout << "z: " << z[i] << endl;
            }
            cout << "*******z******" << endl;
            */
            float* activations = sigmoid(z, num * hiddenLayer);
            /*
            cout << "*******activations******" << endl;
            for (int i = 0; i < num * hiddenLayer; i++) {
                cout << "activations: " << activations[i] << endl;
            }
            cout << "*******activations******" << endl;
            */

            // zh = multiply activations * weightMatrix 2
            //  (numSamples x hiddenLayer) * (hiddenLayer x outputLayer)
            float* zh = new float[num * outputLayer];
            matrixMult(activations, weightMatrix2, zh, num, hiddenLayer, outputLayer);        
            /*
            cout << "**********zh**********" << endl;
            for (int i = 0; i < num * outputLayer; i++) {
                cout << "zh = " << zh[i] << endl;
            }
            cout << "*********zh**********" << endl;
             */
            
            float* pHat = sigmoid(zh, num * outputLayer);
            /*
            cout << "**********predict values***********" << endl;
            for (int i = 0; i < num*outputLayer; i++) {
                cout << "phat = " << pHat[i] << endl;
            }
            cout << "**********predict values***********" << endl;
            */
            delete[] z;
            delete[] zh;
            
            return pHat;
        }

        void updateWeights(float* partialDeriv1, float* partialDeriv2) {
            // update weight matrix1 = input x hidden
            for (int i = 0; i < inputLayer; i++) {
                for (int j = 0; j < hiddenLayer; j++) {
                    weightMatrix1[(i * hiddenLayer) + j] -= lr * partialDeriv1[(i * hiddenLayer) + j];
                }
            }

            // update weight matrix2 = hidden x output
            for (int i = 0; i < hiddenLayer; i++) {
                for (int j = 0; j < outputLayer; j++) {
                    weightMatrix2[(i * outputLayer) + j] -= lr * partialDeriv2[(i * outputLayer) + j];
                }
            }
        }

        void train() {
            for (int i = 0; i < iters; i++) {
                cout << "train iter = " << i << endl;
                // if ((i % 20) == 0) {
                    cout << "train iter = " << i << endl;
                // }
                // cout << "******" << i << " train weight matrix******" << endl;
                // for (int j = 0; j < inputLayer * hiddenLayer; j++) {
                //     cout << weightMatrix1[j] << " ";
                // }
                // cout << endl << "******" << i << " train weight matrix******" << endl;
                
                float* z1;
                float* activations;
                float* z2;
                float* pHat;
                forwardProp(z1, activations, z2, pHat);

                cout << i << " after forward prop" << endl;
                float* partialDeriv1;
                float* partialDeriv2;
                backProp(z1, activations, z2, pHat, partialDeriv1, partialDeriv2);
                cout << i << " after back prop" << endl;

                updateWeights(partialDeriv1, partialDeriv2);
                cout << i << " after update weights" << endl;

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
        cout << "yTrue[i] " << yTrue[i] << " | " << "yPred[i] " << yPred[i] << endl;
        int trueVal = (int) lround(yTrue[i]);
        int pred = (int) lround(yPred[i]);
        cout << "yTrue " << trueVal << " | " << "yPred " << pred << endl;
        if (trueVal == pred) {
            sum++;
        }
    }
    return (sum / (1.0 * size));
}

void parseMNISTData(string dataFileStr, int numTrain, int numTest, float* Xtrain, float* ytrain, float* Xtest, float* ytest, int numFeatures) {
    ifstream inputFile;
    inputFile.open(dataFileStr);
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
        int label;
        double pixels[NUM_PIXELS];
        string temp = "";

        stringstream inputString(line);
        // ss >> xData1 >> xData2 >> cls;
        getline(inputString, temp, ',');
        label = atoi(temp.c_str());
        for (int i = 0; i < NUM_PIXELS; i++) {
            getline(inputString, temp, ',');
            pixels[i] = atof(temp.c_str());
        }        

        if (total == numTrain) {
            idx = 0;
        }
        // cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        // cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
        if (total < numTrain) {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtrain[(idx * numFeatures) + i] = 1.0 * pixels[i];
            }
            ytrain[idx] = 1.0 * label;
        } else {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtest[(idx * numFeatures) + i] = 1.0 * pixels[i];
            }
            ytest[idx] = 1.0 * label;
        }

        line = "";
        total++;
        idx++;

        if (total == (numTrain + numTest)) {
            break;
        }

    }
        
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

}

int main() {

    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 250;
    double testSize = 0.3;
    double trainSize = 0.2;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 2;

    // bool blobs = true;
    bool blobs = false;

    // SVM hyperparameters
    float learningRate = 0.001; //1e-3
    int iters = 1000;
    int numHidden = 3;
    
    cout << "defined params" << endl;

    float* Xtrain;
    float* ytrain;
    float* Xtest;
    float* ytest;

    if (!blobs) {
        numSamples = 42000;
        testSize = 0.3;
        trainSize = 0.2;
        numTrain = trainSize * numSamples;
        numTest = testSize * numSamples;
        numFeatures = NUM_PIXELS;
        numHidden = 524;
        
        //allocate memory for training and test data
        Xtrain = new float[numTrain * numFeatures];
        ytrain = new float[numTrain];
        // for (int i = 0; i < numTrain; i++) {
        //     Xtrain[i] = new float[numFeatures];
        // }
        
        Xtest = new float[numTest * numFeatures];
        ytest = new float[numTest];
        // for (int i = 0; i < numTest; i++) {
        //     Xtest[i] = new float[numFeatures];
        // }
        
        cout << "finished allocation" << endl;
        parseMNISTData("mnist_train.csv", numTrain, numTest, Xtrain, ytrain, Xtest, ytest, numFeatures);
    } else {
        
        //allocate memory for training and test data
        Xtrain = new float[numTrain * numFeatures];
        ytrain = new float[numTrain];
        // for (int i = 0; i < numTrain; i++) {
        //     Xtrain[i] = new float[numFeatures];
        // }
        
        Xtest = new float[numTest * numFeatures];
        ytest = new float[numTest];
        // for (int i = 0; i < numTest; i++) {
        //     Xtest[i] = new float[numFeatures];
        // }
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
            // cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
            // cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
            if (total < numTrain) {
                Xtrain[(idx * numFeatures) + 0] = xData1;
                Xtrain[(idx * numFeatures) + 1] = xData2;
                ytrain[idx] = 1.0 * cls;
            } else {
                Xtest[(idx * numFeatures) + 0] = xData1;
                Xtest[(idx * numFeatures) + 1] = xData2;
                ytest[idx] = 1.0 * cls;
            }

            line = "";
            total++;
            idx++;

        }
        
        cout << "file read" << endl;
        inputFile.close();
        cout << "file closed" << endl;
    }

    // MLP(float lr_in, int inputLayer_in, int hiddenLayer_in, 
                // int iters_in, float* designMatrix_in, 
                // int numSamples_in, float* y_in)
        
    cout << "before constructor" << endl;
    MLP classifier = MLP(learningRate, numFeatures, numHidden, iters, Xtrain, numTrain, ytrain);
    cout << "after constructor" << endl;
    classifier.train();
    float* predictions = classifier.predict(Xtest, numTest);
    
    cout << "cassifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("MLP Accuracy: %f\n", acc);

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

    cout << "WERTY done" << endl;
    
}