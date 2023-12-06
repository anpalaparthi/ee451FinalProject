// https://medium.com/geekculture/logistic-regression-from-scratch-59e88bea2ba2

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;

#define NUM_FEATURES 9
#define SEED 42

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// numTrain threads
// numFeatures = NUM_FEATURES, for memory allocation at compile time
//X = (numTrain x numFeatures), w = (numFeatures x 1) --> (numTrain x 1)
// each thread computes 1 element of output (numTrain x 1)
// prob[id] = dot product X[id] with w (size numFeatures)
__global__ void kernelWeightedSigmoid(double* x, double* w, double* probs, int iter) {
    int id = threadIdx.x; 
    // if (id == 0) {
    //     printf("id = 0, iter = %d, w[0] = %f, w[1] = %f\n", iter, w[0], w[1]);
    // }

    double val = 0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        val += x[(id * NUM_FEATURES) + i] * w[i];
    }
    probs[id] = 1.0 / (1.0 + exp(-1.0 * val));
}

//numFeatures threads
// each thread updates w[id]
// first, gradW[id] += (probs[j] - y[j]) * x[j][id], for all 0 <= j < numTrain
// then, w[id] -= eta * gradW[id]
__global__ void kernelUpdateWeight(double* x, int* y, double* w, double* probs, int numTrain, double eta, double* gradW) {
    int id = threadIdx.x; 
    double val = gradW[id];
    for (int j = 0; j < numTrain; j++) {
        val += (probs[j] - (1.0 * y[j])) * x[(j * NUM_FEATURES) + id];
    }
    w[id] -= eta * val;
    gradW[id] = val;
    // if (id == 0) {
    // printf("id = %d, val = %f\n", id, val);
    // }
    
}


class LogisticRegression {

    private:
        double eta;
        int numIters;
        double tolerance;
        double* w;

        void matVecMul(double** A, double* b, int numRowsX, int numCols, double* result) {
            for (int i = 0; i < numRowsX; i++) {
                result[i] = 0;
            };

            for (int i = 0; i < numRowsX; i++) {
                for (int j = 0; j < numCols; j++) {
                    result[i] += A[i][j] * b[j];
                }
            }
        }

        void weightedSigmoid(double** X, int numRowsX, int lenW, double* result) {
            // denominator = 1 + np.exp(-(X * w).sum(axis=1))
            matVecMul(X, w, numRowsX, lenW, result);
            for (int i = 0; i < numRowsX; i++) {
                result[i] = 1.0 / (1.0 + exp(-1.0 * result[i]));
            }
        }

        // assumes numFeatures includes a bias feature
        void initWeights(int numFeatures) {
            w = new double[numFeatures];
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0,1.0);
            for (int i = 0; i < numFeatures; i++) {
                w[i] = distribution(generator);
            }  
        }

    public:
        LogisticRegression(double etaParam, int numItersParam, double toleranceParam) {
            eta = etaParam;
            numIters = numItersParam;
            tolerance = toleranceParam;
            w = NULL;
        }

        void setWeight(double* weight) {
            w = weight;
        }

        ~LogisticRegression() {
            // if (w != NULL) {
            //     delete[] w;
            // }
        }

        // free gradW, probs
        void fit(double** Xtrain, int* y, int numRowsX, int numFeatures) {
            // generate initial weights as small random numbers
            initWeights(numFeatures);

            double* gradW = new double[numFeatures];
            double* probs = new double[numRowsX];

            // initialize gradW
            for (int j = 0; j < numFeatures; j++) {
                gradW[j] = 0;
            }

            // gradient descent
            for (int i = 0; i < numIters; i++) {
                // calculate gradient -> (y^ - y)x_j
                weightedSigmoid(Xtrain, numRowsX, numFeatures, probs);
                for (int j = 0; j < numRowsX; j++) {
                    for (int k = 0; k < numFeatures; k++) {
                        gradW[k] += (probs[j] - y[j]) * Xtrain[j][k];
                    }
                }

                // update rule
                for (int j = 0; j < numFeatures; j++) {
                    w[j] -= eta * gradW[j];
                }

                // break tolerance (optional, write later)
            }

            // free gradW, probs
            delete[] gradW;
            delete[] probs;
        }

        // free probs
        void predict(double** Xtest, int numRowsX, int numFeatures, int* result) {
            // apply trained weighted sigmoid to get probabilities
            cout << "weight after fit: " << endl;
            for (int i = 0; i < NUM_FEATURES; i++) {
                cout << w[i] << ", ";
            }
            cout << endl;
            
            double* probs = new double[numRowsX];
            weightedSigmoid(Xtest, numRowsX, numFeatures, probs);

            for (int i = 0; i < numRowsX; i++) {
                cout << "probs i=" << i << ": " << probs[i] << endl;
            }

            // compute the class predictions
            double threshold = 0.5; // take more likely value
            for (int i = 0; i < numRowsX; i++) {
                result[i] = (probs[i] > threshold) ?  1 : 0; 
            }

            for (int i = 0; i < numRowsX; i++) {
                cout << "result i=" << i << ": " << result[i] << endl;
            }

            delete[] probs;
        }
};

// both yTrue and yPred have size elements
double accuracy(int* yTrue, int* yPred, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        cout << "yTrue " << yTrue[i] << " | " << "yPred " << yPred[i] << endl;
        if (yTrue[i] == yPred[i]) {
            sum++;
        }
    }
    return (sum / (1.0 * size));
}

// free yHatTest
int main() {
    // get the data from csv
    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 891;
    double testSize = 0.25;
    double trainSize = 0.75;
    int numTrain = ((trainSize) * numSamples) + 1;
    int numTest = testSize * numSamples;
    int numFeatures = 9; // 8 + bias

    // Logistic Regression hyperparameters
    double eta = 0.001; //1e-3
    double numIters = 1000;
    double tolerance = 0; // we're not using this 

    
    cout << "defined params" << endl;

    //allocate memory for training and test data
    double** Xtrain = new double*[numTrain];
    int* ytrain = new int[numTrain];
    for (int i = 0; i < numTrain; i++) {
        Xtrain[i] = new double[numFeatures];
    }
    
    double** Xtest = new double*[numTest];
    int* ytest = new int[numTest];
    for (int i = 0; i < numTest; i++) {
        Xtest[i] = new double[numFeatures];
    }

    
    cout << "finished allocation" << endl;

    // read from csv: https://www.youtube.com/watch?v=NFvxA-57LLA
    ifstream inputFile;
    inputFile.open("titanic_prep.csv");


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
        double xData1;
        double xData2;
        double xData3;
        double xData4;
        double xData5;
        double xData6;
        double xData7;
        double xData8;
        double xData9;
        int cls;
        int rowNum;
        string temp = "";

        stringstream inputString(line);
        getline(inputString, temp, ',');
        rowNum = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData1 = atof(temp.c_str());

        getline(inputString, temp, ',');
        cls = atoi(temp.c_str());

        getline(inputString, temp, ',');
        xData2 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData3 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData4 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData5 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData6 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData7 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData8 = atof(temp.c_str());

        getline(inputString, temp, ',');
        xData9 = atof(temp.c_str());

        if (total == numTrain) {
            idx = 0;
        }
        cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        cout << xData1 << ", " << xData2 << ", " << xData3 << ", " << xData4 << ", " << xData5 << ", " << xData6 << ", " << xData7 << ", " << xData8 << ", " << xData9 << " | cls = " << cls << endl;
        if (total < numTrain) {
            Xtrain[idx][0] = xData2;
            Xtrain[idx][1] = xData3;
            Xtrain[idx][2] = xData4;
            Xtrain[idx][3] = xData5;
            Xtrain[idx][4] = xData6;
            Xtrain[idx][5] = xData7;
            Xtrain[idx][6] = xData8;
            Xtrain[idx][7] = xData9;
            Xtrain[idx][8] = 1.0; // add bias term
            ytrain[idx] = cls;
        } else {
            // Xtest[idx][0] = xData1;
            cout << "iteration finished" << endl;
            Xtest[idx][0] = xData2;
            Xtest[idx][1] = xData3;
            Xtest[idx][2] = xData4;
            Xtest[idx][3] = xData5;
            Xtest[idx][4] = xData6;
            Xtest[idx][5] = xData7;
            Xtest[idx][6] = xData8;
            Xtest[idx][7] = xData9;
            Xtest[idx][8] = 1.0; // add bias term
            ytest[idx] = cls;
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

    
    double* Xtrain1D = new double[numTrain * numFeatures];
    double* Xtest1D = new double[numTest * numFeatures];
    for (int i = 0; i < numTrain; i++) {
        for (int j = 0; j < numFeatures; j++) {
            Xtrain1D[(i * numFeatures) + j] = Xtrain[i][j];
        }
    }
    
    for (int i = 0; i < numTest; i++) {
        for (int j = 0; j < numFeatures; j++) {
            Xtest1D[(i * numFeatures) + j] = Xtest[i][j];
        }
    }

    // define the model
    LogisticRegression classifier = LogisticRegression(eta, numIters, tolerance);

    cout << "model defined" << endl;

    // fit
    // classifier.fit(Xtrain, ytrain, numTrain, numFeatures);

    // generate initial weights as small random numbers
    double* w = new double[numFeatures];
    std::default_random_engine generator;
    generator.seed(SEED);
    std::normal_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < numFeatures; i++) {
        w[i] = distribution(generator);
    }  

    double* gradW = new double[numFeatures];
    double* probs = new double[numTrain];

    // initialize gradW
    for (int j = 0; j < numFeatures; j++) {
        gradW[j] = 0;
    }

    //malloc on gpu
    // x, w, probs, y, 
    double* gpuX;
    double* gpuW;
    double* gpuProbs;
    double* gpuGradW;
    int* gpuY;
    cudaMalloc((void**)&gpuX, sizeof(double)*numTrain*numFeatures); 
    cudaMalloc((void**)&gpuW, sizeof(double)*numFeatures); 
    cudaMalloc((void**)&gpuProbs, sizeof(double)*numTrain);  
    cudaMalloc((void**)&gpuGradW, sizeof(double)*numFeatures);  
    cudaMalloc((void**)&gpuY, sizeof(int)*numTrain); 
    
    struct timespec start, stop; 
    double time;
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    
    //memcpy cpu to gpu: x, w, y
    cudaMemcpy(gpuX, Xtrain1D, sizeof(double)*numTrain*numFeatures, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuW, w, sizeof(double)*numFeatures, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuGradW, gradW, sizeof(double)*numFeatures, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuY, ytrain, sizeof(int)*numTrain, cudaMemcpyHostToDevice);

    //do training
    // gradient descent
    dim3 dimGrid(1);
    dim3 dimBlock1(numTrain);
    dim3 dimBlock2(NUM_FEATURES);
    for (int i = 0; i < numIters; i++) {

        // calculate gradient -> (y^ - y)x_j
        //weightedSigmoid
        //__global__ void kernelWeightedSigmoid(double* x, double* w, double* probs) {
        kernelWeightedSigmoid<<<dimGrid, dimBlock1>>>(gpuX, gpuW, gpuProbs, i);

        //calculate weight gradient
        //__global__ void kernelUpdateWeight(double* x, int* y, double* w, double* probs, int numTrain, double eta) {
        kernelUpdateWeight<<<dimGrid, dimBlock2>>>(gpuX, gpuY, gpuW, gpuProbs, numTrain, eta, gpuGradW);
    }
    // memcpy gpu to cpu: w
    cudaMemcpy(w, gpuW, sizeof(double)*numFeatures, cudaMemcpyDeviceToHost);

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %f ms\n", time*1e3);	

    // free gradW, probs
    delete[] gradW;
    delete[] probs;


    cout << "model fit" << endl;

    // predict
    int* yHatTest = new int[numTest];
    classifier.setWeight(w);
    classifier.predict(Xtest, numTest, numFeatures, yHatTest);

    // accuracy
    cout << "classifier trained " << endl;
    double acc = accuracy(ytest, yHatTest, numTest);
    printf("Logistic Regression Accuracy: %f\n", acc);

    delete[] yHatTest;

    //free memory of training and test data
    for (int i = 0; i < numTrain; i++) {
        delete[] Xtrain[i];
    }
    delete[] Xtrain;
    delete[] ytrain;
    
    for (int i = 0; i < numTest; i++) {
        delete[] Xtest[i];
    }
    delete[] Xtest;
    delete[] ytest;
    
    delete[] Xtrain1D;
    delete[] Xtest1D;

    cudaFree(gpuX);
    cudaFree(gpuW);
    cudaFree(gpuProbs);
    cudaFree(gpuY);
    cudaFree(gpuGradW);

    cout << "done" << endl;
}