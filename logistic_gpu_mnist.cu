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

// #define NUM_FEATURES 9
#define SEED 42

#define NUM_CLASSES 10
#define NUM_PIXELS 784
#define NUM_FEATURES 784
#define MNIST_MEAN 0.1307
#define MNIST_STD 0.3081
#define MNIST_MAX 255.0

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

        LogisticRegression() {
            eta = 0;
            numIters = -1;
            tolerance = -1;
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

// both yTrue and yPred have size elements, predictions = NUM_CLASS double* values (each double* = array of length 'size')
double accuracy(int* yTrue, int* predictions[], int size) {
    double sum = 0;
    int* yPred = new int[size];


    for (int j = 0; j < size; j++) {
        double max = 0;
        int maxId;
        for (int i = 0; i < NUM_CLASSES; i++) {
            // cout << "prediction class=" << i << " j=" << j << " pred=" << predictions[i][j] << endl;
            if (predictions[i][j] > max) {
                max = predictions[i][j];
                maxId = i;
            }
        }
        yPred[j] = maxId;
    }
    
    for (int i = 0; i < size; i++) {
        if ((i % 100) == 0) {
            cout << "yTrue " << yTrue[i] << " | " << "yPred " << yPred[i] << endl;
        }
        if (yTrue[i] == yPred[i]) {
            sum++;
        }
    }

    delete[] yPred;
    return (sum / (1.0 * size));
}

void parseMNISTData(string dataFileStr, int numTrain, int numTest, double** Xtrain, int* ytrain, double** Xtest, int* ytest) {
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
                Xtrain[idx][i] = pixels[i];
                Xtrain[idx][i] /= MNIST_MAX;
                Xtrain[idx][i] -= MNIST_MEAN;
                Xtrain[idx][i] /= MNIST_STD;

            }
            Xtrain[idx][NUM_PIXELS] = 1.0; // add a bias term
            ytrain[idx] = label;
        } else {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtest[idx][i] = pixels[i];
                Xtest[idx][i] /= MNIST_MAX;
                Xtest[idx][i] -= MNIST_MEAN;
                Xtest[idx][i] /= MNIST_STD;
            }
            Xtest[idx][NUM_PIXELS] = 1.0; // add a bias term
            ytest[idx] = label;
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


// free yHatTest
int main() {
    // get the data from csv
    int numSamples = 42000;
    double testSize = 0.3;
    double trainSize = 0.2;
    int numTrain = trainSize * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = NUM_PIXELS + 1; // add a bias term

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
    string dataFileStr = "mnist_train.csv";

    if (dataFileStr == "mnist_train.csv") {
        parseMNISTData(dataFileStr, numTrain, numTest, Xtrain, ytrain, Xtest, ytest);
    } else {
        cout << "File " << dataFileStr << " not supported" << endl;
    }

    
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
    LogisticRegression classifiers[10];
    for (int i = 0; i < 10; i++) {
        classifiers[i] = LogisticRegression(eta, numIters, tolerance);
    }
    cout << "model defined" << endl;

    // fit
    // classifier.fit(Xtrain, ytrain, numTrain, numFeatures);

    // generate initial weights as small random numbers
    double totalTime = 0;
    int* predictions[NUM_CLASSES];
    int* yTrainOVR = new int[numTrain];
    int* yTestOVR = new int[numTest];
    for (int i = 0; i < 10; i++) {
        // convert ytrain and ytest to be one-vs-all
        for (int j = 0; j < numTrain; j++) {
            yTrainOVR[j] = (ytrain[j] == i) ? 1 : 0;
        }

        for (int j = 0; j < numTest; j++) {
            yTestOVR[j] = (ytest[j] == i) ? 1 : 0;
        }

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
        cudaMemcpy(gpuY, yTrainOVR, sizeof(int)*numTrain, cudaMemcpyHostToDevice);

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
        // printf("time is %f ms\n", time*1e3);
        totalTime += time;	

        // free gradW, probs
        delete[] gradW;
        delete[] probs;
        classifiers[i].setWeight(w);

        int* prediction = new int[numTest];
        classifiers[i].predict(Xtest, numTest, numFeatures, prediction);
        // for (int k = 0; k < numTest; k++) {
        //     cout << "probs out classifier i=" << i << ": " << prediction[k] << ", " << endl;
        // }
        predictions[i] = prediction;
        delete[] w;
        
        cudaFree(gpuX);
        cudaFree(gpuW);
        cudaFree(gpuProbs);
        cudaFree(gpuY);
        cudaFree(gpuGradW);
    }

    
    delete[] yTrainOVR;
    delete[] yTestOVR;


    cout << "classifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("Logistic Regression Accuracy: %f\n", acc);
    printf("Training Execution Time: %f sec\n", totalTime);

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
    //DELETE PREDICTIONS    
    for (int i = 0; i < NUM_CLASSES; i++) {
        delete[] predictions[i];
    }
    cout << "done" << endl;
}