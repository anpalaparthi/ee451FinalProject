// Based on python tutorial
// https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch


#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <cmath>
#include <random>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#define SEED 100

using namespace std;

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

// number of threads = hiddenLayer
// each thread computes 1 element for outputL1[]
// x = (1 x inputLayer), weightHidden = (inputLayer x hiddenLayer)
// outputL1 = 1 x hiddenLayer
// outputL1[id] = dot product x and weightHidden[:][id] (: from 0 to inputLayer)
// outputL1[id] = sigmoid(outputL1[id] + biasHidden[id])
//   sigmoid = 1.0 / (1.0 + exp(-1.0 * val))
__global__ void kernelOutputL1(double* x, double* weightHidden, double* biasHidden, double* outputL1, int inputLayer, int hiddenLayer) {
    int id = threadIdx.x;
    double val = 0;
    for (int i = 0; i < inputLayer; i++) {
        val += x[i] * weightHidden[(i * hiddenLayer) + id];
    }
    val += biasHidden[id];
    outputL1[id] = 1.0 / (1.0 + exp(-1.0 * val));
}

// number of threads = outputLayer
// each thread computes 1 element of outputL2[]
// outputL1 = (1 x hiddenLayer), weightOutput = (hiddenlayer x outputLayer)
// outputL2 = 1 x outputLayer
// outputL2[id] = dot product outputL1 and weightOutput[:][id] (: j from 0 to hiddenLayer)
// outputL2[id] = sigmoid (outputL2[id] + biasOutput[id])
//   sigmoid = 1.0 / (1.0 + exp(-1.0 * val))
__global__ void kernelOutputL2(double* outputL1, double* weightOutput, double* biasOutput, int hiddenLayer, int outputLayer, int yVal, int* output) {
    int id = threadIdx.x;
    double val = 0;
    for (int i = 0; i < hiddenLayer; i++) {
        val += outputL1[i] * weightOutput[(i * outputLayer) + id];
    }
    val += biasOutput[id];
    outputL1[id] = 1.0 / (1.0 + exp(-1.0 * val));

    if (id == yVal) {
        output[id] = 1;
    } else {
        output[id] = 0;
    }
}

// number of threads = outputLayer
// each thread computes deltaOutput[id], weightOutput[i][id], biasOutput[id]
__global__ void kernelUpdateWeightOutput(int* output, double* outputL1, double* outputL2, double* weightOutput, double* biasOutput, double* deltaOutput, double lr, int hiddenLayer, int outputLayer) {
    int id = threadIdx.x;
    double outputL2Val = outputL2[id];
    double deltaVal = (1.0 * output[id]) - outputL2Val;
    deltaVal = (-1.0 * deltaVal) - (outputL2Val * (1.0 - outputL2Val));

    for (int i = 0; i < hiddenLayer; i++) {
        weightOutput[(i * outputLayer) + id] -= (lr * deltaVal * outputL1[i]);
    }
    biasOutput[id] -= (lr * deltaVal * hiddenLayer);
    deltaOutput[id] = deltaVal;
}

// number of threads = hiddenLayer
// each thread computes product[id] (implicit), deltaHidden[id] (implicit), weightHidden[i][id], biasHidden[id]
__global__ void kernelUpdateWeightHidden(double* input, double* outputL1, double* weightOutput, double* weightHidden, double* biasHidden, double* deltaOutput, double lr, int inputLayer, int hiddenLayer, int outputLayer) {
    int id = threadIdx.x;
    double outputL1Val = outputL1[id];
    double deltaVal;
    /*
    double* product = new double[hiddenLayer * 1];
    for (int i = 0; i < hiddenLayer; i++) {
        product[i] = 0;
        for (int k = 0; k < outputLayer; k++) {
            product[i] += weightOutput[(i * outputLayer) + k] + deltaOutput[k];
        }
    }
    */
    double productVal = 0;
    for (int k = 0; k < outputLayer; k++) {
        productVal += weightOutput[(id * outputLayer) + k] + deltaOutput[k];
    }

    deltaVal = productVal * (outputL1Val * (1.0 - outputL1Val));

    for (int i = 0; i < inputLayer; i++) {
        weightHidden[(i * hiddenLayer) + id] -= (lr * deltaVal * input[i]);
    }
    biasHidden[id] -= (lr * deltaVal * inputLayer);
}

class MLP {
    private:
        int inputLayer;
        int hiddenLayer;
        int outputLayer;
        double lr;
        int maxEpochs;
        double biasHiddenValue;
        double biasOutputValue;

        double* weightHidden; // inputLayer x hiddenLayer
        double* weightOutput; // hiddenLayer x outputLayer
        double* biasHidden; //hiddenLayer x 1
        double* biasOutput; // outputLayer x 1
        int numClasses;

    public:
        MLP (int inputLayer_in,
            int hiddenLayer_in,
            int outputLayer_in,
            double learningRate_in,
            int maxEpochs_in,
            double biasHiddenValue_in,
            double biasOutputValue_in,
            int numClasses_in) 
            {
                inputLayer = inputLayer_in;
                hiddenLayer = hiddenLayer_in;
                outputLayer = outputLayer_in;
                lr = learningRate_in;
                maxEpochs = maxEpochs_in;
                biasHiddenValue = biasHiddenValue_in;
                biasOutputValue = biasOutputValue_in;
                numClasses = numClasses_in;

                //initialize weights
                weightHidden = new double[inputLayer * hiddenLayer]; // inputLayer x hiddenLayer
                weightOutput = new double[hiddenLayer * outputLayer]; // hiddenLayer x outputLayer
                biasHidden = new double[hiddenLayer]; //hiddenLayer x 1
                biasOutput = new double[outputLayer]; // outputLayer x 1

                srand(SEED);
                //init weightHidden
                for (int i = 0; i < inputLayer; i++) {
                    for (int j = 0; j < hiddenLayer; j++) {
                        weightHidden[(i * hiddenLayer) + j] = (2.0 * ((double) rand() / (RAND_MAX))) - 1;
                        // weightHidden[(i * hiddenLayer) + j] = ((double) rand() / (RAND_MAX)) + 1;
                    }
                }

                // init weightOutput
                for (int i = 0; i < hiddenLayer; i++) {
                    for (int j = 0; j < outputLayer; j++) {
                        weightOutput[(i * outputLayer) + j] = (2.0 * ((double) rand() / (RAND_MAX))) - 1;
                        // weightOutput[(i * outputLayer) + j] = ((double) rand() / (RAND_MAX)) + 1;
                    }
                }

                //init biasHidden
                for (int i = 0; i < hiddenLayer; i++) {
                    biasHidden[i] = biasHiddenValue;
                }
                
                //init biasOutput
                for (int i = 0; i < outputLayer; i++) {
                    biasOutput[i] = biasOutputValue;
                }
            }

        ~MLP() {
            delete[] weightHidden; // inputLayer x hiddenLayer
            delete[] weightOutput; // hiddenLayer x outputLayer
            delete[] biasHidden; //hiddenLayer x 1
            delete[] biasOutput; // outputLayer x 1
        }

        //val needs to be outputted by a sigmoid
        double derivativeSingle(double val) {
            return (val * (1.0 - val));
        }

        double sigmoidSingle(double val) {
            return 1.0 / (1.0 + exp(-1.0 * val));
        }
        
        // input = inputLayer x 1
        // output = outputLayer x 1
        // outputL1 = hiddenLayer x 1
        // outputL2 = outputLayer x 1
        void backPropagation(double* input, double* output, double* outputL1, double* outputL2) {
            // Error output layer
            double* deltaOutput = new double[outputLayer];
            for (int i = 0; i < outputLayer; i++) {
                //errorOutput = output - outputL2
                deltaOutput[i] = output[i] - outputL2[i];

                //deltaOutput = -1*errorOutput*deriv(outputL2)
                deltaOutput[i] = (-1.0) * deltaOutput[i] - derivativeSingle(outputL2[i]);
            }

            //update weights outputLayer and hiddenLayer
            for (int i = 0; i < hiddenLayer; i++) {
                for (int j = 0; j < outputLayer; j++) {
                    weightOutput[(i * outputLayer) + j] -= (lr * deltaOutput[j] * outputL1[i]);
                    biasOutput[j] -= (lr * deltaOutput[j]);
                }
            }

            // hidden layer
            // matmul weightOutput x deltaOutput 
            // = (hiddenLayer x outputLayer) x (outputLayer x 1)
            // = hiddenLayer x 1
            double* product = new double[hiddenLayer * 1];
            for (int i = 0; i < hiddenLayer; i++) {
                for (int j = 0; j < 1; j++) {
                    product[(i * 1) + j] = 0;
                    for (int k = 0; k < outputLayer; k++) {
                        product[(i * 1) + j] += weightOutput[(i * outputLayer) + k] + deltaOutput[(k * 1) + j];
                    }
                }
            }

            // product x deriv(outputL1) 
            // = (hiddenLayer x 1) x (hiddenLayer x 1) (element wise)
            double* deltaHidden = new double[hiddenLayer];
            for (int i = 0; i < hiddenLayer; i++) {
                deltaHidden[i] = product[i] * derivativeSingle(outputL1[i]);
            }

            //update weights hidden layer and input layer
            for (int i = 0; i < inputLayer; i++) {
                for (int j = 0; j < hiddenLayer; j++) {
                    weightHidden[(i * hiddenLayer) + j] -= (lr * deltaHidden[j] * input[i]);
                    biasHidden[j] -= (lr * deltaHidden[j]);
                }
            }

            delete[] deltaOutput;
            delete[] product;
            delete[] deltaHidden;            
        }
        
        void fit(double* xtrain, int* ytrain, int numSamples) {
            // double* output = new double[numClasses]; // numClasses = outputLayer
            // double* outputL1 = new double[hiddenLayer];
            // double* outputL2 = new double[outputLayer];

            int* gpuOutput;
            double* gpuOutputL1;
            double* gpuOutputL2;
            double* gpuXtrain;
            double* gpuWeightHidden;
            double* gpuWeightOutput;
            double* gpuBiasHidden;
            double* gpuBiasOutput;
            double* gpuDeltaOutput;

            cudaMalloc((void**)&gpuOutput, sizeof(int)*numClasses);
            cudaMalloc((void**)&gpuOutputL1, sizeof(double)*hiddenLayer);
            cudaMalloc((void**)&gpuOutputL2, sizeof(double)*outputLayer);
            cudaMalloc((void**)&gpuXtrain, sizeof(double)*numSamples*inputLayer);
            cudaMalloc((void**)&gpuWeightHidden, sizeof(double)*inputLayer*hiddenLayer); 
            cudaMalloc((void**)&gpuWeightOutput, sizeof(double)*hiddenLayer*outputLayer); 
            cudaMalloc((void**)&gpuBiasHidden, sizeof(double)*hiddenLayer); 
            cudaMalloc((void**)&gpuBiasOutput, sizeof(double)*outputLayer); 
            cudaMalloc((void**)&gpuDeltaOutput, sizeof(double)*outputLayer); 

            struct timespec start, stop; 
            double time;
            if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
            

            cudaMemcpy(gpuXtrain, xtrain, sizeof(double)*numSamples*inputLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuWeightHidden, weightHidden, sizeof(double)*inputLayer*hiddenLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuWeightOutput, weightOutput, sizeof(double)*hiddenLayer*outputLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuBiasHidden, biasHidden, sizeof(double)*hiddenLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuBiasOutput, biasOutput, sizeof(double)*outputLayer, cudaMemcpyHostToDevice);

            dim3 dimGrid(1);
            dim3 dimBlockHidden(hiddenLayer);
            dim3 dimBlockOutput(outputLayer);

            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                if (epoch % 50 == 0) {
                    cout << "epoch = " << epoch << endl;
                }
                for (int sample = 0; sample < numSamples; sample++) {
                    //Forward propagation
                    // x = &xtrain[sample * inputLayer];
                    kernelOutputL1<<<dimGrid, dimBlockHidden>>>(&gpuXtrain[sample * inputLayer], gpuWeightHidden, gpuBiasHidden, gpuOutputL1, inputLayer, hiddenLayer);
                    kernelOutputL2<<<dimGrid, dimBlockOutput>>>(gpuOutputL1, gpuWeightOutput, gpuBiasOutput, hiddenLayer, outputLayer, ytrain[sample], gpuOutput);

                    //backprop
                    kernelUpdateWeightOutput<<<dimGrid, dimBlockOutput>>>(gpuOutput, gpuOutputL1, gpuOutputL2, gpuWeightOutput, gpuBiasOutput, gpuDeltaOutput, lr, hiddenLayer, outputLayer);
                    kernelUpdateWeightHidden<<<dimGrid, dimBlockHidden>>>(&gpuXtrain[sample * inputLayer], gpuOutputL1, gpuWeightOutput, gpuWeightHidden, gpuBiasHidden, gpuDeltaOutput, lr, inputLayer, hiddenLayer, outputLayer);
                }
            }

            cudaMemcpy(weightHidden, gpuWeightHidden, sizeof(double)*inputLayer*hiddenLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(weightOutput, gpuWeightOutput, sizeof(double)*hiddenLayer*outputLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(biasHidden, gpuBiasHidden, sizeof(double)*hiddenLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(biasOutput, gpuBiasOutput, sizeof(double)*outputLayer, cudaMemcpyDeviceToHost);
            
            if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
            time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
            printf("time is %f sec\n", time);	

            cudaFree(gpuOutput);
            cudaFree(gpuOutputL1);
            cudaFree(gpuOutputL2);
            cudaFree(gpuXtrain);
            cudaFree(gpuWeightHidden); 
            cudaFree(gpuWeightOutput); 
            cudaFree(gpuBiasHidden); 
            cudaFree(gpuBiasOutput); 
            cudaFree(gpuDeltaOutput); 
        }

        //prediction = empty array allocated for size = num * numClasses
        void predict(double* xtest, int* prediction, int num) {
            double* outputL1 = new double[hiddenLayer];
            double* outputL2 = new double[outputLayer];
            double* x;
            int* p;
            for (int sample = 0; sample < num; sample++) {
                //Forward propagation
                x = &xtest[sample * inputLayer];
                p = &prediction[sample * numClasses];
                // find outputL1 = sigmoid(input x weightHidden + biasHidden.T)
                // (input is transposed to 1 x inputLayer, weightHidden = inputLayer x hiddenLayer)
                // input x weightHidden (1 x inputLayer) x (inputLayer x hiddenLayer)
                for (int i = 0; i < hiddenLayer; i++) {
                    outputL1[i] = 0;
                    for (int j = 0; j < inputLayer; j++) {
                        outputL1[i] += x[j] * weightHidden[(j * hiddenLayer) + i];
                    }
                    outputL1[i] = sigmoidSingle(outputL1[i] + biasHidden[i]);
                }

                // find outputL2 = sigmoid(outputL1 x weightOutput + biasOutput.T)
                // outputL1 = (transposed) 1 x hiddenLayer
                // weightOutput = hiddenLayer x outputLayer
                // outputL2 = (1 x hiddenLayer) x (hiddenLayer x outputLayer) = 1 x outputLayer
                for (int i = 0; i < outputLayer; i++) {
                    outputL2[i] = 0;
                    for (int j = 0; j < hiddenLayer; j++) {
                        outputL2[i] += outputL1[j] * weightOutput[(j * outputLayer) + i];
                    }
                    outputL2[i] = sigmoidSingle(outputL2[i] + biasOutput[i]);
                }

                double max = -1;
                int maxId = -1;
                for (int i = 0; i < numClasses; i++) {
                    if (outputL2[i] > max) {
                        max = outputL2[i];
                        maxId = i;
                    }
                }
                
                // one-hot encoding
                // for (int i = 0; i < numClasses; i++) {
                //     output[i] = 0;
                // }
                // output[(int)(y[sample])] = 1;
                if (maxId == 0) {
                    p[0] = 1;
                    p[1] = 0;
                } else {
                    p[0] = 0;
                    p[1] = 1;
                }
                cout << "maxId = " << maxId << ", outputL2[0] = " << outputL2[0] << ", outputL2[1] = " << outputL2[1] << endl;  
            }

            delete[] outputL1;
            delete[] outputL2;
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

int main() {
    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 250;
    double testSize = 0.1;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 2;
    int numHidden = 3;
    int numClasses = 2;
    int biasHiddenValue = -1;
    int biasOutputValue = -1;

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    // double iters = 1000;
    double iters = 20;
    
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

    double* Xtrain1D = new double[numTrain * numFeatures];
    double* Xtest1D = new double[numTest * numFeatures];
    
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
        double xData1;
        double xData2;
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
            Xtrain[idx][0] = xData1;
            Xtrain[idx][1] = xData2;
            ytrain[idx] = cls;
        } else {
            Xtest[idx][0] = xData1;
            Xtest[idx][1] = xData2;
            ytest[idx] = cls;
        }

        line = "";
        total++;
        idx++;

    }

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
    
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

    MLP classifier = MLP(numFeatures, numHidden, numClasses, learningRate, iters, 
                            biasHiddenValue, biasOutputValue, numClasses);
    
    classifier.fit(Xtrain1D, ytrain, numTrain);
    cout << "classifier trained" << endl;

    int* predictions = new int[numTest * numClasses];
    classifier.predict(Xtest1D, predictions, numTest);
    cout << "predictions completed" << endl;

    int* predictedLabels = new int[numTest];
    for (int i = 0; i < numTest; i++) {
        if ((predictions[(i * numClasses) + 0] == 1) && (predictions[(i * numClasses) + 1] == 0)) {
            predictedLabels[i] = 0;
        } else if ((predictions[(i * numClasses) + 0] == 0) && (predictions[(i * numClasses) + 1] == 1)) {
            predictedLabels[i] = 1;
        } else {
            cout << "YIKES! p[0] = " << predictions[(i * numClasses) + 0] << ", p[1] = " << predictions[(i * numClasses) + 1] << endl;
        }
    }

    double acc = accuracy(ytest, predictedLabels, numTest);
    printf("MLP Accuracy: %f\n", acc);

    delete[] predictions;
    delete[] predictedLabels;
    delete[] Xtrain1D;
    delete[] Xtest1D;
    
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
}