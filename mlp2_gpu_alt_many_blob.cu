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
#define HIDDEN_LAYER 268 //CHANGE
#define OUTPUT_LAYER 2
#define LR 0.0001

#define NUM_PIXELS 400 // CHANGE

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

//hiddenLayer > outputLayer, so hiddenLayer number of threads
__global__ void kernel(int maxEpochs, int numSamples, int inputLayer, int hiddenLayer, int outputLayer, double* xtrain, double* weightHidden, 
                        double* biasHidden, double* weightOutput, double* biasOutput, int* y) {
    int id = threadIdx.x;       
    __shared__ double outputL1[HIDDEN_LAYER];
    __shared__ double deltaOutput[OUTPUT_LAYER];
    double outputL2Id;
    double outputVal;
    double* x;

    double lr = LR;

    // if (id == 0) {
    //     printf("id = %d, maxEpochs = %d, numSamples = %d, inputLayer = %d, hiddenLayer = %d, outputLayer = %d, lr = %f\n", 
    //             id, maxEpochs, numSamples, inputLayer, hiddenLayer, outputLayer, lr);
    // }
    
    for (int epoch = 0; epoch < maxEpochs; epoch++) {
        // if ((id == 0) && (epoch % 100 == 0)) {
        //     printf("epoch = %d\n", epoch);
        // }
        for (int sample = 0; sample < numSamples; sample++) {
            //Forward propagation
            x = &xtrain[sample * inputLayer];
            //kernelOutputL1 = hiddenLayers, all threads
                                   
            double val1 = 0;
            for (int i = 0; i < inputLayer; i++) {
                val1 = val1 + ((1.0 * x[i]) * weightHidden[(i * hiddenLayer) + id]);
            }
            val1 = val1 + biasHidden[id];
            outputL1[id] = 1.0 / (1.0 + exp(-1.0 * val1));
            __syncthreads();

            // outputLayer, some threads
            if (id < outputLayer) {
                //kernelOutputL2
                double val2 = 0;
                for (int i = 0; i < hiddenLayer; i++) {
                    val2 = val2 + (outputL1[i] * weightOutput[(i * outputLayer) + id]);
                }
                val2 = val2 + biasOutput[id];
                outputL2Id = 1.0 / (1.0 + exp(-1.0 * val2));

                if (id == y[sample]) {
                    outputVal = 1;
                    // printf("id = %d, output = 1\n", id);
                } else {
                    outputVal = 0;
                    // printf("id = %d, output = 0\n", id);
                }
                // printf("id = %d, output[%d] = %d\n", id, id, output[id]);
                // __syncthreads();

                //kernelUpdateWeightOutput
                double outputL2Val = outputL2Id;
                double deltaVal = (1.0 * outputVal) - outputL2Val;
                deltaVal = (-1.0) * deltaVal - (outputL2Val * (1.0 - outputL2Val));

                for (int i = 0; i < hiddenLayer; i++) {
                    weightOutput[(i * outputLayer) + id] = weightOutput[(i * outputLayer) + id] - (lr * deltaVal * outputL1[i]);
                }
                biasOutput[id] -= (lr * deltaVal * hiddenLayer);
                // biasOutput[id] = biasOutput[id] - (lr * deltaVal);
                deltaOutput[id] = deltaVal;
            }
            __syncthreads();
            
            //kernelUpdateWeightHidden = hiddenLayers, all threads
            double outputL1Val = outputL1[id];
            double deltaValHidden;
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
                productVal = productVal + (weightOutput[(id * outputLayer) + k] + deltaOutput[k]);
            }

            deltaValHidden = productVal * (outputL1Val * (1.0 - outputL1Val));

            for (int i = 0; i < inputLayer; i++) {
                weightHidden[(i * hiddenLayer) + id] = weightHidden[(i * hiddenLayer) + id] - (lr * deltaValHidden * x[i]);
            }
            biasHidden[id] -= (lr * deltaValHidden * inputLayer);
            // biasHidden[id] = biasHidden[id] - (lr * deltaValHidden);
            __syncthreads();
        }
    }
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
                        // weightHidden[(i * hiddenLayer) + j] = ((double) rand() / (RAND_MAX)) - 1;
                    }
                }

                // init weightOutput
                for (int i = 0; i < hiddenLayer; i++) {
                    for (int j = 0; j < outputLayer; j++) {
                        weightOutput[(i * outputLayer) + j] = (2.0 * ((double) rand() / (RAND_MAX))) - 1;
                        // weightOutput[(i * outputLayer) + j] = ((double) rand() / (RAND_MAX)) - 1;
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

                //print weightHidden
                // for (int i = 0; i < inputLayer; i++) {
                //     for (int j = 0; j < hiddenLayer; j++) {
                //         cout << weightHidden[(i * hiddenLayer) + j] << endl;
                //     }
                // }

                // // init weightOutput
                // for (int i = 0; i < hiddenLayer; i++) {
                //     for (int j = 0; j < outputLayer; j++) {
                //         cout << weightOutput[(i * outputLayer) + j] << endl;
                //     }
                // }
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

            double* gpuX;
            double* gpuWeightHidden;
            double* gpuBiasHidden;
            double* gpuWeightOutput;
            double* gpuBiasOutput;
            int* gpuY;


            cudaMalloc((void**)&gpuX, sizeof(double)*numSamples*inputLayer);
            cudaMalloc((void**)&gpuWeightHidden, sizeof(double)*inputLayer*hiddenLayer); 
            cudaMalloc((void**)&gpuWeightOutput, sizeof(double)*hiddenLayer*outputLayer); 
            cudaMalloc((void**)&gpuBiasHidden, sizeof(double)*hiddenLayer); 
            cudaMalloc((void**)&gpuBiasOutput, sizeof(double)*outputLayer); 
            cudaMalloc((void**)&gpuY, sizeof(int)*numSamples); 

            struct timespec start, stop; 
            double time;
            if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

            cudaMemcpy(gpuX, xtrain, sizeof(double)*numSamples*inputLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuWeightHidden, weightHidden, sizeof(double)*inputLayer*hiddenLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuWeightOutput, weightOutput, sizeof(double)*hiddenLayer*outputLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuBiasHidden, biasHidden, sizeof(double)*hiddenLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuBiasOutput, biasOutput, sizeof(double)*outputLayer, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuY, ytrain, sizeof(int)*numSamples, cudaMemcpyHostToDevice);

            dim3 dimGrid(1);
            dim3 dimBlock(hiddenLayer);

            // __global__ void kernel(int maxEpochs, int numSamples, int inputLayer, int hiddenLayer, int outputLayer, int lr, double* xtrain, double* weightHidden, 
            //             double* biasHidden, double* weightOutput, double* biasOutput, int* y) {
            kernel<<<dimGrid, dimBlock>>>(maxEpochs, numSamples, inputLayer, hiddenLayer, outputLayer, gpuX, 
                                            gpuWeightHidden, gpuBiasHidden, gpuWeightOutput, gpuBiasOutput, gpuY);

            cudaMemcpy(weightHidden, gpuWeightHidden, sizeof(double)*inputLayer*hiddenLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(weightOutput, gpuWeightOutput, sizeof(double)*hiddenLayer*outputLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(biasHidden, gpuBiasHidden, sizeof(double)*hiddenLayer, cudaMemcpyDeviceToHost);
            cudaMemcpy(biasOutput, gpuBiasOutput, sizeof(double)*outputLayer, cudaMemcpyDeviceToHost);
            
            if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
            time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
            printf("time is %f sec\n", time);	

            cudaFree(gpuX);
            cudaFree(gpuWeightHidden);
            cudaFree(gpuBiasHidden);
            cudaFree(gpuWeightOutput);
            cudaFree(gpuBiasOutput);
            cudaFree(gpuY); 
        }

        //prediction = empty array allocated for size = num * numClasses
        /*
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
        */
       void predict(double* xtest, int* prediction, int num) {
            //forward propogation
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
                if (sample % 2 == 0) {
                    cout << "outputL1 pre sigmoid: " ;
                }
                for (int i = 0; i < hiddenLayer; i++) {
                    outputL1[i] = 0;
                    for (int j = 0; j < inputLayer; j++) {
                        outputL1[i] += x[j] * weightHidden[(j * hiddenLayer) + i];
                    }
                    if (sample % 2 == 0) {
                        cout << outputL1[i] << " ";
                    }
                    outputL1[i] = sigmoidSingle(outputL1[i] + biasHidden[i]);
                }
                if (sample % 2 == 0) {
                    cout << endl;
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
                for (int i = 0; i < outputLayer; i++) {
                    if (outputL2[i] > max) {
                        max = outputL2[i];
                        maxId = i;
                    }
                }
                
                // one-hot encoding
                // for (int i = 0; i < outputLayer; i++) {
                //     p[i] = 0;
                // }
                // p[maxId] = 1;
                if (maxId == 0) {
                    p[0] = 1;
                    p[1] = 0;
                } else {
                    p[0] = 0;
                    p[1] = 1;
                }
                
                
                if (sample % 2 == 0) {
                    cout << "x: ";
                    for (int i = 0; i < inputLayer; i++) {
                        cout << x[i] << " " ;
                    }
                    cout << endl;
                    cout << "maxId = " << maxId;
                    for (int i = 0; i < outputLayer; i++) {
                        cout << ", outputL2[" << i << "] = " << outputL2[i];
                    }
                    cout << endl;  
                    cout << "outputL1: " ;
                    for (int i = 0; i < hiddenLayer; i++) {
                        cout << outputL1[i] << " ";
                    }
                    cout << endl;
                    cout << "biasHidden: " ;
                    for (int i = 0; i < hiddenLayer; i++) {
                        cout << biasHidden[i] << " ";
                    }
                    cout << endl;
                    cout << "biasOutput: " ;
                    for (int i = 0; i < outputLayer; i++) {
                        cout << biasOutput[i] << " ";
                    }
                    cout << endl;
                    cout << "weightHidden: " ;
                    for (int i = 0; i < (inputLayer * hiddenLayer); i++) {
                        cout << weightHidden[i] << " ";
                    }
                    cout << endl;  
                    cout << "weightOutput: " ;
                    for (int i = 0; i < (hiddenLayer * outputLayer); i++) {
                        cout << weightOutput[i] << " ";
                    }
                    cout << endl;  
                }
                
                
            }

            delete[] outputL1;
            delete[] outputL2;
        }
};


// both yTrue and yPred have size elements
double accuracy(int* yTrue, int* yPred, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        if (i % 100 == 0) {
            cout << "yTrue " << yTrue[i] << " | " << "yPred " << yPred[i] << endl;
        }
        if (yTrue[i] == yPred[i]) {
            sum++;
        }
    }
    return (sum / (1.0 * size));
}

void parseMNISTData(string dataFileStr, int numTrain, int numTest, double** Xtrain, int* ytrain, double** Xtest, int* ytest) {
    ifstream inputFile;
    inputFile.open(dataFileStr);
    // // cout << "open file" << endl;
    
    string line = "";
    int total = 0;
    bool flag = true;
    int idx = 0;
    while (getline(inputFile, line)) {
        // if (flag) {
        //     flag = false;
        //     continue;
        // }
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
        // // // cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        // // // cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
        if (total < numTrain) {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtrain[idx][i] = pixels[i];
            }
            ytrain[idx] = label;
        } else {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtest[idx][i] = pixels[i];
            }
            ytest[idx] = label;
        }

        line = "";
        total++;
        idx++;

        if (total == (numTrain + numTest)) {
            break;
        }

    }
        
    // // cout << "file read" << endl;
    inputFile.close();
    // // cout << "file closed" << endl;

}


int main() {
    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 20000; // CHANGE!
    double testSize = 0.1;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 400; // CHANGE!
    // int numHidden = 3;
    int numHidden = 268;
    int numClasses = 2;
    int biasHiddenValue = -1;
    int biasOutputValue = -1;

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    // double iters = 1000;
    double iters = 1000;
    
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
    string dataFileStr = "blob_400d.csv"; // CHANGE!

    if (dataFileStr == "blob_400d.csv") { // CHANGE!
        parseMNISTData(dataFileStr, numTrain, numTest, Xtrain, ytrain, Xtest, ytest);
    } else {
        // cout << "File " << dataFileStr << " not supported" << endl;
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