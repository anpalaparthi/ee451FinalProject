// Based on python tutorial
// https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch


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

#define SEED 100

#define NUM_PIXELS 700 // CHANGE

using namespace std;

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
                
                //print weightHidden
                for (int i = 0; i < inputLayer; i++) {
                    for (int j = 0; j < hiddenLayer; j++) {
                        // cout << weightHidden[(i * hiddenLayer) + j] << endl;
                    }
                }

                // init weightOutput
                for (int i = 0; i < hiddenLayer; i++) {
                    for (int j = 0; j < outputLayer; j++) {
                        // cout << weightOutput[(i * outputLayer) + j] << endl;
                    }
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
        void backPropagation(double* input, int* output, double* outputL1, double* outputL2) {
            // Error output layer
            double* deltaOutput = new double[outputLayer];
            for (int i = 0; i < outputLayer; i++) {
                //errorOutput = output - outputL2
                deltaOutput[i] = (1.0 * output[i]) - outputL2[i];

                //deltaOutput = -1*errorOutput*deriv(outputL2)
                deltaOutput[i] = (-1.0) * deltaOutput[i] - derivativeSingle(outputL2[i]);
            }

            //update weights outputLayer and hiddenLayer
            for (int i = 0; i < hiddenLayer; i++) {
                for (int j = 0; j < outputLayer; j++) {
                    weightOutput[(i * outputLayer) + j] -= (lr * deltaOutput[j] * outputL1[i]);
                    // biasOutput[j] -= (lr * deltaOutput[j]);
                }
            }

            for (int j = 0; j < outputLayer; j++) {
                biasOutput[j] -= (lr * deltaOutput[j]);
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
                }
            }

            for (int j = 0; j < hiddenLayer; j++) {
                biasHidden[j] -= (lr * deltaHidden[j]);
            }

            delete[] deltaOutput;
            delete[] product;
            delete[] deltaHidden;            
        }
        
        void fit(double* xtrain, int* ytrain, int numSamples) {
            int* output = new int[numClasses]; // numClasses = outputLayer
            double* outputL1 = new double[hiddenLayer];
            double* outputL2 = new double[outputLayer];
            double* x;
            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                if (epoch % 50 == 0) {
                    cout << "epoch = " << epoch << endl;
                }
                for (int sample = 0; sample < numSamples; sample++) {
                    //Forward propagation
                    x = &xtrain[sample * inputLayer];
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

                    // one-hot encoding
                    // for (int i = 0; i < numClasses; i++) {
                    //     output[i] = 0;
                    // }
                    // output[(int)(y[sample])] = 1;
                    if (ytrain[sample] == 0) {
                        output[0] = 1;
                        output[1] = 0;
                    } else {
                        output[0] = 0;
                        output[1] = 1;
                    }

                    //backprop
                    backPropagation(x, output, outputL1, outputL2);
                }
            }

            delete[] output;
            delete[] outputL1;
            delete[] outputL2;
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
                if (sample % 2 == 0) {
                    // cout << "outputL1 pre sigmoid: " ;
                }
                for (int i = 0; i < hiddenLayer; i++) {
                    outputL1[i] = 0;
                    for (int j = 0; j < inputLayer; j++) {
                        outputL1[i] += x[j] * weightHidden[(j * hiddenLayer) + i];
                    }
                    if (sample % 2 == 0) {
                        // cout << outputL1[i] << " ";
                    }
                    outputL1[i] = sigmoidSingle(outputL1[i] + biasHidden[i]);
                }
                if (sample % 2 == 0) {
                    // cout << endl;
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
                if (sample % 2 == 0) {
                    // cout << "x: ";
                    for (int i = 0; i < inputLayer; i++) {
                        // cout << x[i] << " " ;
                    }
                    // cout << endl;
                    // cout << "maxId = " << maxId;
                    for (int i = 0; i < outputLayer; i++) {
                        // cout << ", outputL2[" << i << "] = " << outputL2[i];
                    }
                    // cout << endl;  
                    // cout << "outputL1: " ;
                    for (int i = 0; i < hiddenLayer; i++) {
                        // cout << outputL1[i] << " ";
                    }
                    // cout << endl;
                    // cout << "biasHidden: " ;
                    for (int i = 0; i < hiddenLayer; i++) {
                        // cout << biasHidden[i] << " ";
                    }
                    // cout << endl;
                    // cout << "biasOutput: " ;
                    for (int i = 0; i < outputLayer; i++) {
                        // cout << biasOutput[i] << " ";
                    }
                    // cout << endl;
                    // cout << "weightHidden: " ;
                    // for (int i = 0; i < outputLayer; i++) {
                        // cout << weightHidden[0] << " " << weightHidden[3]<< " " << weightHidden[4]<< " " << weightHidden[5]<< " " << weightHidden[6] << endl;
                    // }
                    // cout << endl;  
                    // cout << "weightOutput: " ;
                    // for (int i = 0; i < outputLayer; i++) {
                        // cout << weightOutput[0] << " " << weightOutput[3]<< " " << weightOutput[4]<< " " << weightOutput[5]<< " " << weightOutput[6] << endl;
                    // }
                    // cout << endl;  
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
        // cout << "yTrue " << yTrue[i] << " | " << "yPred " << yPred[i] << endl;
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
    // cout << "start" << endl;
    // training/test data parameters
    int numSamples = 20000; // CHANGE!
    double testSize = 0.1;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 700; // CHANGE!
    // int numHidden = 3;
    int numHidden = 468;
    int numClasses = 2;
    int biasHiddenValue = -1;
    int biasOutputValue = -1;

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    // double iters = 1000;
    double iters = 10;
    
    // cout << "defined params" << endl;

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
    
    // cout << "finished allocation" << endl;

    // read from csv: https://www.youtube.com/watch?v=NFvxA-57LLA
    string dataFileStr = "blob_700d.csv"; // CHANGE!

    if (dataFileStr == "blob_700d.csv") { // CHANGE!
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
    
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	
    classifier.fit(Xtrain1D, ytrain, numTrain);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    // cout << "classifier trained" << endl;
    printf("Training Execution Time: %f sec\n", time);


    int* predictions = new int[numTest * numClasses];
    classifier.predict(Xtest1D, predictions, numTest);
    // cout << "predictions completed" << endl;

    int* predictedLabels = new int[numTest];
    for (int i = 0; i < numTest; i++) {
        if ((predictions[(i * numClasses) + 0] == 1) && (predictions[(i * numClasses) + 1] == 0)) {
            predictedLabels[i] = 0;
        } else if ((predictions[(i * numClasses) + 0] == 0) && (predictions[(i * numClasses) + 1] == 1)) {
            predictedLabels[i] = 1;
        } else {
            // cout << "YIKES! p[0] = " << predictions[(i * numClasses) + 0] << ", p[1] = " << predictions[(i * numClasses) + 1] << endl;
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