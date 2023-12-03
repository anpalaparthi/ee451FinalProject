// https://medium.com/geekculture/logistic-regression-from-scratch-59e88bea2ba2
#include <stdlib.h>
#include <stdio.h>
// #include <cublas.h>
#include <time.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;

#define NUM_CLASSES 10
#define NUM_PIXELS 784
#define MNIST_MEAN 0.1307
#define MNIST_STD 0.3081
#define MNIST_MAX 255.0

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

        void weightedSigmoid(double** X, int numRowsX, int lenW, double* result, bool test) {
            // denominator = 1 + np.exp(-(X * w).sum(axis=1))
            matVecMul(X, w, numRowsX, lenW, result);
            for (int i = 0; i < numRowsX; i++) {
                // if (test) {
                // cout << "result before i=" << i << ": " << result[i] << endl;}
                result[i] = 1.0 / (1.0 + exp(-1.0 * result[i]));
            //     if (test) {
            //     // cout << "result after i=" << i << ": " << result[i] << endl;}
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
            numIters = 0;
            tolerance = 0;
            w = NULL;
        }

        ~LogisticRegression() {
            if (w != NULL) {
                delete[] w;
            }
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
                weightedSigmoid(Xtrain, numRowsX, numFeatures, probs, false);
                for (int j = 0; j < numRowsX; j++) {
                    for (int k = 0; k < numFeatures; k++) {
                        gradW[k] += ((probs[j] - y[j]) * Xtrain[j][k]);
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

        // now we return the raw probabilities and handle class selection in accuracy
        void predict(double** Xtest, int numRowsX, int numFeatures, double* result) {
            // apply trained weighted sigmoid to get probabilities
            
            // cout << "weights before we predict";
            // for (int i = 0; i < numFeatures; i++) {
            //     cout << "weights: " << w[i] << ", ";
            // }
            weightedSigmoid(Xtest, numRowsX, numFeatures, result, true);
        }
};

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

// both yTrue and yPred have size elements, predictions = NUM_CLASS double* values (each double* = array of length 'size')
double accuracy(int* yTrue, double* predictions[], int size) {
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

int main() {

    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 42000;
    double testSize = 0.3;
    double trainSize = 0.2;
    int numTrain = trainSize * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = NUM_PIXELS + 1; // add a bias term

    cout << "numTrain = " << numTrain << endl;
    cout << "numTest = " << numTest << endl; 

    // Logistic Regression hyperparameters
    double etaParam = 0.001; //1e-3
    double numItersParam = 1000;
    double toleranceParam = 0; // we're not using this 

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

    LogisticRegression classifiers[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        classifiers[i] = LogisticRegression(etaParam, numItersParam, toleranceParam);
    }

    double* predictions[NUM_CLASSES];
    int* yTrainOVR = new int[numTrain];
    int* yTestOVR = new int[numTest];
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "training Logistic Regression " << i << "..." << endl;
        
        // convert ytrain and ytest to be one-vs-all
        for (int j = 0; j < numTrain; j++) {
            yTrainOVR[j] = (ytrain[j] == i) ? 1 : 0;
        }

        for (int j = 0; j < numTest; j++) {
            yTestOVR[j] = (ytest[j] == i) ? 1 : 0;
        }

        classifiers[i].fit(Xtrain, yTrainOVR, numTrain, numFeatures);
        cout << "predicting Logistic Regression " << i << "..." << endl;

        double* prediction = new double[numTest];
        classifiers[i].predict(Xtest, numTest, numFeatures, prediction);
        // for (int k = 0; k < numTest; k++) {
        //     cout << "probs out classifier i=" << i << ": " << prediction[k] << ", " << endl;
        // }
        predictions[i] = prediction;
        
    }

    delete[] yTrainOVR;
    delete[] yTestOVR;

    
    
    cout << "classifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("Logistic Regression Accuracy: %f\n", acc);
    //DELETE PREDICTIONS    
    for (int i = 0; i < NUM_CLASSES; i++) {
        delete[] predictions[i];
    }

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

    cout << "done" << endl;
    
}