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

#define SEED 42

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
    int numSamples = 250;
    double testSize = 0.1;
    int numTrain = (1 - testSize) * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = 3; // 2 + 1 bias

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
        // cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        // cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
        if (total < numTrain) {
            Xtrain[idx][0] = xData1;
            Xtrain[idx][1] = xData2;
            // add bias term
            Xtrain[idx][2] = 1.0;
            ytrain[idx] = cls;
        } else {
            Xtest[idx][0] = xData1;
            Xtest[idx][1] = xData2;
            // add bias term
            Xtest[idx][2] = 1.0;
            ytest[idx] = cls;
        }

        line = "";
        total++;
        idx++;

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

    // gradient descent
    for (int i = 0; i < numIters; i++) {
        // calculate gradient -> (y^ - y)x_j
        //weightedSigmoid
        printf("iter = %d, w[0] = %f, w[1] = %f\n", i, w[0], w[1]);
        //matrix-vector multiplication
        for (int i = 0; i < numTrain; i++) {
            probs[i] = 0;
            for (int j = 0; j < numFeatures; j++) {
                // probs[i] += Xtrain[i][j] * w[j];
                probs[i] += Xtrain1D[(i * numFeatures) + j] * w[j];
            }
        }

        for (int i = 0; i < numTrain; i++) {
            probs[i] = 1.0 / (1.0 + exp(-1.0 * probs[i]));
        }

        //calculate weight gradient
        // for (int j = 0; j < numTrain; j++) {
        //     for (int k = 0; k < numFeatures; k++) {
        //         // gradW[k] += (probs[j] - ytrain[j]) * Xtrain[j][k];
        //         gradW[k] += (probs[j] - ytrain[j]) * Xtrain1D[(j * numFeatures) + k];
        //     }
        // }

        for (int k = 0; k < numFeatures; k++) {
            for (int j = 0; j < numTrain; j++) {
                // gradW[k] += (probs[j] - ytrain[j]) * Xtrain[j][k];
                gradW[k] += (probs[j] - ytrain[j]) * Xtrain1D[(j * numFeatures) + k];
            }
        }

        // update rule
        for (int j = 0; j < numFeatures; j++) {
            w[j] -= eta * gradW[j];
        }
 
        for (int j = 0; j < numFeatures; j++) {
            printf("id = %f, val = %f\n", j, gradW[j]);
        }

        // break tolerance (optional, write later)
    }

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

    cout << "done" << endl;
}