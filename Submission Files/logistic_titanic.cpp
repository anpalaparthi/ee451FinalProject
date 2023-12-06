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

#define SEED 42

using namespace std;

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
            generator.seed(SEED);
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
                weightedSigmoid(Xtrain, numRowsX, numFeatures, probs);
                for (int j = 0; j < numRowsX; j++) {
                    for (int k = 0; k < numFeatures; k++) {
                        gradW[k] += ((probs[j] - y[j]) * Xtrain[j][k]);
                    }
                }

                // update rule
                for (int j = 0; j < numFeatures; j++) {
                    w[j] -= eta * gradW[j];
                    
                }
                cout << endl;

                // break tolerance (optional, write later)
            }

            // free gradW, probs
            delete[] gradW;
            delete[] probs;
        }

        // free probs
        void predict(double** Xtest, int numRowsX, int numFeatures, int* result) {
            cout << "weight after fit: " << endl;
            for (int i = 0; i < 9; i++) {
                cout << w[i] << ", ";
            }
            cout << endl;
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

            // for (int i = 0; i < numRowsX; i++) {
            //     cout << "result i=" << i << ": " << result[i] << endl;
            // }

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
    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 891;
    double testSize = 0.25;
    double trainSize = 0.75;
    int numTrain = ((trainSize) * numSamples) + 1;
    int numTest = testSize * numSamples;
    int numFeatures = 9; // 8 + bias

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

    // define the model
    LogisticRegression classifier = LogisticRegression(etaParam, numItersParam, toleranceParam);

    cout << "model defined" << endl;

    // fit
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	
    classifier.fit(Xtrain, ytrain, numTrain, numFeatures);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    cout << "model fit" << endl;

    // predict
    int* yHatTest = new int[numTest];
    classifier.predict(Xtest, numTest, numFeatures, yHatTest);

    // accuracy
    cout << "classifier trained " << endl;
    double acc = accuracy(ytest, yHatTest, numTest);
    printf("Logistic Regression Accuracy: %f\n", acc);
    printf("Training Execution Time: %f sec\n", time);

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

    cout << "done" << endl;
}