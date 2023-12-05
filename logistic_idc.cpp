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

const int NUM_PIXELS = 7500;

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

            // for (int i = 0; i < numRowsX; i++) {
            //     cout << "probs i=" << i << ": " << probs[i] << endl;
            // }

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
                // Xtrain[idx][i] /= MNIST_MAX;
                // Xtrain[idx][i] -= MNIST_MEAN;
                // Xtrain[idx][i] /= MNIST_STD;

            }
            Xtrain[idx][NUM_PIXELS] = 1.0; // add a bias term
            ytrain[idx] = label;
        } else {
            for (int i = 0; i < NUM_PIXELS; i++) {
                Xtest[idx][i] = pixels[i];
                // Xtest[idx][i] /= MNIST_MAX;
                // Xtest[idx][i] -= MNIST_MEAN;
                // Xtest[idx][i] /= MNIST_STD;
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
    int numSamples = 40000;
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
    string dataFileStr = "idc_dataset40k_shuffled.csv";

    if (dataFileStr == "idc_dataset40k_shuffled.csv") {
        parseMNISTData(dataFileStr, numTrain, numTest, Xtrain, ytrain, Xtest, ytest);
    } else {
        cout << "File " << dataFileStr << " not supported" << endl;
    }

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