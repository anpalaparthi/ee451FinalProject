// https://towardsdatascience.com/implementing-svm-from-scratch-784e4ad0bc6a
// Implementation based off of python SVM techniques from this medium article

// Basic setup/initialization of weights and biases (w, b)
// Map the class labels from {0,1} to {-1,1}
// Perform gradient descent for n iterations, which involves the computation
//      of gradients and updated the weights an biases accordingly
// Make the final prediction

#include <stdlib.h>
#include <stdio.h>
// #include <cublas.h>
#include <time.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class SVM {

    private:
        double lr;
        double lambdaParam;
        int numIters;
        double* w;
        int lenW;
        double b;
        double* clsMap;

            
        void initWeightsBias(int numFeatures) { //self, X
            w = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                w[i] = 0;
            }
            lenW = numFeatures;
            b = 0;
        }

        void getClsMap(int* y, int size) { //self, y
            if (clsMap == NULL) {
                clsMap = new double[size];
            }
            for (int i = 0; i < size; i++) {
                if (y[i] == 0) {
                    clsMap[i] = -1;
                } else {
                    clsMap[i] = 1;
                }
            }
        }

        // a and b must be the same length
        double dotProduct(double* a, double* b, int size) {
            double sum = 0;
            for (int i = 0; i < size; i++) {
                sum += a[i] * b[i];
            }
            return sum;
        }

        bool satisfyConstraint(double* x, int idx) { //self, x, idx
            // dot product linearModel = x * w + b
            double linearModel = dotProduct(x, w, lenW);
            linearModel += b;

            if ( (clsMap[idx] * linearModel) >= 1) {
                return true;
            }
            return false;
        }

        // FREE dw
        void getGradients(bool constraint, double* x, int idx, double*& dw, double& db) { //self, constrain, x, idx
            dw = new double[lenW];
            if (constraint) {
                for (int i = 0; i < lenW; i++) {
                    dw[i] = (w[i] * lambdaParam);
                }
                db = 0;
                return;
            }

            for (int i = 0; i < lenW; i++) {
                dw[i] = (lambdaParam * w[i]) - (clsMap[idx] * x[i]);
            }
            db = -clsMap[idx];
        }

        void updateWeights(double* dw, double db) { //self, dw, db
            for (int i = 0; i < lenW; i++) {
                w[i] -= lr * dw[i];
            }
            b -= lr * db;
        } 

    public:
        SVM(double learningRate, double lamba, double iters) {
            lr = learningRate;
            lambdaParam = lamba;
            numIters = iters;
            w = NULL;
            b = 0;
            clsMap = NULL;
        }

        ~SVM() {
            if (w != NULL) {
                printf("free memory w\n");
                delete[] w;
            }
            if (clsMap != NULL) {
                printf("free memory clsMap\n");
                delete[] clsMap;
            }
        }

        //size = length of y, length of X (X dims = [size][numFeatures])
        void fit(double** X, int numFeatures, int* y, int size) { //self, X, y
            initWeightsBias(numFeatures);
            getClsMap(y, size); // updates clsMap

            for (int i = 0; i < numIters; i++) {
                // numIters training iterations

                for (int j = 0; j < size; j++) {
                    bool constraint = satisfyConstraint(X[j], j);
                    double* dw = NULL;
                    double db = 0;
                    getGradients(constraint, X[j], j, dw, db);
                    updateWeights(dw, db);

                    //Free dw
                    delete[] dw;
                }
            }
        }

        //X is of dims [size][numFeatures]
        // FREE estimate
        int* predict(double** X, int size) { //self, X
            int* estimate = new int[size]; //hold dot products
            for (int i = 0; i < size; i++) {
                estimate[i] = dotProduct(X[i], w, lenW);
                estimate[i] += b;
            }

            //predict class labels
            for (int i = 0; i < size; i++) {
                if (estimate[i] < 0) {
                    estimate[i] = 0;
                } else {
                    estimate[i] = 1;
                }
            }
            return estimate;
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
    int numSamples = 891;
    double testSize = 0.25;
    double trainSize = 0.75;
    int numTrain = ((trainSize) * numSamples) + 1;
    int numTest = testSize * numSamples;
    int numFeatures = 8;

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    double lamba = 0.01; //1e-2 
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

    SVM classifier = SVM(learningRate, lamba, iters);
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	
    classifier.fit(Xtrain, numFeatures, ytrain, numTrain);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    int* predictions = classifier.predict(Xtest, numTest);
    
    
    cout << "cassifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("SVM Accuracy: %f\n", acc);
    printf("Training Execution Time: %f sec\n", time);

    delete[] predictions;

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