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

const int NUM_PIXELS = 7500;

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
            cout << "weight: " << w[0] << ", " << w[34] << ", " << w[1450] << ", " << w[2354] << ", " << w[6458] << endl; 

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

void parseMNISTData(string dataFileStr, int numTrain, int numTest, double** Xtrain, int* ytrain, double** Xtest, int* ytest) {
    ifstream inputFile;
    inputFile.open(dataFileStr);
    cout << "open file" << endl;
    
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
        // cout << "total = " << total << " | numTrain = " << numTrain << " | numTest = " << numTest << " | idx = " << idx << endl;
        // cout << "xData1 = " << xData1 << " | xData2 = " << xData2 << " | cls = " << cls << endl;
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
        
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

}

int main() {

    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 40000;
    double testSize = 0.3;
    double trainSize = 0.2;
    int numTrain = trainSize * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = NUM_PIXELS;

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    double lamba = 0.01; //1e-2 
    // double iters = 1000; 
    double iters = 100;
    
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

    SVM classifier = SVM(learningRate, lamba, iters);
    struct timespec start, stop; 
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	
    classifier.fit(Xtrain, numFeatures, ytrain, numTrain);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    int* predictions = classifier.predict(Xtest, numTest);
    
    
    cout << "classifier trained " << endl;
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