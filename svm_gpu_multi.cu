// https://towardsdatascience.com/implementing-svm-from-scratch-784e4ad0bc6a
// Implementation based off of python SVM techniques from this medium article

// Basic setup/initialization of weights and biases (w, b)
// Map the class labels from {0,1} to {-1,1}
// Perform gradient descent for n iterations, which involves the computation
//      of gradients and updated the weights an biases accordingly
// Make the final prediction

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <float.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

#define NUM_CLASSES 10
#define NUM_PIXELS 784
#define MORE_PIXELS 1024
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


//NUM_PIXELS = lenW = 784

__global__ void kernel(int numIters, int size, int lenW, double* w, double* b, double* cls, double* X, double lr, double lambdaParam, int digit) {

    int id = threadIdx.x;
    __shared__ double myW[NUM_PIXELS];
    __shared__ bool constraint;
    __shared__ double myB;
    __shared__ double x[NUM_PIXELS];

    myW[id] = 0;
    myB = 0;
    // if (id == 0) {
    printf("id = %d, digit = %d, myB = %f\n", id, digit, myB);
    // }

    for (int i = 0; i < 8400; i++) {
        if (i % 1000 == 0) {
            printf("id = %d, i = %d, X[i] = %f\n", id, i, X[i]);
        }
    }

    double result = 0;
    double xVal = 0;
    for (int i = 0; i < numIters; i++) {
        for (int j = 0; j < size; j++) {
            //check constraint: dotProduct with recursive doubling
            
            xVal = X[(j * NUM_PIXELS) + id];

            // xVal = X[(j * BAD_PIXELS) + id];
            x[id] = xVal * myW[id];
            __syncthreads();
            /*
            for (int k = 2; k <= MORE_PIXELS; k *= 2) {
            // for (int k = 2; k <= BAD_PIXELS; k *= 2) {
                __syncthreads();
                if ((id % k) == 0) {
                    // printf("DOUBLING k = %d, digit = %d, id = %d, i = %d, j = %d, myW[id] = %f, xVal = %f, cls[j] = %f\n", k, digit, id, i, j, myW[id], xVal, cls[j]);

                    if ((id + (k/2)) < NUM_PIXELS) {
                        x[id] += x[id + (k/2)];
                        // printf("DOUBLING k = %d, digit = %d, id = %d, i = %d, j = %d, myW[id] = %f, xVal = %f, cls[j] = %f\n", k, digit, id, i, j, myW[id], xVal, cls[j]);
                    } 
                    // else {
                    //     printf("NOOOOO k = %d, digit = %d, id = %d, i = %d, j = %d, myW[id] = %f, xVal = %f, cls[j] = %f\n", k, digit, id, i, j, myW[id], xVal, cls[j]);
                    // }
                }
            }
            */

            if (id == 0) {
                // result = x[id] + myB;
                result = 0;
                for (int k = 0; k < NUM_PIXELS; k++) {
                    result += x[k];
                }
                result += myB;

                if ((cls[j] * result) >= 1) {
                    constraint = true;
                } else {
                    constraint = false;
                    myB -= lr * (-cls[j]);
                }
                
            }
            __syncthreads();

            //get and update gradients
            if (constraint) {
                myW[id] = myW[id] - (lr * myW[id] * lambdaParam);
            } else {
                myW[id] = myW[id] - (lr * ((lambdaParam * myW[id]) - (cls[j] * xVal)));
            }
            
        }
        __syncthreads();
    }
    
    w[id] = myW[id];
    if (id == 0) {
        b[0] = myB;
    }
    printf("digit = %d, id = %d, end w[id] = %f, b = %f\n", digit, id, w[id], b[0]);
    
}

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
                if (i % 100 == 0) {
                    cout << "dot product i = " << i << ", a[i] = " << a[i] << ", b[i] = " << b[i] << endl;
                }
                sum += a[i] * b[i];
            }
            return sum;
        }

        // GPU
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
        
        SVM() {
            lr = -1;
            lambdaParam = -1;
            numIters = -1;
            w = NULL;
            b = 0;
            clsMap = NULL;
        }

        ~SVM() {
        }

        void setWeight(double* weights) {
            w = weights;
        }
        
        void setB(double bval) {
            b = bval;
        }

        void printWeight() {
            cout << "w[0] = " << w[0] << "| b = " << b << endl;
        }
        
        //DELETE THIS
        double* getBArr() {
            double* bArr = new double[1];
            bArr[0] = b;
            return bArr;
        }
        
        //size = length of y, length of X (X dims = [size][numFeatures])
        void fit(double* X, int numFeatures, int* y, int size) { //self, X, y
            
            initWeightsBias(numFeatures);
            getClsMap(y, size); // updates clsMap

            double* gpuX;
            double* gpuCLS;
            double* gpuW;
            double* gpuB;

            double* bArr = new double[1];
            bArr[0] = b;

	        cudaMalloc((void**)&gpuX, sizeof(double)*size*numFeatures); 
	        cudaMalloc((void**)&gpuCLS, sizeof(double)*size); 
	        cudaMalloc((void**)&gpuW, sizeof(double)*lenW);  
	        cudaMalloc((void**)&gpuB, sizeof(double)); 
	        cudaMemcpy(gpuX, X, sizeof(double)*size*numFeatures, cudaMemcpyHostToDevice);
	        cudaMemcpy(gpuCLS, clsMap, sizeof(double)*size, cudaMemcpyHostToDevice);
	        cudaMemcpy(gpuW, w, sizeof(double)*lenW, cudaMemcpyHostToDevice);
            cudaMemcpy(gpuB, bArr, sizeof(double), cudaMemcpyHostToDevice);

            //kernel(int numIters, int size, int lenW, double* w, double b, double* cls, double* X) {
            // cout << "lenW = " << lenW << endl;
            dim3 dimGrid(1);
            dim3 dimBlock(lenW);
            	
            struct timespec start, stop; 
            double time;
            if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
            // kernel<<<dimGrid, dimBlock>>>(numIters, size, lenW, gpuW, gpuB, gpuCLS, gpuX, lr, lambdaParam);				
// 
            if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
            time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
            printf("time is %f ns\n", time*1e9);	 
            
            cudaMemcpy(X, gpuX, sizeof(double)*size*numFeatures, cudaMemcpyDeviceToHost);
	        cudaMemcpy(clsMap, gpuCLS, sizeof(double)*size, cudaMemcpyDeviceToHost);
	        cudaMemcpy(w, gpuW, sizeof(double)*lenW, cudaMemcpyDeviceToHost);
	        cudaMemcpy(bArr, gpuB, sizeof(double), cudaMemcpyDeviceToHost);

            b = bArr[0];
            // printf("after kernel bArr = %f, b = %f\n", bArr[0], b);
            delete[] bArr;

            cudaFree(gpuX);  
            cudaFree(gpuCLS);  
            cudaFree(gpuW);  
            cudaFree(gpuB);

        }

        //X is of dims [size][numFeatures]
        // FREE estimate
        double* predict(double** X, int size) { //self, X
            double* estimate = new double[size]; //hold dot products
            cout << "inside predict" << endl;
            cout << "predict w[0] = " << w[0] << endl;
            cout << "predict b = " << b << endl;
            cout << "predict X[0] = " << X[0][0] << endl;
            for (int i = 0; i < size; i++) {
                // cout << "predict i = " << i << endl;
                estimate[i] = dotProduct(X[i], w, NUM_PIXELS);
                estimate[i] += b;
            }

            // don't predict class labels for MNIST, return raw for prediction between classes
            return estimate;
        }

};

// both yTrue and yPred have size elements, predictions = NUM_CLASS double* values (each double* = array of length 'size')
double accuracy(int* yTrue, double* predictions[], int size) {
    double sum = 0;
    int* yPred = new int[size];


    for (int j = 0; j < size; j++) {
        // double max = 0;
        double max = -DBL_MAX;
        int maxId = -1;
        for (int i = 0; i < NUM_CLASSES; i++) {
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


void parseBlobData(string dataFileStr, double** Xtrain, int* ytrain, int numTrain, double** Xtest, int* ytest) {
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
        cout << "total = " << total << " | numTrain = " << numTrain << " | idx = " << idx << endl;
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
        
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

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


// FINISH THIS?
/*
void parseTitanicData(string dataFileStr, double** Xtrain, double* ytrain, , double** Xtest, double* ytest) {
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
        int p1;
        int survived;
        int pclass;
        // string name;
        string sex;
        int age;
        int sibSp;
        int parch;
        string ticket;
        double fare;
        string cabin;
        string embarked;

        string temp = "";

        stringstream inputString(line);
        // ss >> xData1 >> xData2 >> cls;
        getline(inputString, temp, ',');
        p1 = atoi(temp.c_str());
        getline(inputString, temp, ',');
        survived = atof(temp.c_str());
        getline(inputString, temp, ',');
        pclass = atoi(temp.c_str());

        

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
        
    cout << "file read" << endl;
    inputFile.close();
    cout << "file closed" << endl;

}
*/

int main() {

    cout << "start" << endl;
    // training/test data parameters
    int numSamples = 42000;
    double testSize = 0.3;
    double trainSize = 0.2;
    int numTrain = trainSize * numSamples;
    int numTest = testSize * numSamples;
    int numFeatures = NUM_PIXELS;

    cout << "numTrain = " << numTrain << endl;
    cout << "numTest = " << numTest << endl; 

    // SVM hyperparameters
    double learningRate = 0.001; //1e-3
    double iters = 1000;
    // double iters = 20;
    double lamba = 1.0 / iters; //1e-2 

    int nStreams = NUM_CLASSES;
    // int nStreams = 1;
    int devId = 0;
    cudaDeviceProp prop;
    dim3 dimGrid(1);
    dim3 dimBlock(NUM_PIXELS);
    cout << "for get device properties" << endl;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId) );

    printf("nStreams = %d\n", nStreams);
    float ms; // elapsed time in milliseconds
    
    
    // create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nStreams];
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&dummyEvent) );

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
    
    // double* Xtrain1D = new double[numTrain * numFeatures];
    double* Xtrain1D;
    double* weights[NUM_CLASSES];
    double* bArrs[NUM_CLASSES];
    double* clsMaps[NUM_CLASSES];

    double* clsMap0;
    double* clsMap1;
    double* clsMap2;
    double* clsMap3;
    double* clsMap4;
    double* clsMap5;
    double* clsMap6;
    double* clsMap7;
    double* clsMap8;
    double* clsMap9;
    checkCuda( cudaMallocHost((void**)&clsMap0, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap1, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap2, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap3, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap4, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap5, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap6, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap7, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap8, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&clsMap9, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMallocHost((void**)&Xtrain1D, sizeof(double)*numTrain*numFeatures) );      // host pinned
    for (int i = 0; i < NUM_CLASSES; i++) {
        checkCuda( cudaMallocHost((void**)&weights[i], sizeof(double)*numFeatures) );      // host pinned
        checkCuda( cudaMallocHost((void**)&bArrs[i], sizeof(double)) );      // host pinned
        checkCuda( cudaMallocHost((void**)&clsMaps[i], sizeof(double)*numTrain) );      // host pinned
    }

    
    double* gpuWeights[NUM_CLASSES];
    double* gpuBArrs[NUM_CLASSES];
    double* gpuClsMaps[NUM_CLASSES];
    double* gpuX;

    double* gpuClsMap0;
    double* gpuClsMap1;
    double* gpuClsMap2;
    double* gpuClsMap3;
    double* gpuClsMap4;
    double* gpuClsMap5;
    double* gpuClsMap6;
    double* gpuClsMap7;
    double* gpuClsMap8;
    double* gpuClsMap9;
    checkCuda( cudaMalloc((void**)&gpuClsMap0, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap1, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap2, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap3, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap4, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap5, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap6, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap7, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap8, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuClsMap9, sizeof(double)*numTrain) );      // host pinned
    checkCuda( cudaMalloc((void**)&gpuX, sizeof(double)*numTrain*numFeatures) );      // host pinned
    for (int i = 0; i < NUM_CLASSES; i++) {
        checkCuda( cudaMalloc((void**)&gpuWeights[i], sizeof(double)*numFeatures) );      // host pinned
        checkCuda( cudaMalloc((void**)&gpuBArrs[i], sizeof(double)) );      // host pinned
        checkCuda( cudaMalloc((void**)&gpuClsMaps[i], sizeof(double)*numTrain) );      // host pinned
    }

    for (int i = 0; i < numTrain; i++) {
        for (int j = 0; j < numFeatures; j++) {
            Xtrain1D[(i * numFeatures) + j] = Xtrain[i][j];
        }
    }

    SVM classifiers[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        classifiers[i] = SVM(learningRate, lamba, iters);
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        // classifiers[i].initWeightBias(numFeatures); (weights = 0, bias = 0)
        for (int j = 0; j < numFeatures; j++) {
            weights[i][j] = 0;
        }
        
        bArrs[i][0] = 0;
    }
        //init getClsMap
    for (int j = 0; j < numTrain; j++) {
        if (ytrain[j] == 0) {
            clsMap0[j] = 1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 0" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        } else if (ytrain[j] == 1) {
            clsMap0[j] = -1;
            clsMap1[j] = 1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 1" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        } else if (ytrain[j] == 2) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = 1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 2" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        } else if (ytrain[j] == 3) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = 1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 3" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }  else if (ytrain[j] == 4) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = 1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 4" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }  else if (ytrain[j] == 5) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = 1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 5" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }  else if (ytrain[j] == 6) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = 1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 6" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        } else if (ytrain[j] == 7) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = 1;
            clsMap8[j] = -1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 7" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }  else if (ytrain[j] == 8) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = 1;
            clsMap9[j] = -1;
            cout << "cls = +1, i = 8" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }  else if (ytrain[j] == 9) {
            clsMap0[j] = -1;
            clsMap1[j] = -1;
            clsMap2[j] = -1;
            clsMap3[j] = -1;
            clsMap4[j] = -1;
            clsMap5[j] = -1;
            clsMap6[j] = -1;
            clsMap7[j] = -1;
            clsMap8[j] = -1;
            clsMap9[j] = 1;
            cout << "cls = +1, i = 9" << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl; 
        }
    }
    

    /*
    for (int i = 0; i < NUM_CLASSES; i++) {
        // classifiers[i].initWeightBias(numFeatures); (weights = 0, bias = 0)
        for (int j = 0; j < numFeatures; j++) {
            weights[i][j] = 0;
        }
         //init getClsMap
        for (int j = 0; j < numTrain; j++) {
            if (ytrain[j] == i) {
                clsMaps[i][j] = 1;
            } else {
                clsMaps[i][j] = -1;
                cout << "cls = -1, i = " << i << ", j = " << j << ", ytrain[j] = " << ytrain[j] << endl;
            }
        }
        bArrs[i][0] = 0;
    }
    */
  

    checkCuda(cudaMemcpy(gpuX, Xtrain1D, sizeof(double)*numTrain*numFeatures, cudaMemcpyHostToDevice));
    
    //training
    for (int i = 0; i < nStreams; ++i) {
        checkCuda( cudaStreamCreate(&stream[i]) );
    }

    checkCuda( cudaEventRecord(startEvent,0) );
    for (int i = 0; i < nStreams; i++) { //nStreams = NUM_CLASSES
            
            cout << "iters = " << iters << endl;
            cout << "numTrain = " << numTrain << endl;
            cout << "numFeatures = " << numFeatures << endl;
            cout << "learningRate = " << learningRate << endl;
            cout << "lamba = " << lamba << endl;
            cout << "i = " << i << endl;
            // checkCuda( cudaMemcpyAsync(gpuWeights[i], weights[i], 
            //                     sizeof(double)*numFeatures, cudaMemcpyHostToDevice, 
            //                     stream[i]) );
            // checkCuda( cudaMemcpyAsync(gpuBArrs[i], bArrs[i], 
            //                     sizeof(double), cudaMemcpyHostToDevice, 
            //                     stream[i]) );

            // checkCuda( cudaMemcpyAsync(gpuClsMaps[i], clsMaps[i], 
            //                     sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
            //                     stream[i]) );

            if (i == 0) {
                checkCuda( cudaMemcpyAsync(gpuClsMap0, clsMap0, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap0, gpuX, learningRate, lamba, i);	
            }
            else if (i == 1) {
                checkCuda( cudaMemcpyAsync(gpuClsMap1, clsMap1, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap1, gpuX, learningRate, lamba, i);
            }
            else if (i == 2) {
                checkCuda( cudaMemcpyAsync(gpuClsMap2, clsMap2, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap2, gpuX, learningRate, lamba, i);
            }
            else if (i == 3) {
                checkCuda( cudaMemcpyAsync(gpuClsMap3, clsMap3, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap3, gpuX, learningRate, lamba, i);
            }
            else if (i == 4) {
                checkCuda( cudaMemcpyAsync(gpuClsMap4, clsMap4, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap4, gpuX, learningRate, lamba, i);
            }
            else if (i == 5) {
                checkCuda( cudaMemcpyAsync(gpuClsMap5, clsMap5, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap5, gpuX, learningRate, lamba, i);
            }
            else if (i == 6) {
                checkCuda( cudaMemcpyAsync(gpuClsMap6, clsMap6, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap6, gpuX, learningRate, lamba, i);
            }
            else if (i == 7) {
                checkCuda( cudaMemcpyAsync(gpuClsMap7, clsMap7, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap7, gpuX, learningRate, lamba, i);
            }
            else if (i == 8) {
                checkCuda( cudaMemcpyAsync(gpuClsMap8, clsMap8, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap8, gpuX, learningRate, lamba, i);
            }
            else if (i == 9) {
                checkCuda( cudaMemcpyAsync(gpuClsMap9, clsMap9, 
                                sizeof(double)*numTrain, cudaMemcpyHostToDevice, 
                                stream[i]) );
                kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
                                        gpuBArrs[i], gpuClsMap9, gpuX, learningRate, lamba, i);
            }
            
                                
// ?__global__ void kernel(int numIters, int size, int lenW, double* w, double* b, double* cls, double* X, double lr, double lambdaParam, int digit) {

            // kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(iters, numTrain, numFeatures, gpuWeights[i], 
            //                             gpuBArrs[i], gpuClsMaps[i], gpuX, learningRate, lamba, i);	
                                                                                			
            checkCuda( cudaMemcpyAsync(weights[i], gpuWeights[i], 
                                sizeof(double)*numFeatures, cudaMemcpyDeviceToHost,
                                stream[i]) );
            checkCuda( cudaMemcpyAsync(bArrs[i], gpuBArrs[i], 
                                sizeof(double), cudaMemcpyDeviceToHost,
                                stream[i]) );
            cout << "after memcpy for digit = " << i << endl;
    }
    
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("Time for asynchronous training (ms): %f\n", ms);


    double* predictions[NUM_CLASSES];

    for (int i = 0; i < NUM_CLASSES; i++) {
        // cout << "training SVM " << i << "..." << endl;
        // classifiers[i].fit(Xtrain, numFeatures, ytrain, numTrain);
        classifiers[i].setWeight(weights[i]);
        classifiers[i].setB(bArrs[i][0]);
        cout << "predicting SVM " << i << "..." << endl;
        double* prediction = classifiers[i].predict(Xtest, numTest); 
        predictions[i] = prediction;

        cout << "digit = " << i << " , weight[i][50] = " << weights[i][50] << " b[i] = " << bArrs[i][0] << endl;

        // cout << "digit " << i << ": w[0] = " << weights[i][0] << " | b = " << bArrs[i][0] << endl;
        // classifiers[i].printWeight();
    }

    
    
    cout << "classifier trained " << endl;
    double acc = accuracy(ytest, predictions, numTest);
    printf("SVM Accuracy: %f\n", acc);
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
    
    // delete[] Xtrain1D;

    cudaFreeHost(Xtrain1D);
    for (int i = 0; i < NUM_CLASSES; i++) {
        cudaFreeHost(weights[i]);
        cudaFreeHost(bArrs[i]);
        cudaFreeHost(clsMaps[i]);
    }
    
    cudaFree(gpuX);
    for (int i = 0; i < NUM_CLASSES; i++) {
        cudaFree(gpuWeights[i]);
        cudaFree(gpuBArrs[i]);
        cudaFree(gpuClsMaps[i]);
    }

    cudaFreeHost(clsMap0);
    cudaFree(gpuClsMap0);
    cudaFreeHost(clsMap1);
    cudaFree(gpuClsMap1);
    cudaFreeHost(clsMap2);
    cudaFree(gpuClsMap2);
    cudaFreeHost(clsMap3);
    cudaFree(gpuClsMap3);
    cudaFreeHost(clsMap4);
    cudaFree(gpuClsMap4);
    cudaFreeHost(clsMap5);
    cudaFree(gpuClsMap5);
    cudaFreeHost(clsMap6);
    cudaFree(gpuClsMap6);
    cudaFreeHost(clsMap7);
    cudaFree(gpuClsMap7);
    cudaFreeHost(clsMap8);
    cudaFree(gpuClsMap8);
    cudaFreeHost(clsMap9);
    cudaFree(gpuClsMap9);

    cout << "done" << endl;
}