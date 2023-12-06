# EE451 Final Project: Accelerating Classification Models on GPUs with CUDA
# By Anisha Palaparthi and Sriya

Here are the files that contain all the source code and related files, including cpp files for serial implementations, cu files for CUDA implementations, job files for using CARC, csv files for training data, and a Makefile for CUDA files. As part of the output, the accuracy and training execution time will be printed out (though many runs may take several minutes or even hours to complete). Here is the link to our Github repository as well: https://github.com/anpalaparthi/ee451FinalProject

For running SVM serial implementations, use the command "g++  svm.cpp - o svm" to compile and ./svm to run. This will train an SVM for the Blobs-2D dataset. For other datasets, replace 'svm' (in both the cpp file name and executable name) with svm_titanic, svm_multi (for MNIST), and svm_many_blob (for Blobs-100/400/700D). For many blobs, set the number features as 100/400/700 and change the csv file name to blob_100/400/700d.csv to use any of these datasets.

For running Logistic Regression serial implementations, use the command "g++  logistic.cpp - o logistic" to compile and ./logistic to run. This will train a logistic regression model for the Blobs-2D dataset. For other datasets, replace 'logistic' (in both the cpp file name and executable name) with logistic_titanic, and logistic_many_blob (for Blobs-100/400/700D). For many blobs, set the number features as 100/400/700 and change the csv file name to blob_100/400/700d.csv to use any of these datasets.

For running Multi-Layer Perceptron serial implementations, use the command "g++  mlp2.cpp - o mlp2" to compile and ./mlp2 to run. This will train an MLP for the Blobs-2D dataset. For other datasets, replace 'mlp2' (in both the cpp file name and executable name) with mlp2_titanic, and mlp2_many_blob (for Blobs-100/400/700D). For many blobs, set the number features as 100/400/700, set the hidden layer to 68/268/468, and change the csv file name to blob_100/400/700d.csv to use any of these datasets.

For running any of the CUDA implementations, you should run it on USC's HPC Cluster Discovery. There are 2 steps: you must make and the submit the batch job. Here is the list of the make commands and corresponding batch job to run. (Please disregard the naming convention, they are simply a way for me to keep track of different file versions). Output files are of the form 'jobscript_name'.out -- for example, when running SVM Titanic, the output file would be named "svm_gpu_titanic.out".

| Model/Dataset      | Make Command | Batch Script |
| ----------- | ----------- | ----------- |
| SVM Blobs-2D     | make svm_gpu       | job_svm_gpu.sl       |
| SVM Titanic   | make svm_gpu_titanic | job_svm_gpu_titanic.sl|
| SVM MNIST   | make svm_gpu_multi | job_svm_gpu_multi.sl|
| SVM Blobs 100/400/700 D (use same procedure as serial to modify feature size)   | make svm_gpu_many_blob | job_svm_gpu_many_blobs.sl|
| Logistic Blobs-2D     | make logistic_gpu       | job_logistic_gpu.sl       |
| Logistic Titanic   | make logistic_gpu_titanic | job_logistic_gpu_titanic.sl|
| Logistic Blobs 100/400/700 D (use same procedure as serial to modify feature size)   | make logistic_gpu_many_blob | job_logistic_gpu_many_blobs.sl|
| MLP Blobs-2D     | make mlp2_gpu_alt       | job_mlp2_gpu_alt.sl       |
| MLP Titanic   | make mlp2_gpu_titanic | job_mlp2_gpu_titanic.sl|
| MLP Blobs 100/400/700 D (use same procedure as serial to modify feature size)   | make mlp2_gpu_alt_many_blob | job_mlp2_gpu_alt_many_blobs.sl|