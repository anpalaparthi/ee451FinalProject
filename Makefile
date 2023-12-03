CC = nvcc

svm: svm.cu
	$(CC) -o svm svm.cu
	
svm_gpu: svm_gpu.cu
	$(CC) -o svm_gpu svm_gpu.cu
	
svm_gpu_multi: svm_gpu_multi.cu
	$(CC) -o svm_gpu_multi svm_gpu_multi.cu

svm_multi: svm_multi.cu
	$(CC) -o svm_multi svm_multi.cu
	
logistic_gpu: logistic_gpu.cu
	$(CC) -o logistic_gpu logistic_gpu.cu
	
mlp2_gpu: mlp2_gpu.cu
	$(CC) -o mlp2_gpu mlp2_gpu.cu

clean:
	rm svm svm_multi svm_gpu svm_gpu_multi logistic_gpu mlp2_gpu