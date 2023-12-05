CC = nvcc

svm: svm.cu
	$(CC) -o svm svm.cu
	
svm_gpu: svm_gpu.cu
	$(CC) -o svm_gpu svm_gpu.cu

svm_gpu_NOUSE: svm_gpu_NOUSE.cu
	$(CC) -o svm_gpu_NOUSE svm_gpu_NOUSE.cu
	
svm_gpu_multi: svm_gpu_multi.cu
	$(CC) -o svm_gpu_multi svm_gpu_multi.cu
	
svm_gpu_titanic: svm_gpu_titanic.cu
	$(CC) -o svm_gpu_titanic svm_gpu_titanic.cu
	
svm_gpu_many_blob: svm_gpu_many_blob.cu
	$(CC) -o svm_gpu_many_blob svm_gpu_many_blob.cu
		
svm_gpu_idc: svm_gpu_idc.cu
	$(CC) -o svm_gpu_idc svm_gpu_idc.cu

svm_multi: svm_multi.cu
	$(CC) -o svm_multi svm_multi.cu
	
logistic_gpu: logistic_gpu.cu
	$(CC) -o logistic_gpu logistic_gpu.cu
	
logistic_gpu_titanic: logistic_gpu_titanic.cu
	$(CC) -o logistic_gpu_titanic logistic_gpu_titanic.cu
	
logistic_gpu_mnist: logistic_gpu_mnist.cu
	$(CC) -o logistic_gpu_mnist logistic_gpu_mnist.cu
	
logistic_gpu_many_blob: logistic_gpu_many_blob.cu
	$(CC) -o logistic_gpu_many_blob logistic_gpu_many_blob.cu
	
mlp2_gpu: mlp2_gpu.cu
	$(CC) -o mlp2_gpu mlp2_gpu.cu
	
mlp2_gpu_alt: mlp2_gpu_alt.cu
	$(CC) -o mlp2_gpu_alt mlp2_gpu_alt.cu
	
mlp2_gpu_alt_many_blob: mlp2_gpu_alt_many_blob.cu
	$(CC) -o mlp2_gpu_alt_many_blob mlp2_gpu_alt_many_blob.cu
	
mlp2_gpu_titanic: mlp2_gpu_titanic.cu
	$(CC) -o mlp2_gpu_titanic mlp2_gpu_titanic.cu
	
mlp2_gpu_many_blob: mlp2_gpu_many_blob.cu
	$(CC) -o mlp2_gpu_many_blob mlp2_gpu_many_blob.cu

clean:
	rm svm svm_multi svm_gpu svm_gpu_multi logistic_gpu mlp2_gpu svm_gpu_NOUSE logistic_gpu_titanic
	logistic_gpu_mnist mlp2_gpu_titanic mlp2_gpu_many_blob logistic_gpu_many_blob svm_gpu_many_blob mlp2_gpu_alt