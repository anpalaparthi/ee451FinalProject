CC = nvcc

svm: svm.cu
	$(CC) -o svm svm.cu

svm_multi: svm_multi.cu
	$(CC) -o svm_multi svm_multi.cu

clean:
	rm svm svm_multi