CC = nvcc

svm: svm.cu
	$(CC) -o svm svm.cu

clean:
	rm svm