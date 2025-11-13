CXX=g++
CXX_FLAGS = -O3 -fopenmp -std=c++11

CC=gcc

omp: src/openmp_single/*.cpp  src/common/*.cpp src/common/*.h
	${CXX} ${CXX_FLAGS} $^ -I src/openmp_single -I src/common -o $@

upload:
	scp -i ~/.ssh/id_rsa_hpc -r Makefile src *.cpp test_omp.sh edu-cmc-skmodel25-627-08@polus.hpc.cs.msu.ru:~/project/

test_omp: omp
	./test_omp.sh

fetch_omp_results:
	rm -rf omp_tests
	scp -i ~/.ssh/id_rsa_hpc -r edu-cmc-skmodel25-627-08@polus.hpc.cs.msu.ru:~/project/omp_tests .
