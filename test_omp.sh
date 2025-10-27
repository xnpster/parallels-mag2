#!/bin/bash

rm -rf omp_tests
mkdir omp_tests

for N in 128 256 512 1024
do
    for K in 10
    do
        for threads in 1 2 4 8
        do
            run_id=run_${threads}_${N}_${K}

            mpisubmit.pl -p 1 -t ${threads} \
                --stdout omp_tests/${run_id}.out \
                --stderr omp_tests/${run_id}.err \
                ./omp -- \
                -t ${threads} \
                -N ${N} \
                -K ${K} \
                -logdir omp_tests \
                -logfile ${run_id}.log
        done
    done
done