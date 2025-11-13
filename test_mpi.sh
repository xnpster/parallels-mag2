#!/bin/bash

test_dir=mpi_tests

rm -rf ${test_dir}
mkdir ${test_dir}

for N in 128
do
    for K in 20
    do
        for procs in 1 4 8 16 32
        do
            for L in 1 3.141593
            do
                run_id=run_${procs}_${N}_${K}_${L}

                mpisubmit.pl -p ${procs} -t 1 \
                    --stdout ${test_dir}/${run_id}.out \
                    --stderr ${test_dir}/${run_id}.err \
                    ./mpi -- \
                    -t 1 \
                    -N ${N} \
                    -K ${K} \
                    -Lx ${L} \
                    -Ly ${L} \
                    -Lz ${L} \
                    -logdir ${test_dir} \
                    -logfile ${run_id}.log
            done
        done
    done
done