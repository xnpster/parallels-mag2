#!/bin/bash

test_dir=combined_tests

rm -rf ${test_dir}
mkdir ${test_dir}

for N in 1024
do
    for K in 20
    do
        for procs in 8 32
        do
            for t in 1 2 4 8 16
            do
                for L in 1
                do
                    run_id=run_${procs}_${t}_${N}_${K}_${L}

                    mpisubmit.pl -p ${procs} -t ${t} \
                        --stdout ${test_dir}/${run_id}.out \
                        --stderr ${test_dir}/${run_id}.err \
                        ./mpi -- \
                        -t ${t} \
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
done
