#!/bin/bash

. /mnt/pfs/vaishak.p/soft/anaconda3_2023/bin/activate
conda activate wprep

GIT_DIR="$(git rev-parse --show-toplevel)"

TERMINATE=$(cat ${GIT_DIR}/scripts/terminate.txt)e

#TERMINATE = "$(boolean "${terminate_int}")"

echo "Terminate? ${TERMINATE}"


while [ "$TERMINATE" != "1" ]
do
    echo 1
    python get_sim_status.py
    bash commit_sim_status.sh
    echo "Done one update iteration"
    sleep 60m
done
