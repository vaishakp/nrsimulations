#!/bin/bash

conda activate wprep

TERMINATE=$(cat scripts/terminate.txt)

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
