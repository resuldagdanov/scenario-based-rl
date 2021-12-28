#!/bin/bash

#TODO: add time difference to see the training time 
train () {
    working_dir=$(pwd)

    cd $CARLA_ROOT
    ./CarlaUE4.sh -RenderOffScreen &
    sleep 15
    export pid_carla=$(ps -elf | grep "CarlaUE4/Binaries/Linux" | grep -v grep | awk '{print $4}')
    
    cd $working_dir
    ./run_agent.sh &
    export pid_training=$!
    wait $pid_training

    kill -9 $pid_carla
    printf "carla server is killed\n"
}

python3 init_training_parameters.py

for i in $(seq 0 1 1)
do
    #printf "\n\nBatch $i started\n"
    train
done