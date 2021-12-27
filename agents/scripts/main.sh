#!/bin/bash

train () {
    working_dir=$(pwd)

    cd $CARLA_ROOT
    ./CarlaUE4.sh -RenderOffScreen &
    sleep 15
    export pid_carla=$(ps -elf | grep "CarlaUE4/Binaries/Linux" | grep -v grep | awk '{print $4}')
    
    cd $working_dir
    printf "training is starting\n"
    ./run_agent.sh &
    export pid_training=$!

    wait $pid_training
    printf "training is completed\n"

    echo $pid_carla
    kill -9 $pid_carla
    printf "carla server is killed\n"
}

for i in $(seq 1 1 10)
do
    printf "\n\nTraining batch $i started\n"
    train
done