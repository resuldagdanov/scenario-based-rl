#!/bin/bash

SECONDS=0

# TODO: this can be given from command prompt as well
# global variables
export imitation_learning=false # to evaluate imitation learning model make it true, to train or evaluate dqn model make it false
export evaluate=true # true
export model_name="Jan_20_2022-16_25_23" # only used if evaluate true, make sure it exists
export load_episode_number=71 # only used if evaluate true, make sure it exists

export repetitions=5
export max_episode_batch_num=1
export json_file="all_towns_traffic_scenarios_WOR.json" #"town05_all_scenarios.json" #
export xml_file="failed_routes/town05_long/stuck_vehicle_1.xml" #"failed_routes/town01_short/red_light_18.xml" #"original_routes/routes_town01_short.xml" #"failed_routes/town05_short/collision_vehicle_1.xml" #"failed_routes/town01_short/collisions_layout_5.xml" #"failed_routes/town05_long/stuck_vehicle_1.xml"

max_episode_num=`expr $repetitions \* $max_episode_batch_num`

echo "max_episode_num:" $max_episode_num;
echo "evaluate: $evaluate";
echo "repetitions: $repetitions";
echo "xml_file: $xml_file";
echo "json_file: $json_file";
echo "imitation_learning: $imitation_learning";


if ${evaluate}; then # evaluate
    python3 init_training_parameters.py --evaluate --model_name $model_name --load_episode_number $load_episode_number --xml_file $xml_file --json_file $json_file
else # train
    python3 init_training_parameters.py --xml_file $xml_file --json_file $json_file
fi

#export CUBLAS_WORKSPACE_CONFIG=:16:8 # to run with CUDA while using pytorch deterministic algorithms (e.g. torch.set_deterministic(True))

train () {
    working_dir=$(pwd)

    cd $CARLA_ROOT
    ./CarlaUE4.sh &
    sleep 15
    export pid_carla=$(ps -elf | grep "CarlaUE4/Binaries/Linux" | grep -v grep | awk '{print $4}')
    
    cd $working_dir
    ./run_agent.sh -e $evaluate -r $repetitions -x $xml_file -j $json_file -i $imitation_learning & 
    export pid_training=$!
    wait $pid_training

    kill -9 $pid_carla
    printf "carla server is killed\n"
    sleep 10
}

SECONDS=0

for i in $(seq 1 1 $max_episode_batch_num)
do
    train
done

convertsecs() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "It took %02d hours %02d minutes %02d seconds.\n" $h $m $s
}

duration=$SECONDS
echo "$(convertsecs $duration)"