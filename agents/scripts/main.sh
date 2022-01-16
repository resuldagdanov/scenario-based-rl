#!/bin/bash

#TODO: this can be given from command prompt as well
#global variables
export imitation_learning=false #to evaluate imitation learning model make it true, to train or evaluate dqn model make it false
export evaluate=false #true
export model_name="Jan_15_2022-22_08_22" #only used if evaluate true, make sure it exists
export load_episode_number=140 #only used if evaluate true, make sure it exists

export repetitions=1
export max_episode_batch_num=1
export xml_file="stuck_vehicle_1.xml" #"stuck_vehicle_2.xml" #"stop_sign_6.xml"
export json_file="all_towns_traffic_scenarios_WOR.json" 

max_episode_num=`expr $repetitions \* $max_episode_batch_num`

echo "max_episode_num:" $max_episode_num;
echo "evaluate: $evaluate";
echo "repetitions: $repetitions";
echo "xml_file: $xml_file";
echo "json_file: $json_file";
echo "imitation_learning: $imitation_learning";

if ${evaluate}; then #evaluate
    python3 init_training_parameters.py --evaluate --model_name $model_name --load_episode_number $load_episode_number --xml_file $xml_file --json_file $json_file
else #train
    python3 init_training_parameters.py --xml_file $xml_file --json_file $json_file
fi

train () {
    working_dir=$(pwd)

    cd $CARLA_ROOT
    ./CarlaUE4.sh -RenderOffScreen &
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