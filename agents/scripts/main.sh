#!/bin/bash

#TODO: this can be given from command prompt as well
#global variables
export evaluate=false #true
export model_name="Jan_09_2022-18_31_00" #only used if evaluate true, make sure it exists
export load_episode_number=30 #only used if evaluate true, make sure it exists

export repetitions=15
export max_episode_batch_num=10
export xml_file="red_light_1.xml"
export json_file="all_towns_traffic_scenarios_WOR.json"

max_episode_num=`expr $repetitions \* $max_episode_batch_num`

echo "max_episode_num:" $max_episode_num
echo "evaluate: $evaluate";
echo "repetitions: $repetitions";
echo "xml_file: $xml_file";
echo "json_file: $json_file";

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
    ./run_agent.sh -e $evaluate -r $repetitions -x $xml_file -j $json_file & 
    export pid_training=$!
    wait $pid_training

    kill -9 $pid_carla
    printf "carla server is killed\n"
}

SECONDS=0

for i in $(seq 1 1 $max_episode_batch_num)
do
    train
done

duration=$SECONDS
echo "It took $(($duration / 3600)) hours $(($duration / 60)) minutes $(($duration % 60)) seconds."