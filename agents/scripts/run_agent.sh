#!/bin/bash
pkill -9 python

while getopts e:r:x:j:i: flag
do
    case "${flag}" in
        e) evaluate=${OPTARG};;
        r) repetitions=${OPTARG};;
        x) xml_file=${OPTARG};;
        j) json_file=${OPTARG};;
        i) imitation_learning=${OPTARG};;
    esac
done

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner

export BASE_CODE_PATH="$(dirname $(dirname "$(pwd)"))" # automatically find scenario-based-rl folder
export ROUTES=${BASE_CODE_PATH}/data/routes/${xml_file}
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/${json_file}
export CHECKPOINT_PATH=${BASE_CODE_PATH}/checkpoint/
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}

if ${imitation_learning}; then # use offset agent (imitation learning model)
    export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/offset_agent.py

    if ${evaluate}; then # evaluate
        python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} --repetitions ${repetitions} --evaluate --imitation_learning
    else # train
        python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} --repetitions ${repetitions} --imitation_learning
    fi

else # train or evaluate dqn agent
    export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/dqn_agent.py #waypoint_agent.py

    if ${evaluate}; then # evaluate
        python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} --repetitions ${repetitions} --evaluate
    else # train
        python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} --repetitions ${repetitions}
    fi
fi