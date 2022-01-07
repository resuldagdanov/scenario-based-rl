#!/bin/bash
pkill -9 python

while getopts e:r:x:j: flag
do
    case "${flag}" in
        e) evaluate=${OPTARG};;
        r) repetitions=${OPTARG};;
        x) xml_file=${OPTARG};;
        j) json_file=${OPTARG};;
    esac
done

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner

# NOTE: change the following exports
export BASE_CODE_PATH="$(dirname $(dirname "$(pwd)"))" # automatically find scenario-based-rl folder
export ROUTES=${BASE_CODE_PATH}/data/routes/${xml_file}
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/${json_file}
export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/dqn_agent.py #waypoint_agent.py # offset_agent.py # TODO:
export CHECKPOINT_PATH=${BASE_CODE_PATH}/checkpoint/

export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}

if ${evaluate}; then # evaluate
    python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} 0 --repetitions ${repetitions} --evaluate
else # train
    python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} 0 --repetitions ${repetitions}
fi