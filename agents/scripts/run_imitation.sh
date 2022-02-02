#!/bin/bash

pkill -9 python
pkill -9 python
pkill -9 python

export xml_file="original_routes/routes_town05_short.xml"
export json_file="all_towns_traffic_scenarios_WOR.json"

export BASE_CODE_PATH="$(dirname $(dirname "$(pwd)"))"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/agents:agents
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/leaderboard:leaderboard
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/scenario_runner:scenario_runner

export LEADERBOARD_ROOT=${BASE_CODE_PATH}/leaderboard
export TEAM_CONFIG=${BASE_CODE_PATH}

export ROUTES=${BASE_CODE_PATH}/data/routes/${xml_file}
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/${json_file}

now=`date +"%Y_%m_%d_%H_%M_%S"`
json=".json"
CHECKPOINT_ENDPOINT_FILE_NAME="${now}${json}"

export CHECKPOINT_PATH=${BASE_CODE_PATH}/checkpoint/
export CHECKPOINT_ENDPOINT=${BASE_CODE_PATH}/${CHECKPOINT_ENDPOINT_FILE_NAME}
export SAVE_DATASET_NAME=${now}
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export RESUME=False

export TEAM_AGENT=${BASE_CODE_PATH}/agents/imitation_training/autopilot_agent.py

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=1 \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME} \
            --evaluate \
            --imitation_learning