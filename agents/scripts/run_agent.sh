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

export BASE_CODE_PATH="$(dirname $(dirname "$(pwd)"))" # automatically find scenario-based-rl folder

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/agents:agents
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/leaderboard:leaderboard
export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}/scenario_runner:scenario_runner

echo $PYTHONPATH

export LEADERBOARD_ROOT=${BASE_CODE_PATH}/leaderboard
export TEAM_CONFIG=${BASE_CODE_PATH}

export ROUTES=${BASE_CODE_PATH}/data/routes/${xml_file}
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/${json_file}
export CHECKPOINT_ENDPOINT=${BASE_CODE_PATH}/x.json
export CHECKPOINT_PATH=${BASE_CODE_PATH}/checkpoint/

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export RESUME=False

if ${imitation_learning}; then # use offset agent (imitation learning model)
    export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/imitation_learning_agent.py #offset_agent.py
    if ${evaluate}; then # evaluate
        echo "imitation learning eval"
        python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=${repetitions} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME} \
            --evaluate \
            --imitation_learning
    else # train
        python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=${repetitions} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME} \
            --imitation_learning
    fi
else # train or evaluate dqn agent
    export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/dqn_agent.py
    if ${evaluate}; then # evaluate
        python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=${repetitions} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME} \
            --evaluate
    else # train
        python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=${repetitions} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME}
    fi
fi