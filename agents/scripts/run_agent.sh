#!/bin/bash

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner

# TODO: change the following exports
# ------------------------------------------------------------------------------------------------------ #
export BASE_CODE_PATH=~/Research/Codes/Carla/scenario-based-rl
export ROUTES=${BASE_CODE_PATH}/data/routes/routes_town01_short.xml
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/all_towns_traffic_scenarios_autopilot.json
export TEAM_AGENT=${BASE_CODE_PATH}/agents/rl_training/waypoint_agent.py
export CHECKPOINT_PATH=${BASE_CODE_PATH}/checkpoint/

export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}

python3 ${BASE_CODE_PATH}/agents/_scenario_runner/scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} 0
