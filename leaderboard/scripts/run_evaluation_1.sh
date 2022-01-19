pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python
pkill -9 python

export MODEL_ROOT=/home/feyza/depo/research/carla_rl/scenario-based-rl #/home/resul/Eatron/Company/carla_challenge/ea202101001_platooning_demo/carla_ws # TODO: must be changed
export BASE_CODE_PATH=${MODEL_ROOT}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${MODEL_ROOT}/leaderboard:leaderboard
export PYTHONPATH=$PYTHONPATH:${MODEL_ROOT}/scenario_runner:scenario_runner

export LEADERBOARD_ROOT=${MODEL_ROOT}/leaderboard
export TEAM_CONFIG=${MODEL_ROOT}
export TEAM_AGENT=${MODEL_ROOT}/agents/rl_training/imitation_learning_agent.py

export ROUTES=${MODEL_ROOT}/data/routes/failed_routes/town01_short/collisions_layout_5.xml # original_routes/routes_town01_short.xml  #
export SCENARIOS=${MODEL_ROOT}/data/scenarios/all_towns_traffic_scenarios_WOR.json
export CHECKPOINT_ENDPOINT=${MODEL_ROOT}/x.json

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=False

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT}