pkill -9 python

export MODEL_ROOT=/home/resul/Eatron/Company/carla_challenge/ea202101001_platooning_demo/carla_ws # TODO: must be changed
export CARLA_ROOT=//home/resul/Eatron/Company/carla_challenge/CARLA_0.9.10.1 # TODO: must be changed

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${MODEL_ROOT}/tools/leaderboard
export PYTHONPATH=$PYTHONPATH:${MODEL_ROOT}/tools/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:${MODEL_ROOT}/tools/scenario_runner

export LEADERBOARD_ROOT=${MODEL_ROOT}/tools/leaderboard
export TEAM_CONFIG=${MODEL_ROOT}/model_checkpoint
export TEAM_AGENT=${MODEL_ROOT}/tools/leaderboard/team_code/eatron_agent.py
export SAVE_PATH=${MODEL_ROOT}/results/data

export ROUTES=${MODEL_ROOT}/tools/leaderboard/data/routes/training_routes/routes_town01_short.xml # TODO: please change this route
export SCENARIOS=${MODEL_ROOT}/tools/leaderboard/data/scenarios/all_towns_traffic_scenarios_WOR.json # TODO: please change this scenario
export CHECKPOINT_ENDPOINT=${MODEL_ROOT}/results/town01_short_results.json # TODO: please change this inflation path

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True

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
--trafficManagerPort=${TM_PORT}