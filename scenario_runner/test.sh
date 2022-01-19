export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town01_all_scenarios.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes/training_routes/routes_town01_tiny.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/eatron/e2e_agent.py


python scenario_runner.py --agent ${TEAM_AGENT} --route ${ROUTES} ${SCENARIOS} --reloadWorld --scenarios group 