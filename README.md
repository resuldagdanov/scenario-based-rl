# imitation-learning in carla simulator
Implementation of Imitation Learning Method on Carla Simulation with Trainings Based on Scenarios

## Installation Steps

* Download [Carla 0.9.10.1](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz)
* unzip folder to PATH_TO_CARLA_ROOT_SH (user defined path)

```sh
conda create -n carla python=3.7

conda activate carla

pip install -r requirements.txt
```

## versions
* carla version 0.9.10.1
* python 3.7.11
* unreal engine 4.24
* scenario_runner 0.9.9
* leaderboard
* pygame 2.0.1
<!--add pytorch version-->

```sh
pip install -r requirements.txt
```

## Prepare Necessary Directory Exports to Bashrc

```sh
gedit ~/.bashrc

export DeFIX_PATH=PATH_TO_MAIN_DeFIX_REPO
export CARLA_ROOT=PATH_TO_CARLA_ROOT_SH

export SCENARIO_RUNNER_ROOT="${DeFIX_PATH}/scenario_runner"
export LEADERBOARD_ROOT="${DeFIX_PATH}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":${PYTHONPATH}

source ~/.bashrc
```

## Run Carla Server on GPU
```sh
cd $CARLA_ROOT
./CarlaUE4.sh -prefernvidia
```

## Run Carla Autopilot (Expert Policy)
```sh
cd $DeFIX_PATH/imitation-learning/scripts
. run_autopilot.sh
```

## Run Imitation Learning Agent (Brake Classifier Agent)
```sh
cd $DeFIX_PATH/imitation-learning/scripts
. run_imitation.sh
```

## List of Supported Scenarios

Welcome to the ScenarioRunner for CARLA! This document provides a list of all
currently supported scenarios, and a short description for each one.

### FollowLeadingVehicle
The scenario realizes a common driving behavior, in which the user-controlled
ego vehicle follows a leading car driving down a given road in Town01. At some
point the leading car slows down and finally stops. The ego vehicle has to react
accordingly to avoid a collision. The scenario ends either via a timeout, or if
the ego vehicle stopped close enough to the leading vehicle

### FollowLeadingVehicleWithObstacle
This scenario is very similar to 'FollowLeadingVehicle'. The only difference is,
that in front of the leading vehicle is a (hidden) obstacle that blocks the way.

### VehicleTurningRight
In this scenario the ego vehicle takes a right turn from an intersection where
a cyclist suddenly drives into the way of the ego vehicle,which has to stop
accordingly. After some time, the cyclist clears the road, such that ego vehicle
can continue driving.

### VehicleTurningLeft
This scenario is similar to 'VehicleTurningRight'. The difference is that the ego
vehicle takes a left turn from an intersection.

### OppositeVehicleRunningRedLight
In this scenario an illegal behavior at an intersection is tested. An other
vehicle waits at an intersection, but illegally runs a red traffic light. The
approaching ego vehicle has to handle this situation correctly, i.e. despite of
a green traffic light, it has to stop and wait until the intersection is clear
again. Afterwards, it should continue driving.

### StationaryObjectCrossing
In this scenario a cyclist is stationary waiting in the middle of the road and
blocking the way for the ego vehicle. Hence, the ego vehicle has to stop in
front of the cyclist.

### DynamicObjectCrossing
This is similar to 'StationaryObjectCrossing', but with the difference that the
cyclist is dynamic. It suddenly drives into the way of the ego vehicle, which
has to stop accordingly. After some time, the cyclist will clear the road, such
that the ego vehicle can continue driving.

### NoSignalJunctionCrossing (not supported!)
This scenario tests negotiation between two vehicles crossing cross each other
through a junction without signal.
The ego vehicle is passing through a junction without traffic lights
And encounters another vehicle passing across the junction. The ego vehicle has
to avoid collision and navigate across the junction to succeed.

### ControlLoss
In this scenario control loss of a vehicle is tested due to bad road conditions, etc
and it checks whether the vehicle is regained its control and corrected its course.

### ManeuverOppositeDirection (not working!)
In this scenario vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction and encroaches into another
vehicle traveling in the opposite direction.

### OtherLeadingVehicle
The scenario realizes a common driving behavior, in which the user-controlled ego
vehicle follows a leading car driving down a given road.
At some point the leading car has to decelerate. The ego vehicle has to react
accordingly by changing lane to avoid a collision and follow the leading car in
other lane. The scenario ends via timeout, or if the ego vehicle drives certain
distance.

### SignalizedJunctionRightTurn
In this scenario right turn of hero actor without collision at signalized intersection
is tested. Hero Vehicle is turning right in an urban area, at a signalized intersection and
turns into the same direction of another vehicle crossing straight initially from
a lateral direction.

### SignalizedJunctionLeftTurn
In this scenario hero vehicle is turning left in an urban area,
at a signalized intersection and cuts across the path of another vehicle
coming straight crossing from an opposite direction.

<!--https://leaderboard.carla.org/scenarios/-->