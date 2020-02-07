#!/bin/bash

file=$1
port=$2

tensorboard --logdir /project/cq-training-1/project1/teams/team10/tensorboard/$file \
            --port $port \
            --host localhost \
            --reload_interval 5
