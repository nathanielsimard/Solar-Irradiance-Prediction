#!/bin/bash

file=$1
port=$2

tensorboard --logdir $file \
            --port $port \
            --host localhost \
            --reload_interval 5
