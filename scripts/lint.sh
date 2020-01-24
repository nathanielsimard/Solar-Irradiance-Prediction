#!/bin/sh 

flake8 || exit 1
pydocstyle || exit 1
black . || exit 1
mypy --ignore-missing-imports --package src --package tests || exit 1
