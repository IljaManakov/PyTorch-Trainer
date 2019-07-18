#!/usr/bin/env bash

python setup.py sdist bdist_wheel
python -m twine upload dist/*
rm -r /home/ilja/Documents/Promotion/Project_Helpers/trainer/dist
rm -r /home/ilja/Documents/Promotion/Project_Helpers/trainer/build
rm -r /home/ilja/Documents/Promotion/Project_Helpers/trainer/pt_trainer.egg-info