#!/usr/bin/env bash

python setup.py sdist bdist_wheel
python -m twine upload dist/*
rm -r /home/kazuki/Documents/Coding_Projects/PyTorch-Trainer/dist
rm -r /home/kazuki/Documents/Coding_Projects/PyTorch-Trainer/build
rm -r /home/kazuki/Documents/Coding_Projects/PyTorch-Trainer/pt_trainer.egg-info