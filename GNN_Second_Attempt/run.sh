#!/bin/bash


source venv/bin/activate
python train_gnn.py && python test_gnn.py && python analyze.py