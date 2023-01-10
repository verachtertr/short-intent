#! /bin/bash

python3 run.py --dataset amazon_toys_and_games --algorithm Popularity --timeout 3600
python3 run.py --dataset amazon_toys_and_games --algorithm ItemKNN --timeout 7200
python3 run.py --dataset amazon_toys_and_games --algorithm EASE --timeout 7200
python3 run.py --dataset amazon_toys_and_games --algorithm SequentialRules --timeout 7200
python3 run.py --dataset amazon_toys_and_games --algorithm GRU4RecNegSampling --timeout 21600