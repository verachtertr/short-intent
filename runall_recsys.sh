#! /bin/bash

python3 run.py --dataset recsys2015 --algorithm Popularity --timeout 3600
python3 run.py --dataset recsys2015 --algorithm ItemKNN --timeout 7200
python3 run.py --dataset recsys2015 --algorithm EASE --timeout 7200
python3 run.py --dataset recsys2015 --algorithm TARSItemKNNLiu --timeout 7200
python3 run.py --dataset recsys2015 --algorithm TARSItemKNNDing --timeout 7200
python3 run.py --dataset recsys2015 --algorithm SequentialRules --timeout 7200
python3 run.py --dataset recsys2015 --algorithm GRU4RecCrossEntropy --timeout 21600
python3 run.py --dataset recsys2015 --algorithm STAN --timeout 21600