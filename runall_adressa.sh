#! /bin/bash

python3 run.py --dataset adressa --algorithm Popularity --timeout 3600
python3 run.py --dataset adressa --algorithm ItemKNN --timeout 7200
python3 run.py --dataset adressa --algorithm EASE --timeout 7200
python3 run.py --dataset adressa --algorithm TARSItemKNNLiu --timeout 7200
python3 run.py --dataset adressa --algorithm TARSItemKNNDing --timeout 7200
python3 run.py --dataset adressa --algorithm SequentialRules --timeout 7200
python3 run.py --dataset adressa --algorithm GRU4RecNegSampling --timeout 21600
python3 run.py --dataset adressa --algorithm STAN --timeout 21600
