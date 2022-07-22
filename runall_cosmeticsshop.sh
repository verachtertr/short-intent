#! /bin/bash

python3 run.py --dataset cosmeticsshop --algorithm Popularity --timeout 3600
python3 run.py --dataset cosmeticsshop --algorithm ItemKNN --timeout 7200
python3 run.py --dataset cosmeticsshop --algorithm EASE --timeout 7200
python3 run.py --dataset cosmeticsshop --algorithm TARSItemKNNLiu --timeout 7200
python3 run.py --dataset cosmeticsshop --algorithm TARSItemKNNDing --timeout 7200
python3 run.py --dataset cosmeticsshop --algorithm SequentialRules --timeout 7200
#python3 run.py --dataset cosmeticsshop --algorithm GRU4RecCrossEntropy --timeout 21600
#python3 run.py --dataset cosmeticsshop --algorithm STAN --timeout 21600
