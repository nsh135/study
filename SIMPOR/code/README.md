





## SIMPOR
This repo implement the idea in the manuscript SIMPOR, a imbalanced learning approache which generates synthetic samples for minority that maximizing posterior ratio. 

(The paper is under review and code is still under development. Please stay tuned. )

* Set up the environment using pip with the requirment in 'environment.txt'

* Run each dataset separately.  
`python main.py --dataset dataset_name -n_runs 2`


## Error: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/tensorflow2/lib/

## re-run particular experiements
Set CONTINUE to True (main.py)

## find and remove exps before re-run
find . -type f -name "*SIMPOR*.csv" -exec rm -f {} \;
find . -type f -name "*FinalResult*.csv" -exec rm -f {} \;


