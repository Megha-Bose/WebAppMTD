# Moving Target Defense under Uncertainty

There are three folders: 

1. The `Data` folder contains the given and generated datasets. The files in the input subfolder are sent as input to the algorithms in Expts folder and their  output is sent to the output subfolder. 

    - The input files go as `[dataset number](attacks.txt, switching.txt, utilities.txt, vulnerabilities.txt)`. 

    - In the output files, the `[dataset number]output_[attacker strategy]` file stores the utilities in each timestep while `[dataset number]overall_out_[attacker strategy]` stores the average switches, runtime and utilities per iteration.


2. `Data_Gen` creates the text files. It contains the following files:
    - BSSG_input.txt: Data from Sengupta's github library
    - data_generator.py: Generates random datasets. To run,
        `python3 data_generator.py n1 n2` generates input files  in `Data/input/` for datasets `n1` to `n2`  
    - parser.py: Parses data from Sengupta's library and makes it dataset 0 inside Data folder


3. The third folder is `Expts` and there are three important files here; the rest of them are just old files.

    - The three files are attacker_eps.py, attacker_BR.py and attacker_fplue.py
    - Run them like `python3 attacker_BR.py [dataset number]` to generate output files for a dataset.

    - These three files correspond to three different attacker strategies: eps corresponds to epsilon optimal -- given some epsilon, the attacker randomly chooses between all the strategies that achieve a (1-epsilon) \times optimal utility for that round. This type assumes knowledge about what move the defender makes. 

    - BR corresponds to Best Response (exactly the same as that of the FPL-UE paper) and fplue corresponds to an attacker whose strategy is decided by the FPL-UE algorithm. The rest of the code in these three files is the same: they correspond to ten defender strategies.

    - Run `compare.py n1 n2` to generate switches, runtime and utility graphs for datasets `n1` to `n2`.


P.S. Only Attacker Best Response has been updated till now. After verification, the code can be pasted in other files as well.
