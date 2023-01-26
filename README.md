# Moving Target Defense under Uncertainty

The data and code are present in five main directories: 

1. **Data**: Contains the given and generated datasets. The files in the input subdirectory are sent as input to the algorithms in Expts directory and their output is sent to the output subdirectory. 

    - This folder contains input datasets for both general sum and zero sum cases. Each dataset consists of vulnerability lists for each configuration, switching costs, reward values for both the attacker and defender and complete list of possible system vulnerabilities in the concerned system of the dataset. The input files go as `[dataset number](attacks.txt, switching.txt, utilities.txt, vulnerabilities.txt)`. 

    - In the output files, the `[dataset number]output_[attacker strategy]` file stores the utilities in each timestep while `[dataset number]overall_out_[attacker strategy]` stores the average switches, runtime and utilities per iteration.

    - General sum game datasets and zero sum game cases are separated in different directories.


2. **Data_Gen**: Creates the text files. It contains the following files:
    - BSSG_input.txt: Data from Sengupta's github library for the small NVD dataset.
    - data_generator.py: Generates random datasets. To run,
        `python3 data_generator.py n1 n2` generates input files  in `Data/input/` for datasets `n1` to `n2`  
        `0` should be added as third argument if the command is run for zero sum game datasets
    - nvd_data_gen.py: Generates NVD-based datasets. 
    
        Make a directory named `nvd_data` and add data of required years from [NIST NVD data feed](https://nvd.nist.gov/vuln/data-feeds). 
    
        To run,
        `python3 nvd_data_gen.py n1 n2 y1 y2` generates input files  in `Data/input/` for datasets `n1` to `n2` using vulnerabilities from NVD database from year `y1` to year `y2`
    - parser.py: Parses data from Sengupta's library and makes it dataset `0` inside `Data/input/general_sum/` directory


3. **Expts**: Contains the main code that uses different attacker strategies to generate output.

    - Run them like `python3 attacker_[attacker_strategy].py n1 n2` to generate output files for datasets `n1` to `n2` using corresponding attacker strategy. `0` should be added as third argument if the command is run for zero sum game datasets. 

    - Run `compare_[attacker strategy].py n1 n2` to generate switches, runtime and utility graphs inside `graphs` directory in the corresponding output directory for datasets `n1` to `n2` for an attacker strategy. `0` should be added as third argument if the command is run for zero sum game datasets

4. **Analysis**: Contains python notebooks to generate graphs for detection of critical vulnerabilities and graphs based on the output of the experiments.

5. **Graphs**: Contains all the graphs generated.
