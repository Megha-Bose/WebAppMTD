import sys
import numpy as np
import matplotlib.pyplot as plt

DIR = "../Data/output/"

strats_list = ["FPLMTD", "FPLMTDLite", "DOBSS", "Random", "RobustRL", "EXP3", "BSSQ", "PaddedExp3", "SwitchingExp3", "FPLGR"]

if __name__ == "__main__":
    switches_list = []
    runtimes_list = []
    utils_list = []
    details = []

    config_num = []
    attack_num = []
    type_num = []

    n_from = int(sys.argv[1])
    n_to = int(sys.argv[2]) + 1

    for dataset_num in range(n_from, n_to):
        f = open(DIR + str(dataset_num) + "overall_out_fplue.txt")
        config_num.append(int(f.readline()))
        attack_num.append(int(f.readline()))
        type_num.append((f.readline()))

        details.append([config_num, attack_num, type_num])


        f.readline()
        f.readline()
        f.readline()

        switches = []
        for strat in strats_list:
            switches.append(float(f.readline()))
        switches_list.append(switches)


        f.readline()
        f.readline()
        f.readline()

        runtime = []
        for strat in strats_list:
            runtime.append(float(f.readline()))
        runtimes_list.append(runtime)


        f.readline()
        f.readline()
        f.readline()

        util = []
        for strat in strats_list:
            util.append(float(f.readline()))
        utils_list.append(util)
        f.close()

    for strat in range(len(strats_list)):
        switch_arr = []
        for d in range(n_from, n_to):
            switch_arr.append(switches_list[d][strat])
        x = range(n_from, n_to)
        y = switch_arr
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        my_xticks = []
        for d in range(n_from, n_to):
            my_xticks.append('Configs: ' + str(config_num[d]) + '\nAttacks: ' + str(attack_num[d]) + '\nTypes: ' + str(type_num[d]))        
        plt.xticks(x, my_xticks)
        plt.plot(x, y, label = strats_list[strat])
        
    plt.xlabel('Dataset Number')
    plt.ylabel('Switches per iteration')
    plt.title('Switches')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig('graphs/switch/switches_graph_fplue.png', bbox_inches='tight')
    plt.clf()

    for strat in range(len(strats_list)):
        runtime_arr = []
        for d in range(n_from, n_to):
            runtime_arr.append(runtimes_list[d][strat])
        x = range(n_from, n_to)
        y = runtime_arr
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        my_xticks = []
        for d in range(n_from, n_to):
            my_xticks.append('Configs: ' + str(config_num[d]) + '\nAttacks: ' + str(attack_num[d]) + '\nTypes: ' + str(type_num[d]))        
        plt.xticks(x, my_xticks)
        plt.plot(x, y, label = strats_list[strat])
        
    plt.xlabel('Dataset Number')
    plt.ylabel('Runtime per iteration')
    plt.title('Runtime')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig('graphs/runtime/runtimes_graph_fplue.png', bbox_inches='tight')
    plt.clf()

    for strat in range(len(strats_list)):
        util_arr = []
        for d in range(n_from, n_to):
            util_arr.append(utils_list[d][strat])
        x = range(n_from, n_to)
        y = util_arr
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        my_xticks = []
        for d in range(n_from, n_to):
            my_xticks.append('Configs: ' + str(config_num[d]) + '\nAttacks: ' + str(attack_num[d]) + '\nTypes: ' + str(type_num[d]))
        plt.xticks(x, my_xticks)
        plt.plot(x, y, label = strats_list[strat])
        
    plt.xlabel('Dataset Number')
    plt.ylabel('Utilities per iteration')
    plt.title('Utilities')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig('graphs/utility/utilities_graph_fplue.png', bbox_inches='tight')

    
        

