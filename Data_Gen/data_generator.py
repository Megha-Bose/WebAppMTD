import sys
import numpy as np
import random
from scipy.stats import truncnorm

DIR = "../Data/input/"

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

if __name__ == "__main__":
    for dataset_num in range(1, int(sys.argv[1])+1):
        print("Dataset " + str(dataset_num) + ":")
        attack_list = []
        attacker_util = []
        defender_util = []

        config_num = random.randint(10, 20)
        type_num = random.randint(3, 6)
        attack_num = random.randint(500, 800)

        # list of all vulnerabilities
        vul_list = ['v'+str(i) for i in range(attack_num)]

        # vulnerability set for each configuration
        vul_set = []
        for i in range(config_num):
            vul_set.append([])

        du = [0.0]*len(vul_list)
        au = [0.0]*len(vul_list)

        for i in range(attack_num):
            du[i] = -random.random() * 10.0
            au[i] = random.random() * 10.0


        # update vulnerabilities
        for t in range(type_num):
            # sampling attacks the attacker type can execute
            # normal distribution with half of atttack_num as mean
            rv = get_truncated_normal(mean=attack_num // 2, sd=50, low=240, upp=attack_num)
            vul_num = int(random.choice(rv.rvs(100)))
            
            cves = random.sample(vul_list, vul_num)
            cves.append('NO-OP\n')

            d_utils = [0.0]*len(cves)
            a_utils = [0.0]*len(cves)

            # add vulnerability if it can be exploited,
            # i.e., defender utility is non-zero

            for i in range(config_num):
                for j in range(len(cves)):
                    # effective attack or not
                    P_eff = 0.05
                    p = random.random()
                    if p < P_eff:
                        du[j] = au[j] = 0
                    if du[j] != 0:
                        vul_set[i].append(cves[j])
                        if(cves[j] == "NO-OP\n"):
                            d_utils[j] = a_utils[j] = 0
                        else:
                            d_utils[j] = du[j]
                            a_utils[j] = au[j]

            # update attacks and defender, attacker utilities 
            attack_list.append(cves)
            attacker_util.append(a_utils)
            defender_util.append(d_utils)

        # ensuing attacks are not considered multiple times
        attack_set = []
        for t in range(type_num):
            for i in attack_list[t]:
                if i not in attack_set:
                    attack_set.append(i)

        f_util = open(DIR + str(dataset_num) + "utilities.txt", "w")
        f_util.write(str(type_num)+"\n")
        # get final utilities
        for t in range(type_num):
            final_d_util = [0.0]*len(attack_set)
            final_a_util = [0.0]*len(attack_set)
            count = 0
            for i in range(len(attack_list[t])):
                if(defender_util[t][i] != 0):
                    count +=1
                    ind = attack_set.index(attack_list[t][i])
                    final_d_util[ind] = defender_util[t][i]/10
                    final_a_util[ind] = attacker_util[t][i]/10
            ind = attack_set.index("NO-OP\n")
            final_a_util[ind] += 0.0000001
            print(count)

            for i in range(len(attack_set)):
                f_util.write(str(round(final_d_util[i], 2)) + " ")
            f_util.write("\n")

            for i in range(len(attack_set)):
                f_util.write(str(round(final_a_util[i], 2)) + " ")
            f_util.write("\n")
        f_util.close()

        # exploitable vulnerabilities in each config
        for i in range(config_num):
            vul_set[i] = list(set(vul_set[i]))

        f_vul = open(DIR + str(dataset_num) + "vulnerabilities.txt", "w")
        f_vul.write(str(config_num) + "\n")
        for i in range(config_num):
            f_vul.write(str(len(vul_set[i])) + "\n")
            for j in range(len(vul_set[i])):
                f_vul.write(str(attack_set.index(vul_set[i][j])) + " ")
            f_vul.write("\n")
        f_vul.close()

        # all attacks in the system
        f_att = open(DIR + str(dataset_num) + "attacks.txt", "w")
        f_att.write(str(len(attack_set)) + "\n")
        for i in attack_set:
            if(i == "NO-OP\n"):
                f_att.write(i[:-1] + "\n")
                continue
            f_att.write(i + "\n")
        f_att.close()

        # switching costs
        f_sc = open(DIR + str(dataset_num) + "switching.txt", "w")
        for i in range(config_num):
            for i in range(config_num):
                sc = round(random.random(), 1)
                f_sc.write(str(sc) + " ")
            f_sc.write("\n")  
        f_sc.close()

        print("\n")

