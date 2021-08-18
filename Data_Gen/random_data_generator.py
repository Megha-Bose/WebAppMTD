import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

attack_list = []
attacker_util = []
defender_util = []

config_num = random.randint(4, 10)
type_num = random.randint(4, 10)
attack_num = random.randint(250, 300)

# list of all vulnerabilities
vul_list = ['v'+str(i) for i in range(attack_num)]

# vulnerability set for each configuration
vul_set = []
for i in range(config_num):
    vul_set.append([])

# update vulnerabilities
for t in range(type_num):
    # sampling attacks the attacker type can execute
    vul_num = random.randint(attack_num - 50, attack_num)
    cves = random.sample(vul_list, vul_num)
    cves.append('NO-OP\n')

    d_utils = [0.0]*len(cves)
    a_utils = [0.0]*len(cves)

    du = [0.0]*len(cves)
    au = [0.0]*len(cves)
    
    # add vulnerability if it can be exploited,
    # i.e., defender utility is non-zero
    for i in range(vul_num):
        du[i] = -random.uniform(0.0, 10.0)
        au[i] = random.uniform(0.0, 10.0)

    for i in range(config_num):
        for j in range(vul_num):
            # effective attack or not
            P_eff = 0.5
            p = random.random()
            if p < P_eff:
                du[j] = au[j] = 0
            if du[j] != 0:
                vul_set[i].append(cves[j])
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

f_util = open("r_utilities.txt", "w")
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

f_vul = open("r_vulnerabilities.txt", "w")
f_vul.write(str(config_num) + "\n")
for i in range(config_num):
    f_vul.write(str(len(vul_set[i])) + "\n")
    for j in range(len(vul_set[i])):
        f_vul.write(str(attack_set.index(vul_set[i][j])) + " ")
    f_vul.write("\n")
f_vul.close()

# all attacks in the system
f_att = open("r_attacks.txt", "w")
f_att.write(str(len(attack_set)) + "\n")
for i in attack_set:
    if(i == "NO-OP\n"):
        f_att.write(i[:-1] + "\n")
        continue
    f_att.write(i + "\n")
f_att.close()

# switching costs
f_sc = open("r_switching.txt", "w")
for i in range(config_num):
    for i in range(config_num):
        sc = round(random.random(), 1)
        f_sc.write(str(sc) + " ")
    f_sc.write("\n")  
f_sc.close()

