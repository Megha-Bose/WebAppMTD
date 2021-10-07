import sys
import numpy as np
import json
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

SEED = 2022
DIR = "../Data/input/"

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

if __name__ == "__main__":

    # seeding random number generator  for reproducability
    rng = np.random.default_rng(SEED)

    dataset_from = int(sys.argv[1])
    dataset_to = int(sys.argv[2]) + 1
    start_year = int(sys.argv[3])
    end_year = int(sys.argv[4]) + 1

    n_sets = dataset_to - dataset_from

    config_num_list = rng.integers(low=10, high=20, size=n_sets)
    type_num_list = rng.integers(low=3, high=6, size=n_sets)
    attack_num_list = rng.integers(low=500, high=800, size=n_sets)

    # getting cvss scores from json files
    nvd_vul_bs_score_count = {}
    for i in range(0, 101):
        nvd_vul_bs_score_count[str(i/10.0)] = 0

    nvd_vul_list = []
    bs = []
    for year in range(start_year, end_year):
        f = open('./nvd_data/nvdcve-1.1-' + str(year) + '.json')
        data = json.load(f)
        for vul in data['CVE_Items']:
            if not 'impact' in vul or not 'baseMetricV3' in vul['impact']:
                continue
            if not 'cvssV3' in vul['impact']['baseMetricV3'] or not 'baseScore' in vul['impact']['baseMetricV3']['cvssV3']:
                continue
            if not 'exploitabilityScore' in vul['impact']['baseMetricV3'] or not 'impactScore' in vul['impact']['baseMetricV3']:
                continue

            baseScore = vul['impact']['baseMetricV3']['cvssV3']['baseScore']
            # exploitabiliyScore = vul['impact']['baseMetricV3']['exploitabilityScore']
            impactScore = vul['impact']['baseMetricV3']['impactScore']

            nvd_vul_list.append({'id' : vul['cve']['CVE_data_meta']['ID'], 'bs': baseScore, 'is': impactScore}) 
            indx = round(baseScore, 1)
            bs.append(round(baseScore, 1))
            nvd_vul_bs_score_count[str(indx)] += 1   
        f.close()

    # plotting base score distribution
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.xlabel('Base Score', fontweight='bold')
    plt.ylabel('Number of Vulnerabilities', fontweight='bold')
    plt.hist(bs, bins=100)
    plt.plot()
    plt.savefig('nvd_BS.png')
    plt.show()

    cnt_values = nvd_vul_bs_score_count.values()
    total_cnt = sum(cnt_values)

    nvd_vul_bs_score_prob = {}
    for cnt_val in nvd_vul_bs_score_count:
        nvd_vul_bs_score_prob[cnt_val] = nvd_vul_bs_score_count[cnt_val]/total_cnt

    max_attack_num = len(nvd_vul_list)

    nvd_vul_prob = {}
    nvd_vul_prob_list = []
    for vul in nvd_vul_list:
        nvd_vul_prob[vul['id']] = nvd_vul_bs_score_prob[str(round(vul['bs'], 1))]
        nvd_vul_prob_list.append(nvd_vul_prob[vul['id']])

    for dataset_num in range(dataset_from, dataset_to):
        indx = dataset_num - dataset_from
        config_num = config_num_list[indx]
        type_num = type_num_list[indx]
        attack_num = attack_num_list[indx]

        print("Dataset " + str(dataset_num) + ":")
        attack_list = []
        attacker_util = []
        defender_util = []
        skill_set = []

        # list of all vulnerabilities for the dataset using dist. of base scores from nvd data
        chosen_vul = rng.choice(nvd_vul_list, size = attack_num, replace = False, p = [i/sum(nvd_vul_prob_list) for i in nvd_vul_prob_list])

        vul_list = []
        du = []
        au = []

        nvd_chosen_vul_prob_list = []
        for v in chosen_vul:
            vul_list.append(v['id'])
            du.append(-v['is'])
            au.append(v['bs'])
            nvd_chosen_vul_prob_list.append(nvd_vul_prob[v['id']])

        # # vulnerability set for each configuration
        # all_vul_set = []
        # for i in range(config_num):
        #     vc = []
        #     for vul in nvd_vul_list:
        #         p = rng.random()
        #         if p < nvd_vul_bs_score_prob[str(round(vul['bs'], 1))]:
        #             vc.append(vul['id'])
        #     all_vul_set.append(vc)

        DIR = "../Data/input/general_sum/"


        # update vulnerabilities
        for t in range(type_num):

            # skill of attacker type
            rv = get_truncated_normal(mean=0.5, sd=1, low=0.1, upp=1.0)
            skill = rng.choice(rv.rvs(100))
            skill_set.append(skill)

            # sampling attacks the attacker type can execute
            # normal distribution with mean being proportional to attacker type skill
            rv = get_truncated_normal(mean=attack_num * skill, sd=40, low=300, upp=attack_num)
            att_num = int(rng.choice(rv.rvs(100)))
            
            # choose vulnerabilities that can be exploited by an attacker type 
            # using base score dist. from nvd data
            cves = (rng.choice(vul_list, size = att_num, replace = False, p = [i/sum(nvd_chosen_vul_prob_list) for i in nvd_chosen_vul_prob_list])).tolist()
            cves.append('NO-OP\n')

            d_utils = [0.0]*len(cves)
            a_utils = [0.0]*len(cves)

            # add vulnerability if it can be exploited,
            # i.e., defender utility is non-zero

            vul_set = []
            for i in range(config_num):
                vul_set.append([])
                for j in range(len(cves)):
                    # effective attack or not
                    P_eff = 0.01
                    p = rng.random()
                    if p < P_eff:
                        du[j] = au[j] = 0
                    if du[j] != 0:
                        vul_set[i].append(cves[j])
                        if(cves[j] == "NO-OP\n"):
                            d_utils[j] = a_utils[j] = 0
                        else:
                            d_utils[j] = du[vul_list.index(cves[j])]
                            a_utils[j] = au[vul_list.index(cves[j])]

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

        P_type = []
        psum = 0.0
        for tau in range(type_num):
            P_type.append(rng.random())
            psum += P_type[tau]
            
        for tau in range(type_num):
            P_type[tau] = P_type[tau] / psum

        for t in range(type_num):
            f_util.write(str(P_type[t]) + " ")
        f_util.write("\n")

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
        sc = [ [0.0]*config_num for i in range(config_num)]
        for i in range(config_num):
            for j in range(config_num):
                if i < j:
                    sc[i][j] = round(rng.random(), 1)
                else:
                    sc[i][j] = sc[j][i]
                f_sc.write(str(sc[i][j]) + " ")
            f_sc.write("\n")  
        f_sc.close()

        # attacker skills
        f_sk = open(DIR + str(dataset_num) + "skills.txt", "w")
        for i in range(type_num):
            f_sk.write(str(skill_set[i]) + "\n") 
        f_sk.close()

        print("\n")

