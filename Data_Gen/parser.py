import numpy as np 

DIR = "../Data/input/"

attack_list = []
attacker_util = []
defender_util = []
f = open("BSSG_input.txt", "r")

config_num = int(f.readline())
type_num = int(f.readline())

vul_set = []
for i in range(config_num):
	vul_set.append([])

for t in range(type_num):

	cves = f.readline()
	cves = f.readline()
	cves = f.readline().split('|')
	d_utils = [0.0]*len(cves)
	a_utils = [0.0]*len(cves)

	for i in range(config_num):
		rews = f.readline().split(' ')
		for j in range(len(rews)):
			temp = rews[j].split(',')
			du = float(temp[0])
			au = float(temp[1])

			if(du != 0):
				vul_set[i].append(cves[j])
				d_utils[j] = du
				a_utils[j] = au 

	attack_list.append(cves)
	attacker_util.append(a_utils)
	defender_util.append(d_utils)


print("Part 1 done")
attack_set = []
for t in range(type_num):
	for i in attack_list[t]:
		if i not in attack_set:
			attack_set.append(i)

f_util = open(DIR + "0utilities.txt", "w")
f_util.write(str(type_num)+"\n")

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
for i in range(config_num):
	vul_set[i] = list(set(vul_set[i]))

f_vul = open(DIR  + "0vulnerabilities.txt", "w")
f_vul.write(str(config_num) + "\n")

for i in range(config_num):
	f_vul.write(str(len(vul_set[i])) + "\n")
	for j in range(len(vul_set[i])):
		f_vul.write(str(attack_set.index(vul_set[i][j])) + " ")
	f_vul.write("\n")

f_vul.close()

f_att = open(DIR + "0attacks.txt", "w")
f_att.write(str(len(attack_set)) + "\n")

for i in attack_set:
	if(i == "NO-OP\n"):
		f_att.write(i[:-1] + "\n")
		continue
	f_att.write(i + "\n")

f_att.close()


