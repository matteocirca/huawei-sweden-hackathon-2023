# This is our main
# How to run? Run this file with python3 main.py
import math
import time
import random

# random.seed(42)

file = "case"
input_folder = "testcases/"
output_folder = "output/"

# INPUT VARIABLES
baseline_cost, action_cost = 0, 0
CPU_cost, MEM_cost = 0, 0 # cloud costs
B, CPU, MEM, ACC, cost_per_set = 0, 0, 0, 0, 0 # BBU profile
num_slices, time_horizon, X = 0, 0, 0
# the next will be repeated for each slice s (num_slices repeats)
cu, du, phy = 0, 1, 2
cpu, mem, acc = 0, 1, 2
l_a, l_b, l_c, l_d = 0, 1, 2, 3
CU = {} # CU resource unit
DU = {} # DU resource unit
PHY = {} # PHY resource unit
L = {} # IO cost
T = {} # T traffic unit

# OUTPUT VARIABLES
CLOUD_cost_list = [] # cloud cost
BBU_cost_list = [] # BBU cost
IO_cost_list = [] # IO cost
score = 0 # score
execution_time = 0 # execution time

action_cost_list = [] # action cost

# OTHER VARIABLES
""" 
    for example deployed[0] = {'cu': [0, 1], 'du': [0, 1], 'phy': [0, 1]} means that at time 0 the service was BBB, at time 1 it was CCC 
    1 for cloud, 0 for BBU
"""
deployed = {} # deployed service to keep track of the deployed services over time


"""
    Parse the input file and store the data in the variables
    @param file_name: the name of the input file
"""
def parse_init(file_name):
    global baseline_cost, action_cost, CPU_cost, MEM_cost
    global B, CPU, MEM, ACC, cost_per_set
    global num_slices, time_horizon, X, CU, DU, PHY, L_a, L_b, L_c, L_d, T
    with open(input_folder + file_name) as f:
        baseline_cost, action_cost = map(int, f.readline().split())
        CPU_cost, MEM_cost = map(int, f.readline().split())
        B, CPU, MEM, ACC, cost_per_set = map(int, f.readline().split())
        num_slices, time_horizon, X = map(int, f.readline().split())
        for i in range(num_slices):
            CU[i] = list(map(int, f.readline().split()))
            DU[i] = list(map(int, f.readline().split()))
            PHY[i] = list(map(int, f.readline().split()))
            L[i] = list(map(int, f.readline().split()))
            T[i] = list(map(int, f.readline().split()))

"""
    Print the output in the output file
"""
def clear():
    global baseline_cost,action_cost,CPU_cost,MEM_cost,B,CPU,MEM,ACC,cost_per_set
    global num_slices,time_horizon,X,cu,du,phy,cpu,mem,acc,l_a, l_b, l_c, l_d
    global CU,DU,PHY,L,T,CLOUD_cost_list,BBU_cost_list,IO_cost_list,score,execution_time
    global action_cost_list,deployed
    baseline_cost, action_cost = 0, 0
    CPU_cost, MEM_cost = 0, 0 # cloud costs
    B, CPU, MEM, ACC, cost_per_set = 0, 0, 0, 0, 0 # BBU profile
    num_slices, time_horizon, X = 0, 0, 0
    # the next will be repeated for each slice s (num_slices repeats)
    cu, du, phy = 0, 1, 2
    cpu, mem, acc = 0, 1, 2
    l_a, l_b, l_c, l_d = 0, 1, 2, 3
    CU = {} # CU resource unit
    DU = {} # DU resource unit
    PHY = {} # PHY resource unit
    L = {} # IO cost
    T = {} # T traffic unit

    # OUTPUT VARIABLES
    CLOUD_cost_list = [] # cloud cost
    BBU_cost_list = [] # BBU cost
    IO_cost_list = [] # IO cost
    score = 0 # score
    execution_time = 0 # execution time

    action_cost_list = [] # action cost
    deployed = {}

def deploy2string(s,t):
    return ""+f"{'C' if deployed[s]['cu'][t] == True else 'B'}"+f"{'C' if deployed[s]['du'][t] == True else 'B'}"+f"{'C' if deployed[s]['phy'][t] == True else 'B'}"


def print_output(OPEX=0, score=0, time=0, output=f"output.csv"):
    with open(output_folder + output, 'w') as f:
        for s in range(0, num_slices):
            for t in range(0, time_horizon):
                f.write(deploy2string(s,t) + " ")
            f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{CLOUD_cost_list[t]} ")
        f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{BBU_cost_list[t]} ")
        f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{IO_cost_list[t]} ")
        f.write("\n")
        f.write(f"{OPEX}\n")
        f.write(f"{score}\n")
        f.write(f"{time}")

def cloud_cost_compute(CPU, MEM, ACC, T):
    CPU_req = CPU + X * ACC
    return CPU_req * T * CPU_cost + MEM * T * MEM_cost

def cloud_cost_func(s, t, cu=False, du=False, phy=False):
    cloud_cost = 0

    if cu:
        cloud_cost += cloud_cost_compute(CU[s][cpu], CU[s][mem], CU[s][acc], T[s][t])
    if du:
        cloud_cost += cloud_cost_compute(DU[s][cpu], DU[s][mem], DU[s][acc], T[s][t])
    if phy:
        cloud_cost += cloud_cost_compute(PHY[s][cpu], PHY[s][mem], PHY[s][acc], T[s][t])

    return cloud_cost

def BBU_cost_compute(CPU_allocated_tot, MEM_allocated_tot, ACC_allocated_tot):
    return max(math.ceil(CPU_allocated_tot / CPU), math.ceil(MEM_allocated_tot / MEM), math.ceil(ACC_allocated_tot / ACC)) * cost_per_set

def BBU_cost_func(s, t, cu=False, du=False, phy=False):
    CPU_allocated, MEM_allocated, ACC_allocated = 0, 0, 0

    if cu:
        CPU_allocated += CU[s][cpu] * T[s][t]
        MEM_allocated += CU[s][mem] * T[s][t]
        ACC_allocated += CU[s][acc] * T[s][t]
    if du:
        CPU_allocated += DU[s][cpu] * T[s][t]
        MEM_allocated += DU[s][mem] * T[s][t]
        ACC_allocated += DU[s][acc] * T[s][t]
    if phy:
        CPU_allocated += PHY[s][cpu] * T[s][t]
        MEM_allocated += PHY[s][mem] * T[s][t]
        ACC_allocated += PHY[s][acc] * T[s][t]

    return CPU_allocated, MEM_allocated, ACC_allocated

"""
    Compute the IO cost
    @param s: the slice
    @param cu: CU resource unit
    @param du: DU resource unit
    @param phy: PHY resource unit
    Here true means that the resource unit is allocated on cloud, false otherwise
    @return IO_cost: the IO cost
"""
def IO_cost_func(s,t, cu=False, du=False, phy=False):
    IO_cost = 0

    if cu:
        if du:
            if phy:
                IO_cost = L[s][l_d]
            else:
                IO_cost = L[s][l_c]
        else:
            IO_cost = L[s][l_b]
    else:
        IO_cost = L[s][l_a]

    return IO_cost * T[s][t]

# def action_cost(s, t):
#     service_reloc = 0

#     # check for relocation from deployed dict
#     if deployed[s]['cu'][t] != deployed[s]['cu'][t-1]:
#         service_reloc += 1
#     if deployed[s]['du'][t] != deployed[s]['du'][t-1]:
#         service_reloc += 1
#     if deployed[s]['phy'][t] != deployed[s]['phy'][t-1]:
#         service_reloc += 1

#     return service_reloc * action_cost

def opex():
    global action_cost_list, CLOUD_cost_list, BBU_cost_list, IO_cost_list
    opex = 0
    #print(action_cost_list)
    for t in range(time_horizon):
        opex += CLOUD_cost_list[t] + BBU_cost_list[t] + IO_cost_list[t] + action_cost_list[t]
    return opex

def score_func(OPEX):
    return max(0, baseline_cost/OPEX - 1)


testcase = {
            0: [(False, False, False), (False, False, False), (True, False, False), (True, False, False), (True, True, False)], 
            1: [(False, False, False), (False, False, False), (False, False, False), (False, False, False), (False, False, False)]
        }

def naive():
    global CLOUD_cost_list, BBU_cost_list, IO_cost_list
    global deployed

    CLOUD_cost_list, BBU_cost_list, IO_cost_list = [], [], []
    CPU_allocated_list, MEM_allocated_list, ACC_allocated_list = [], [], []

    for t in range(time_horizon):
        # print("Time: ", t)
        CLOUD_cost_list.append(0)
        BBU_cost_list.append(0)
        IO_cost_list.append(0)
        cloud_cost_tot, BBU_cost_tot, IO_cost_tot = 0, 0, 0
        CPU_allocated_list.append(0)
        MEM_allocated_list.append(0)
        ACC_allocated_list.append(0)
        # TODO: stiamo cambiando il deployed dic ogni volta, ma non dovrebbe essere così (per lo stesso slice s)
        # TODO: check if BBU is full
        for s in range(num_slices):
            # print("Slice: ", s)
            deployed[s]['cu'].append(False)
            deployed[s]['du'].append(False)
            deployed[s]['phy'].append(False)
            min = 10000000000
            cloud_cost_best, IO_cost_best = 0, 0
            CPU_allocated_best, MEM_allocated_best, ACC_allocated_best = 0, 0, 0
            for combination in [[True, True, True], [True, True, False], [True, False, False], [False, False, False]]:
                tot = 0
                # print("Combination: ", combination)
                cloud_cost = cloud_cost_func(s, t, combination[0], combination[1], combination[2])
                # print("Cloud cost: ", cloud_cost)
                # print("BBU cost func: ", s, not(combination[0]), not(combination[1]), not(combination[2]))
                CPU_allocated, MEM_allocated, ACC_allocated = BBU_cost_func(s, t, not(combination[0]), not(combination[1]), not(combination[2]))
                IO_cost = IO_cost_func(s, t, combination[0], combination[1], combination[2])
                # print("IO cost: ", IO_cost)
                BBU_check = math.ceil(max((CPU_allocated_list[t] + CPU_allocated) / CPU, (MEM_allocated_list[t] + MEM_allocated) / MEM, (ACC_allocated_list[t] + ACC_allocated) / ACC))
                tot += cloud_cost + BBU_check * cost_per_set + IO_cost
                if tot < min and BBU_check <= B:
                    min = tot
                    deployed[s]['cu'][t] = combination[cu]
                    deployed[s]['du'][t] = combination[du]
                    deployed[s]['phy'][t] = combination[phy]
                    cloud_cost_best = cloud_cost
                    IO_cost_best = IO_cost
                    CPU_allocated_best = CPU_allocated
                    MEM_allocated_best = MEM_allocated
                    ACC_allocated_best = ACC_allocated
            # print("Min: ", min)
            # print("Deployed (CU, DU, PHY): ", deployed[s]['cu'][t], deployed[s]['du'][t], deployed[s]['phy'][t])
            cloud_cost_tot += cloud_cost_best
            IO_cost_tot += IO_cost_best
            CPU_allocated_list[t] += CPU_allocated_best
            MEM_allocated_list[t] += MEM_allocated_best
            ACC_allocated_list[t] += ACC_allocated_best
        # print("Current cloud list: ", CLOUD_cost_list)
        # print("Current BBU list: ", BBU_cost_list)
        # print("Current IO list: ", IO_cost_list)
        # print("===================================================")
        CLOUD_cost_list[t] = cloud_cost_tot
        # print("CPU allocated: ", CPU_allocated_list[t])
        # print("MEM allocated: ", MEM_allocated_list[t])
        # print("ACC allocated: ", ACC_allocated_list[t])
        BBU_cost_list[t] = BBU_cost_compute(CPU_allocated_list[t], MEM_allocated_list[t], ACC_allocated_list[t])
        IO_cost_list[t] = IO_cost_tot
                

if __name__ == "__main__":

    # parse_init("case2.txt")

    # # print(baseline_cost, action_cost)
    # # print(CPU_cost, MEM_cost)
    # # print(B, CPU, MEM, ACC, cost_per_set)
    # # print(num_slices, time_horizon, CPU_ACC_ratio)
    # # for i in range(num_slices):
    # #     print(CU[i])
    # #     print(DU[i])
    # #     print(PHY[i])
    # #     print(L[i])
    # #     print(T[i])

    # for s in range(num_slices):
    #     deployed[s] = {'cu': [], 'du': [], 'phy': []}
    # start_time = time.time()
    # naive()
    # execution_time += int((time.time() - start_time) * 1000)
    # # compute action cost list
    # for t in range(time_horizon):
    #     action_cost_list.append(0)
    # for s in range(num_slices):
    #     for t in range(1, time_horizon):
    #         if deployed[s]['cu'][t] != deployed[s]['cu'][t-1]:
    #             action_cost_list[t] += action_cost
    #         if deployed[s]['du'][t] != deployed[s]['du'][t-1]:
    #             action_cost_list[t] += action_cost
    #         if deployed[s]['phy'][t] != deployed[s]['phy'][t-1]:
    #             action_cost_list[t] += action_cost
    # opex_computed = opex()
    # score = score_func(opex_computed)

    # print_output(OPEX=opex_computed, score=score, time=execution_time, output="2.csv")


    for i in range(0,20):
        clear()
        parse_init(file + str(i+1) + ".txt")

         # initialize deployed dic
        for s in range(num_slices):
            deployed[s] = {'cu': [], 'du': [], 'phy': []}

        start_time = time.time()

        naive()

        execution_time += int((time.time() - start_time) * 1000)

    #     # compute action cost list
        for t in range(time_horizon):
            action_cost_list.append(0)
        for s in range(num_slices):
            for t in range(1, time_horizon):
                if deployed[s]['cu'][t] != deployed[s]['cu'][t-1]:
                    action_cost_list[t] += action_cost
                if deployed[s]['du'][t] != deployed[s]['du'][t-1]:
                    action_cost_list[t] += action_cost
                if deployed[s]['phy'][t] != deployed[s]['phy'][t-1]:
                    action_cost_list[t] += action_cost
        opex_computed = opex()
        score = score_func(opex_computed)

        print_output(OPEX=opex_computed, score=score, time=execution_time, output=str(i+1) + ".csv")

    
    