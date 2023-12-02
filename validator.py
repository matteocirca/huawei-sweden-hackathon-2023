import math
import time
import random
import pandas
import csv

random.seed(23)

file = "case"
input_folder = "testcases/"
output_folder = "output/"

# INPUT VARIABLES
baseline_cost, action_cost = 0, 0
CPU_cost, MEM_cost = 0, 0 # cloud costs
B, CPU, MEM, ACC, cost_per_set = 0, 0, 0, 0, 0 # BBU profile
num_slices, time_horizon, X = 0, 0, 0
max_action = 200
action_ratio = 0
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

CPU_boundary, MEM_boundary, ACC_boundary = 0,0,0

deployed = {} # deployed service to keep track of the deployed services over time. True for Cloud, False for BBU

def parse_init(file_name):
    global baseline_cost, action_cost, CPU_cost, MEM_cost
    global B, CPU, MEM, ACC, cost_per_set
    global num_slices, time_horizon, X, CU, DU, PHY, L_a, L_b, L_c, L_d, T
    global CPU_boundary,MEM_boundary,ACC_boundary
    with open(file_name) as f:
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
    CPU_boundary,MEM_boundary,ACC_boundary = B*CPU,B*MEM,B*ACC
    action_ratio = action_cost/max_action

def clear():
    global baseline_cost,action_cost,CPU_cost,MEM_cost,B,CPU,MEM,ACC,cost_per_set
    global num_slices,time_horizon,X,cu,du,phy,cpu,mem,acc,l_a, l_b, l_c, l_d
    global CU,DU,PHY,L,T,CLOUD_cost_list,BBU_cost_list,IO_cost_list,score,execution_time
    global action_cost_list,deployed
    global CPU_boundary,MEM_boundary,ACC_boundary
    CPU_boundary,MEM_boundary,ACC_boundary = 0,0,0
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


def deploy2string(s,t,_deployed):
    return ""+f"{'C' if _deployed[s]['cu'][t] == True else 'B'}"+f"{'C' if _deployed[s]['du'][t] == True else 'B'}"+f"{'C' if _deployed[s]['phy'][t] == True else 'B'}"

def print_output(OPEX=0, score=0, time=0, output=f"output.csv", Cl = [], Bl = [], IOl = [],_deployed = {}):
    with open(output_folder + output, 'w') as f:
        for s in range(0, num_slices):
            for t in range(0, time_horizon):
                f.write(deploy2string(s,t,_deployed) + " ")
            f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{Cl[t]} ")
        f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{Bl[t]} ")
        f.write("\n")
        for t in range(0, time_horizon):
                f.write(f"{IOl[t]} ")
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

def BBU_check(C_t,M_t,A_t,C_r,M_r,A_r):
    flag = True
    BBU_check_p = math.ceil(max((C_t + C_r) / CPU, (M_t + M_r) / MEM, (A_t + A_r) / ACC))
    #if BBU_check_p > B or (C_t + C_r > CPU_boundary) or (M_t + M_r > MEM_boundary) or ( A_t + A_r > ACC_boundary):
    if BBU_check_p > B:
        flag = False
    return flag

def BBU_cost_func(s, t, cu=False, du=False, phy=False,mean =False):
    CPU_allocated, MEM_allocated, ACC_allocated = 0, 0, 0
    if not mean:
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
    else:
        if cu:
            CPU_allocated += CU[s][cpu] * mean
            MEM_allocated += CU[s][mem] * mean
            ACC_allocated += CU[s][acc] * mean
        if du:
            CPU_allocated += DU[s][cpu] * mean
            MEM_allocated += DU[s][mem] * mean
            ACC_allocated += DU[s][acc] * mean
        if phy:
            CPU_allocated += PHY[s][cpu] * mean
            MEM_allocated += PHY[s][mem] * mean
            ACC_allocated += PHY[s][acc] * mean

    return CPU_allocated, MEM_allocated, ACC_allocated

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

def opex(action_cost_list, CLOUD_cost_list, BBU_cost_list, IO_cost_list):
    opex = 0
    for t in range(time_horizon):
        opex += CLOUD_cost_list[t] + BBU_cost_list[t] + IO_cost_list[t] + action_cost_list[t]
    return opex

def score_func(OPEX):
    return max(0, baseline_cost/OPEX - 1)


def validate(file):
    import csv


    with open(file, 'r') as file:
        reader = csv.reader(file,delimiter=" ")
        #print(reader)
        # Assuming the first row contains column headers
        computed_opex = 0
        print(time_horizon)
        deployed = {}
        for s in range(num_slices):
            deployed[s] = {"cu":[],"du":[],"phy":[]}
        print(deployed)
        for t in range(time_horizon):
            count = 0
            for row in reader:
                # #if row[t] == "CBB" or row[t] == "BBB" or row[t] == "CCC" or row[t] == "CCB" :
                if count < num_slices:
                    deployed[count]["cu"].append(True)
                    deployed[count]["du"].append(True)
                    deployed[count]["phy"].append(True)

                    if row[t][0] == "B":
                        deployed[count]["cu"][t] = False
                    if row[t][1] == "B":
                        deployed[count]["du"][t] = False
                    if row[t][2] == "B":
                        deployed[count]["phy"][t] = False
                if count == num_slices+1:
                    computed_cc = row[t]
                if count == num_slices+2:
                    print(count,t,row)
                    computed_bbu = row[t]
                if count == num_slices+3:
                    computed_io = row[t]
                if count == num_slices+4 and t == 0:
                    computed_opex = row[t]
                
                
                count+=1
       
                

                                
            file.seek(0)
            next(reader) 
        print(computed_opex)

    return []



if __name__ == "__main__":

    for i in range(0,1):
        clear()
        parse_init("testcases/case"+ str(i+1) + ".txt")
        #parse_output("output/case"+ str(i+1) + ".txt")
        validate("output/"+ str(i+1) + ".csv")
    
