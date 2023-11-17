# This is our main
# How to run? Run this file with python3 main.py
import math
import time
#import random

#random.seed(9999)

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

CPU_boundary, MEM_boundary, ACC_boundary = 0,0,0

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
    global CPU_boundary,MEM_boundary,ACC_boundary
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
    CPU_boundary,MEM_boundary,ACC_boundary = B*CPU,B*MEM,B*ACC

"""
    Print the output in the output file
"""
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
    if BBU_check_p > B or (C_t + C_r > CPU_boundary) or (M_t + M_r > MEM_boundary) or ( A_t + A_r > ACC_boundary):
        flag = False
    return flag

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

def opex(action_cost_list, CLOUD_cost_list, BBU_cost_list, IO_cost_list):
    opex = 0
    for t in range(time_horizon):
        opex += CLOUD_cost_list[t] + BBU_cost_list[t] + IO_cost_list[t] + action_cost_list[t]
    return opex

def score_func(OPEX):
    return max(0, baseline_cost/OPEX - 1)


testcase = {
            0: [(False, False, False), (False, False, False), (True, False, False), (True, False, False), (True, True, False)], 
            1: [(False, False, False), (False, False, False), (False, False, False), (False, False, False), (False, False, False)]
        }
def TrafficMean(s,t,t1):
    sum = 0
    count = 0
    for i in range(t,t1):
        try:
            sum+= T[s][i]
            count+=1
        except:
            pass
    #print(sum/count)
    return math.ceil(sum/count)
def nextTrafficMean(s,t):
    sum = 0
    count = 0
    for i in range(t,t+6):
        try:
            sum+= T[s][i]
            count+=1
        except:
            pass
    #print(sum/count)
    return math.ceil(sum/count)

def order_slice_by_traffic(t):
    alpha = 0.8
    traffic_list = []
    for s in range(num_slices):
        traffic_list.append((PHY[s][acc]*nextTrafficMean(s,t),s))
    traffic_list.sort(reverse=True)
    return [x[1] for x in traffic_list]
def order_slice_by_metric(t,service,metric):
    traffic_list = []
    for s in range(num_slices):
        traffic_list.append((service[s][metric]*nextTrafficMean(s,t),s))
    traffic_list.sort(reverse=True)
    return [x[1] for x in traffic_list]

def order_slice_by_IOcost(t):
    traffic_list_phy,traffic_list_du,traffic_list_cu = [],[],[]
    for s in range(num_slices):
        #next = nextTrafficMean(s,t)
        _phy = PHY[s][acc]
        _du = DU[s][acc]
        _cu = CU[s][mem]
        traffic_list_phy.append((_phy,s))
        traffic_list_du.append((_du,s))
        traffic_list_cu.append((_cu,s))
    traffic_list_phy.sort(reverse=True)
    traffic_list_du.sort(reverse=True)
    traffic_list_cu.sort(reverse=True)
    return [x[1] for x in traffic_list_phy],[x[1] for x in traffic_list_du],[x[1] for x in traffic_list_cu]

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


def heuristic():
    global CLOUD_cost_list, BBU_cost_list, IO_cost_list
    global deployed

    CLOUD_cost_list, BBU_cost_list, IO_cost_list = [], [], []
    CPU_allocated_list, MEM_allocated_list, ACC_allocated_list = [], [], []



    for t in range(time_horizon):
        CLOUD_cost_list.append(0)
        BBU_cost_list.append(0)
        IO_cost_list.append(0)
        cloud_cost_tot, BBU_cost_tot, IO_cost_tot = 0, 0, 0
        CPU_allocated_list.append(0)
        MEM_allocated_list.append(0)
        ACC_allocated_list.append(0)
        count = 0
        
        for s in order_slice_by_traffic(t):
            deployed[s]['phy'].append(False)
            deployed[s]['cu'].append(False)
            deployed[s]['du'].append(False)
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(False), not(False), not(False))
            new = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
            old = math.ceil(max((CPU_allocated_list[t]) / CPU, (MEM_allocated_list[t]) / MEM, (ACC_allocated_list[t]) / ACC))
            if  old == new and BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p):
                CPU_allocated_list[t]+=CPU_allocated_p
                MEM_allocated_list[t]+=MEM_allocated_p
                ACC_allocated_list[t]+=ACC_allocated_p
            else:
                deployed[s]['cu'][t] = True
                CPU_allocated_p,MEM_allocated_p,ACC_allocated_p = BBU_cost_func(s,t,not(True),not(False),not(False))
                if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p):
                    CPU_allocated_list[t]+=CPU_allocated_p
                    MEM_allocated_list[t]+=MEM_allocated_p
                    ACC_allocated_list[t]+=ACC_allocated_p
                else:
                    deployed[s]['du'][t] = True
                    CPU_allocated_p,MEM_allocated_p,ACC_allocated_p = BBU_cost_func(s,t,not(True),not(True),not(False))
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p):
                        CPU_allocated_list[t]+=CPU_allocated_p
                        MEM_allocated_list[t]+=MEM_allocated_p
                        ACC_allocated_list[t]+=ACC_allocated_p
                    else:
                        deployed[s]['phy'][t] = True

        count+=1
        BBU_cost_list[t] = BBU_cost_compute(CPU_allocated_list[t], MEM_allocated_list[t], ACC_allocated_list[t])
        if(BBU_cost_list[t]> B*cost_per_set or CPU_allocated_list[t] > CPU_boundary or MEM_allocated_list[t] > MEM_boundary or ACC_allocated_list[t] > ACC_boundary):
            exit("INVALID BBU")
        for s in deployed:
            IO_cost_list[t] += IO_cost_func(s, t, deployed[s]['cu'][t], deployed[s]['du'][t], deployed[s]['phy'][t])
            CLOUD_cost_list[t] += cloud_cost_func(s,t,deployed[s]["cu"][t],deployed[s]["du"][t],deployed[s]["phy"][t])
        

                




            

def getTimeIndex(t,d):

    return math.floor(t/d)

def layerheuristic():
    global CLOUD_cost_list, BBU_cost_list, IO_cost_list
    global deployed
    CPU_boundary, MEM_boundary, ACC_boundary = B*CPU,B*MEM,B*ACC
    CLOUD_cost_list, BBU_cost_list, IO_cost_list = [], [], []
    CPU_allocated_list, MEM_allocated_list, ACC_allocated_list = [], [], []
    count=0
    # phy_order = order_slice_by_traffic(0)
    # du_order = phy_order
    # cpu_order = phy_order

    for t in range(time_horizon):
        CLOUD_cost_list.append(0)
        BBU_cost_list.append(0)
        IO_cost_list.append(0)
        CPU_allocated_list.append(0)
        MEM_allocated_list.append(0)
        ACC_allocated_list.append(0)
        if t%10 == 0:
            phy_order, du_order, cu_order = order_slice_by_IOcost(t)
            

        #cycle for phy
        for s in phy_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
           
            deployed[s]['phy'].append(True)
            if count % 10 == 0:
                CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
                #BBU_check_p = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
                if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and PHY[s][acc]>0: 
                    deployed[s]['phy'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['phy'][t] = True
            else:
                if deployed[s]['phy'][t-1] == False:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and PHY[s][acc]>0: 
                        deployed[s]['phy'][t] = False
                        CPU_allocated_list[t] += CPU_allocated_p
                        MEM_allocated_list[t] += MEM_allocated_p
                        ACC_allocated_list[t] += ACC_allocated_p

                
            # else:
            #     deployed[s]['phy'].append(deployed[s]['phy'][t-1])
            #     if(deployed[s]['phy'][t-1] == False):
            #         CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
            #         if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p):
            #             deployed[s]['phy'][t] == False
            #             CPU_allocated_list[t] += CPU_allocated_p
            #             MEM_allocated_list[t] += MEM_allocated_p
            #             ACC_allocated_list[t] += ACC_allocated_p
            #         else:
            #             deployed[s]['phy'][t] == True
                
                        
        for s in du_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
            deployed[s]['du'].append(True)
            if count % 6 == 0:
                
                CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(False), not(True))
                if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and DU[s][acc]>0 and deployed[s]["phy"][t]==False: 
                    deployed[s]['du'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['du'][t] = True
            else:
                if deployed[s]['du'][t-1] == False:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(False), not(True))
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and DU[s][acc]>0 and deployed[s]["phy"][t]==False: 
                        deployed[s]['du'][t] = False
                        CPU_allocated_list[t] += CPU_allocated_p
                        MEM_allocated_list[t] += MEM_allocated_p
                        ACC_allocated_list[t] += ACC_allocated_p

                
        for s in cu_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
            deployed[s]['cu'].append(True)
            
            #if count % 10 == 0:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(False), not(True), not(True))
            new = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
            old = math.ceil(max((CPU_allocated_list[t]) / CPU, (MEM_allocated_list[t]) / MEM, (ACC_allocated_list[t]) / ACC))
            if new == old:
                if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p)and deployed[s]["phy"][t]==False and deployed[s]["du"][t]==False: 
                    deployed[s]['cu'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['cu'][t] = True
            else:
                deployed[s]['cu'][t] = True

        
            # else:
            #     if deployed[s]['cu'][t-1] == False:
            #         CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(False), not(True), not(True))
            #         if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p)and deployed[s]["phy"][t]==False and deployed[s]["du"][t]==False: 
            #             deployed[s]['cu'][t] = False
            #             CPU_allocated_list[t] += CPU_allocated_p
            #             MEM_allocated_list[t] += MEM_allocated_p
            #             ACC_allocated_list[t] += ACC_allocated_p
        
        # print("CPU allocated: ", CPU_allocated_list[t])
        # print("MEM allocated: ", MEM_allocated_list[t])
        # print("ACC allocated: ", ACC_allocated_list[t])
        count+=1
        BBU_cost_list[t] = BBU_cost_compute(CPU_allocated_list[t], MEM_allocated_list[t], ACC_allocated_list[t])
        
        for s in deployed:
            IO_cost_list[t] += IO_cost_func(s, t, deployed[s]['cu'][t], deployed[s]['du'][t], deployed[s]['phy'][t])
            CLOUD_cost_list[t] += cloud_cost_func(s,t,deployed[s]["cu"][t],deployed[s]["du"][t],deployed[s]["phy"][t])





def STATS():
    sum = 0
    tot_time = 0
    time_exceeded = []
    print("==============================================================")
    for i in range(1,21):
        filename = f"output/{i}.csv"
        
        file = open(filename, 'r')
        Lines = file.readlines()
        time=int(Lines[-2:][1].strip())
        tot_time += time
        if time > 1000:
            print(f"Test {i}: TIME EXCEEDED")
            time_exceeded.append(i)
        else:
            print(f"Test {i}: score  {float(Lines[-2:][0].strip())}, execution {time}")
            sum+=float(Lines[-2:][0].strip())
    print("==============================================================")
    print(f"TOTAL SCORE: {sum} , Total time: {tot_time} ,Time exceeded in tests: {time_exceeded}")
    print("==============================================================")        
            


if __name__ == "__main__":

    for i in range(0,20):
        clear()
        parse_init(file + str(i+1) + ".txt")

         # initialize \ dic
        for s in range(num_slices):
            deployed[s] = {'cu': [], 'du': [], 'phy': []}

        start_time = time.time()

        layerheuristic()

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
        layer_act = action_cost_list.copy()
        layer_cloud = CLOUD_cost_list.copy()
        layer_bbu = BBU_cost_list.copy()
        layer_IO = IO_cost_list.copy()
        layer_opex_computed = opex(action_cost_list,CLOUD_cost_list,BBU_cost_list,IO_cost_list)
        layer_score = score_func(layer_opex_computed)
        h_opex_computed = 0
        h_score = 0
        if execution_time < 300:
            second_time = time.time()
            layer_act = action_cost_list.copy()
            layer_cloud = CLOUD_cost_list.copy()
            layer_bbu = BBU_cost_list.copy()
            layer_IO = IO_cost_list.copy()
            layer_execution = execution_time
            layer_deployed = deployed.copy()
            for s in range(num_slices):
                deployed[s] = {'cu': [], 'du': [], 'phy': []}
            action_cost_list = []
            CLOUD_cost_list = []
            BBU_cost_list = []
            IO_cost_list = []
            heuristic()

            execution_time += int((time.time() - second_time) * 1000)
            
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
            
            h_opex_computed = opex(action_cost_list,CLOUD_cost_list,BBU_cost_list,IO_cost_list)
            h_score = score_func(h_opex_computed)

            if h_score > layer_score:
                print_output(OPEX=h_opex_computed, score=h_score, time=execution_time, output=str(i+1) + ".csv",Cl=CLOUD_cost_list,Bl=BBU_cost_list,IOl=IO_cost_list,_deployed=deployed)
            else:
                print_output(OPEX=layer_opex_computed, score=layer_score, time=execution_time, output=str(i+1) + ".csv",Cl=layer_cloud,Bl=layer_bbu,IOl=layer_IO,_deployed=layer_deployed)


        else:
            print_output(OPEX=layer_opex_computed, score=layer_score, time=execution_time, output=str(i+1) + ".csv",Cl=CLOUD_cost_list,Bl=BBU_cost_list,IOl=IO_cost_list,_deployed=deployed)
    STATS()

    
    