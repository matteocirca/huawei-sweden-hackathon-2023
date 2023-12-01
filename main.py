# This is our main
# How to run? Run this file with python3 main.py
import math
import time
import random

random.seed(23)

file = "case"
input_folder = "testcases/"
output_folder = "output/"

# INPUT VARIABLES
baseline_cost, transition_time, bandwith, P = 0, 0, 0, 0
CPU_cost, MEM_cost = 0, 0 # cloud costs
B, CPU, MEM, ACC, cost_per_set = 0, 0, 0, 0, 0 # BBU profile
num_slices, time_horizon, X = 0, 0, 0
max_action = 200
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

CPU_boundary, MEM_boundary, ACC_boundary = 0,0,0

deployed = {} # deployed service to keep track of the deployed services over time. True for Cloud, False for BBU

def parse_init(file_name, input_folder="testcases/"):
    global baseline_cost, transition_time, bandwith, P, CPU_cost, MEM_cost
    global B, CPU, MEM, ACC, cost_per_set
    global num_slices, time_horizon, X, CU, DU, PHY, L_a, L_b, L_c, L_d, T
    global CPU_boundary,MEM_boundary,ACC_boundary
    with open(input_folder + file_name) as f:
        baseline_cost, transition_time, bandwith, P = map(int, f.readline().split())
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

def clear():
    global baseline_cost,transition_time,bandwith,P,CPU_cost,MEM_cost,B,CPU,MEM,ACC,cost_per_set
    global num_slices,time_horizon,X,cu,du,phy,cpu,mem,acc,l_a, l_b, l_c, l_d
    global CU,DU,PHY,L,T,CLOUD_cost_list,BBU_cost_list,IO_cost_list,score,execution_time
    global deployed
    global CPU_boundary,MEM_boundary,ACC_boundary
    CPU_boundary,MEM_boundary,ACC_boundary = 0,0,0
    baseline_cost, transition_time, bandwith, P = 0, 0, 0, 0
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
    if BBU_check_p > (B - math.floor(B * 0.4)):
    # if BBU_check_p > B:
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

def opex(CLOUD_cost_list, BBU_cost_list, IO_cost_list):
    opex = 0
    for t in range(time_horizon):
        opex += CLOUD_cost_list[t] + BBU_cost_list[t] + IO_cost_list[t]
    return opex

def score_func(OPEX):
    return max(0, baseline_cost/OPEX - 1)


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

def order_slice_by_IOcost2(t):
    traffic_list_phy,traffic_list_du,traffic_list_cu = [],[],[]
    for s in range(num_slices):
        benefit = 0
        if t>0:
            benefit = 10 if (deployed[s]["phy"][t-1] == False)and(deployed[s]["du"][t-1] == False) else 0
            benefit = benefit-1 if PHY[s][acc] < 3 else benefit
        #next = nextTrafficMean(s,t)
        _phy = PHY[s][acc]+DU[s][acc]+benefit

        traffic_list_phy.append((_phy,s))

    traffic_list_phy.sort(reverse=True)

    return [x[1] for x in traffic_list_phy]

def order_slice_by_IOcost(t):
    traffic_list_phy,traffic_list_du,traffic_list_cu = [],[],[]
    for s in range(num_slices):
        benefit  = 0
        # if t>0:
        #     benefit = 10 if (deployed[s]["phy"][t-1] == False) else 0
        #     #benefit = benefit-1 if T[s][t] < 3 else benefit
        #     benefit = benefit-1 if PHY[s][acc] < 3 else benefit
        #next = nextTrafficMean(s,t)
        # _phy = PHY[s][acc]+benefit
        # _du = DU[s][acc]+benefit
        # _cu = CU[s][mem]+benefit
        _phy = L[s][l_c]
        _du = DU[s][l_b]
        _cu = CU[s][l_a]
        traffic_list_phy.append((_phy,s))
        traffic_list_du.append((_du,s))
        traffic_list_cu.append((_cu,s))
    traffic_list_phy.sort(reverse=True)
    traffic_list_du.sort(reverse=True)
    traffic_list_cu.sort(reverse=True)
    return [x[1] for x in traffic_list_phy],[x[1] for x in traffic_list_du],[x[1] for x in traffic_list_cu]


def layerheuristic():
    global CLOUD_cost_list, BBU_cost_list, IO_cost_list
    global deployed
    CPU_boundary, MEM_boundary, ACC_boundary = B*CPU,B*MEM,B*ACC
    CLOUD_cost_list, BBU_cost_list, IO_cost_list = [], [], []
    CPU_allocated_list, MEM_allocated_list, ACC_allocated_list = [], [], []
    count=0


    for t in range(time_horizon):
        CLOUD_cost_list.append(0)
        BBU_cost_list.append(0)
        IO_cost_list.append(0)
        CPU_allocated_list.append(0)
        MEM_allocated_list.append(0)
        ACC_allocated_list.append(0)
        if t % 10 ==  0:
            phy_order, du_order, cu_order = order_slice_by_IOcost(t)
            

        #cycle for phy
        for s in phy_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
           
            deployed[s]['phy'].append(True)
            if deployed[s]['phy_lock'] == 0:
                if count % 10 == 0:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
                    #BBU_check_p = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and PHY[s][acc]>0: 
                        deployed[s]['phy'][t] = False
                        CPU_allocated_list[t] += CPU_allocated_p
                        MEM_allocated_list[t] += MEM_allocated_p
                        ACC_allocated_list[t] += ACC_allocated_p
                else:
                    if deployed[s]['phy'][t-1] == False:
                        CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
                        if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and PHY[s][acc]>0: 
                            deployed[s]['phy'][t] = False
                            CPU_allocated_list[t] += CPU_allocated_p
                            MEM_allocated_list[t] += MEM_allocated_p
                            ACC_allocated_list[t] += ACC_allocated_p

                deployed[s]['phy_lock'] = transition_time+1 if deployed[s]['phy'][t] != deployed[s]['phy'][t-1] and t!=0 else 0
            else:
                if deployed[s]['phy'][t-1] == False:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(True), not(False))
                    deployed[s]['phy'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['phy'][t] = True

                
                        
        for s in du_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0

            deployed[s]['du'].append(True)
            if deployed[s]['du_lock'] == 0:
                if count % 6 == 0:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(False), not(True))
                    new = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
                    old = math.ceil(max((CPU_allocated_list[t]) / CPU, (MEM_allocated_list[t]) / MEM, (ACC_allocated_list[t]) / ACC))
                    
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and DU[s][acc]>0 and deployed[s]["phy"][t]==False: 
                        if True:
                            deployed[s]['du'][t] = False
                            CPU_allocated_list[t] += CPU_allocated_p
                            MEM_allocated_list[t] += MEM_allocated_p
                            ACC_allocated_list[t] += ACC_allocated_p
                        else:
                            if new == old:
                                deployed[s]['du'][t] = False
                                CPU_allocated_list[t] += CPU_allocated_p
                                MEM_allocated_list[t] += MEM_allocated_p
                                ACC_allocated_list[t] += ACC_allocated_p
                            else:
                                deployed[s]['du'][t] = True
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

                deployed[s]['du_lock'] = transition_time+1 if deployed[s]['du'][t] != deployed[s]['du'][t-1] and t!=0 else 0
                deployed[s]['phy_lock'] = transition_time+1 if deployed[s]['du'][t] != deployed[s]['du'][t-1] and deployed[s]['du'][t] == False and t!=0 else deployed[s]['phy_lock']
            else:
                if deployed[s]['du'][t-1] == False:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(False), not(True))
                    deployed[s]['du'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['du'][t] = True


                
        for s in cu_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
            deployed[s]['cu'].append(True)
            
            if deployed[s]['cu_lock'] == 0:
                CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(False), not(True), not(True))
                new = math.ceil(max((CPU_allocated_list[t] + CPU_allocated_p) / CPU, (MEM_allocated_list[t] + MEM_allocated_p) / MEM, (ACC_allocated_list[t] + ACC_allocated_p) / ACC))
                old = math.ceil(max((CPU_allocated_list[t]) / CPU, (MEM_allocated_list[t]) / MEM, (ACC_allocated_list[t]) / ACC))
                if new == old:
                    if BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and deployed[s]["phy"][t]==False and deployed[s]["du"][t]==False: 
                        deployed[s]['cu'][t] = False
                        CPU_allocated_list[t] += CPU_allocated_p
                        MEM_allocated_list[t] += MEM_allocated_p
                        ACC_allocated_list[t] += ACC_allocated_p
                    else:
                        deployed[s]['cu'][t] = True
                else:
                    deployed[s]['cu'][t] = True
                
                deployed[s]['cu_lock'] = transition_time+1 if deployed[s]['cu'][t] != deployed[s]['cu'][t-1] and t!=0 else 0
                deployed[s]['phy_lock'] = transition_time+1 if deployed[s]['cu'][t] != deployed[s]['cu'][t-1] and deployed[s]['cu'][t] == False and t!=0 else deployed[s]['phy_lock']
                deployed[s]['du_lock'] = transition_time+1 if deployed[s]['cu'][t] != deployed[s]['cu'][t-1] and deployed[s]['cu'][t] == False and t!=0 else deployed[s]['du_lock']
            else:
                if deployed[s]['cu'][t-1] == False:
                    CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(False), not(True), not(True))
                    deployed[s]['cu'][t] = False
                    CPU_allocated_list[t] += CPU_allocated_p
                    MEM_allocated_list[t] += MEM_allocated_p
                    ACC_allocated_list[t] += ACC_allocated_p
                else:
                    deployed[s]['cu'][t] = True

        count+=1
        BBU_cost_list[t] = BBU_cost_compute(CPU_allocated_list[t], MEM_allocated_list[t], ACC_allocated_list[t])
        
        for s in deployed:
            deployed[s]['phy_lock'] = max(0, deployed[s]['phy_lock'] - 1)
            deployed[s]['du_lock'] = max(0, deployed[s]['du_lock'] - 1)
            deployed[s]['cu_lock'] = max(0, deployed[s]['cu_lock'] - 1)

            IO_cost_list[t] += IO_cost_func(s, t, deployed[s]['cu'][t], deployed[s]['du'][t], deployed[s]['phy'][t])
            CLOUD_cost_list[t] += cloud_cost_func(s,t,deployed[s]["cu"][t],deployed[s]["du"][t],deployed[s]["phy"][t])

        IO_cost_list[t] += max(0, IO_cost_list[t] - bandwith) * P

def semilayerheuristic():
    global CLOUD_cost_list, BBU_cost_list, IO_cost_list
    global deployed
    CPU_boundary, MEM_boundary, ACC_boundary = B*CPU,B*MEM,B*ACC
    CLOUD_cost_list, BBU_cost_list, IO_cost_list = [], [], []
    CPU_allocated_list, MEM_allocated_list, ACC_allocated_list = [], [], []
    count=0


    for t in range(time_horizon):
        CLOUD_cost_list.append(0)
        BBU_cost_list.append(0)
        IO_cost_list.append(0)
        CPU_allocated_list.append(0)
        MEM_allocated_list.append(0)
        ACC_allocated_list.append(0)
        if t % 10 == 0:
            phy_order = order_slice_by_IOcost2(t)
            

        #cycle for phy
        for s in phy_order:
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = 0,0,0
           
            deployed[s]['phy'].append(True)
            deployed[s]['du'].append(True)
           
            CPU_allocated_p, MEM_allocated_p, ACC_allocated_p = BBU_cost_func(s, t, not(True), not(False), not(False))
            if (BBU_check(CPU_allocated_list[t],MEM_allocated_list[t],ACC_allocated_list[t],CPU_allocated_p,MEM_allocated_p,ACC_allocated_p) and PHY[s][acc]>0) : 
                deployed[s]['phy'][t] = False
                deployed[s]['du'][t] = False
                CPU_allocated_list[t] += CPU_allocated_p
                MEM_allocated_list[t] += MEM_allocated_p
                ACC_allocated_list[t] += ACC_allocated_p
            else:
                deployed[s]['phy'][t] = True
                deployed[s]['du'][t] = True


        for s in phy_order:
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
            print(f"Test {i}: score  {float(Lines[-2:][0].strip())} TIME EXCEEDED")
            time_exceeded.append(i)
        else:
            print(f"Test {i}: score  {float(Lines[-2:][0].strip())}, execution {time}")
            sum+=float(Lines[-2:][0].strip())
    print("==============================================================")
    print(f"TOTAL SCORE: {sum}, Total time: {tot_time}, Time exceeded in tests: {time_exceeded}")
    print("==============================================================")        
            


if __name__ == "__main__":

    for i in range(0,20):
        clear()
        parse_init(file + str(i+1) + ".txt", input_folder="Final_Kit/testcases/")

        # initialize \ dic
        for s in range(num_slices):
            deployed[s] = {'cu': [], 'du': [], 'phy': [], 'cu_lock': int(0), 'du_lock': int(0), 'phy_lock': int(0)}

        start_time = time.time()

        layerheuristic()

        execution_time += int((time.time() - start_time) * 1000)



        layer_cloud = CLOUD_cost_list.copy()
        layer_bbu = BBU_cost_list.copy()
        layer_IO = IO_cost_list.copy()
        layer_opex_computed = opex(CLOUD_cost_list,BBU_cost_list,IO_cost_list)
        layer_score = score_func(layer_opex_computed)

        h_opex_computed = 0
        h_score = 0
        if False:
        # if execution_time < 500:
            second_time = time.time()
            layer_cloud = CLOUD_cost_list.copy()
            layer_bbu = BBU_cost_list.copy()
            layer_IO = IO_cost_list.copy()
            layer_execution = execution_time
            layer_deployed = deployed.copy()
            for s in range(num_slices):
                deployed[s] = {'cu': [], 'du': [], 'phy': [], 'cu_lock': int(0), 'du_lock': int(0), 'phy_lock': int(0)}
            CLOUD_cost_list = []
            BBU_cost_list = []
            IO_cost_list = []

            semilayerheuristic()

            execution_time = int((time.time() - start_time) * 1000)
            
            h_opex_computed = opex(CLOUD_cost_list,BBU_cost_list,IO_cost_list)
            h_score = score_func(h_opex_computed)

            if h_score > layer_score:
                print_output(OPEX=h_opex_computed, score=h_score, time=execution_time, output=str(i+1) + ".csv",Cl=CLOUD_cost_list,Bl=BBU_cost_list,IOl=IO_cost_list,_deployed=deployed)
            else:
                print_output(OPEX=layer_opex_computed, score=layer_score, time=execution_time, output=str(i+1) + ".csv",Cl=layer_cloud,Bl=layer_bbu,IOl=layer_IO,_deployed=layer_deployed)

        else:
            print_output(OPEX=layer_opex_computed, score=layer_score, time=execution_time, output=str(i+1) + ".csv",Cl=CLOUD_cost_list,Bl=BBU_cost_list,IOl=IO_cost_list,_deployed=deployed)
    STATS()

    # file = "toy_example_final"

    # clear()

    # parse_init(file + ".txt", input_folder="Final_Kit/")

    # # initialize \ dic
    # for s in range(num_slices):
    #     deployed[s] = {'cu': [], 'du': [], 'phy': [], 'cu_lock': int(0), 'du_lock': int(0), 'phy_lock': int(0)}

    # start_time = time.time()

    # layerheuristic()

    # execution_time += int((time.time() - start_time) * 1000)

    # layer_cloud = CLOUD_cost_list.copy()
    # layer_bbu = BBU_cost_list.copy()
    # layer_IO = IO_cost_list.copy()
    # layer_opex_computed = opex(CLOUD_cost_list,BBU_cost_list,IO_cost_list)
    # layer_score = score_func(layer_opex_computed)

    # print_output(OPEX=layer_opex_computed, score=layer_score, time=execution_time, output="toy_example_final.csv",Cl=CLOUD_cost_list,Bl=BBU_cost_list,IOl=IO_cost_list,_deployed=deployed)

    
    