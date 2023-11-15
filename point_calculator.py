sum = 0
time_exceeded = []
for i in range(1,21):
    filename = f"output/{i}.csv"

    file = open(filename, 'r')
    Lines = file.readlines()
    time=int(Lines[-2:][1].strip())
    if time > 1000:
        time_exceeded.append(i)
    else:
        sum+=float(Lines[-2:][0].strip())
print(f"TOTAL SCORE: {sum} , Time exceeded in tests: {time_exceeded}")