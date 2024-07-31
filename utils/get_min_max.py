import pandas as pd

data = pd.read_csv('data/chengdu/traj.csv')

wgs = [eval(traj) for traj in list(data.gps)]

min_x = 1000000000
max_x = -1000000000

min_y = 10000000000
max_y = -100000000000

for traj in wgs:
    for x, y in traj:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
print(min_x, max_x, min_y, max_y)
