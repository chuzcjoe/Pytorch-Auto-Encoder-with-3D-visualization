import csv
import os

res = []
with open('results.txt') as f:
    for line in f.readlines():
        res.append(line.rstrip().split(' '))


with open('results.csv',mode='w') as csv_file:
    fieldnames = ['X','Y','Z','Object']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for r in res:
        writer.writerow({'X':r[1],'Y':r[2],'Z':r[3],'Object':r[0].split('_')[0]})

print('Done!')
