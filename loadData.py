import pandas
import numpy as np
from datetime import datetime

'''
load data in the format from c
return: index, time, conductivity, turbidity, nitrateMg, nitrateUm
'''
def load_c_data():
    data = pandas.read_csv(
        "data\C_all_final_joined_fixed.csv")
    index = data.index.values
    time = data['Timestamp'].values 
    conductivity = data['Conductivity'].values
    turbidity = data['Turbidity'].values
    nitrateMg = data['mg/L'].values
    nitrateUm = data['uM'].values

    datetimeValues = []
    for i in range(0, time.size):
        datetimeValues.append(datetime.strptime(time[i], '%Y-%m-%dT%H:%M:%S'))
        
    return index, datetimeValues, conductivity, turbidity, nitrateMg, nitrateUm

