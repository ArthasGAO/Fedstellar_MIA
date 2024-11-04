import pandas as pd
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook
from openpyxl import Workbook
import numpy as np

# Load the Excel file
#df = pd.read_excel('./saved_results/Cifar10/class_metric_0.xlsx')

# Ensure that the Size column is correctly formatted
#df['Size'] = df['Size'].astype(int)

# Print the data types of the DataFrame columns to debug the issue
#print(df.dtypes)
#print(df)

# Function to get target value based on Adversary, Name, Size, and Epochs
def get_target_value(adversary, name, topo, iid, rou, target):
    filtered_df = df[df['Name'] == name]
    #print(filtered_df)
    
    filtered_df = filtered_df[filtered_df['Topo'] == topo]
    #print(filtered_df)
    
    filtered_df = filtered_df[filtered_df['iid'] == iid]
    filtered_df = filtered_df[filtered_df['Round'] == rou]
    #print(filtered_df)
    
    if not filtered_df.empty:
        return filtered_df[target].values
    else:
        return None

def append_experiment_results(file_name, data):
    wb = load_workbook(file_name)
    ws = wb.active
    ws.append(data)
    wb.save(file_name)


DATASET = ["Cifar10no", "Cifar10","Cifar10extend"]
# Example usage
#file_path = ['./saved_results/Cifar10/class_metric_0.xlsx', './saved_results/Cifar10extend/class_metric_0.xlsx']

adversary = ['Adversary 1']
name = ['CLC MIA', "CLE MIA", "MCLE MIA"]
topo = ['fully', "star","ring"]
ROUND = [10]
target = ["AP", "AR"]

df = pd.read_excel('./saved_results/Cifar10no/shadow_attack_fl_0.xlsx')
#df['Size'] = df['Size'].astype(int)


for dataset in DATASET:
    for ad in adversary:
        for na in name:
            if ad == 'Adversary 1':
                file_path = f'./saved_results/{dataset}/class_metric_fl_0.xlsx'
            else:
                file_path = f'./saved_results/{dataset}/class_metric_1.xlsx'
                
            df = pd.read_excel(file_path)
            
            for to in topo:
                
                round_ap = []; round_ar = []
                for rou in ROUND:
                    target1 = get_target_value('Adversary 1', na,to, 1, f"Round_{rou}","AP")
                    target2 = get_target_value('Adversary 1', na,to, 1, f"Round_{rou}","AR")
                    
                    if target1 is not None and target1.size > 0:
                        num = np.mean(target1)
                    else:
                        num = 0
                    round_ap.append(num)  
                     
                    if target2 is not None and target2.size > 0:
                        num = np.mean(target2)
                    else:
                        num = 0
                        
                    round_ar.append(num)
                    
                print(round_ap)
                print(round_ar)  
                append_experiment_results("data_collecting.xlsx", round_ap)
                append_experiment_results("data_collecting.xlsx", round_ar)
                #break
            #break
        #break
    #break

                
#print(clc_ls)       
#print(cle_ls) 
#print(mcle_ls)  '=     
