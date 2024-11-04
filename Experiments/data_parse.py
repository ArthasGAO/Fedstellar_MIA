import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_data(file_path, adversary_name, attack_name, topology, round_name, node_name):
    df = pd.read_excel(file_path)
    
    print("Columns in DataFrame:", df.columns)
    print("Data types in DataFrame:", df.dtypes)
    
    try:
        filtered_df = df[(df['Adversary'] == adversary_name) & (df['Name'] == attack_name)]
        print("Filtered by adversary and attack name:")
        print(filtered_df)
        
        filtered_df = filtered_df[filtered_df['Topo'] == topology]
        print("Filtered by topology:")
        print(filtered_df)
        
        filtered_df = filtered_df[filtered_df['Node'] == node_name]
        print("Filtered by node:")
        print(filtered_df)
        
        if not filtered_df.empty:
            # Assuming 'target' is the column you want to retrieve
            target = 'N1'  # Change 'N1' to the actual target column name
            return filtered_df[target].values[0]
        else:
            return None
    except Exception as e:
        print("Error during filtering:", e)
        return None

    
    
DATASET = ["Cifar10no"]#, "Cifar10extend", "Mnist"]#, "Fmnist"]
# MODEL = ["cnn", "mlp"]
TOPOLOGY = ["fully", "star", "ring"]

IID = [1]
ALPHA = 0.1

ROUNDS = range(1, 11, 1) 
EPOCHS_DICT = {1:9, 2:19, 3:29, 4:39, 5:49, 6:59, 7:69, 8:79, 9:89, 10:99}

CLIENT_IDX = range(1, 11, 1)

NUM_CLIENTS = len(CLIENT_IDX)

attack_name = "PC_MIA"

for dataset in DATASET:
    file_path = f'./saved_results/{dataset}/metric_attack_fl_0.xlsx'
                
    for topo in TOPOLOGY:
        
        for iid in IID:
        
            for idx in CLIENT_IDX:
                node_info = f"Node_{idx}"
                data_frames = []
                
                for rou in ROUNDS:
                    round_info = f"Round_{rou}"
                    
                    n1_to_n10 = get_data(file_path, "Adversary 1", attack_name, topo, round_info, node_info)
                    
                    
                    break
                    
                    # print(n1_to_n10)
                
                 # Create DataFrame for heatmap
                '''heatmap_data = pd.DataFrame(data_frames, index=[f"R_{r}" for r in ROUNDS],
                                            columns=[f"N{i}" for i in range(1, 11)])
                
                # Generate heatmap
                plt.figure(figsize=(6, 4))
                sns.heatmap(heatmap_data, annot=False, fmt="d", cmap="Greens", cbar=True, linewidths=.5)
                plt.xlabel("Nodes",fontsize=14, color='darkblue')
                plt.ylabel("Rounds",fontsize=14, color='darkblue')
                
                # Save heatmap
                directory = f'./heatmaps/{attack_name}/{dataset}/{topo}'
                os.makedirs(directory, exist_ok=True)
                
                plt.savefig(f'{directory}/{node_info}.pdf')
                plt.close()
                
                if topo == "fully":
                    break'''
                    
                    
                #break
            #break
        #break
    #break