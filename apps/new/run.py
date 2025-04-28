import os
import subprocess
import threading
import time
import queue
import os
datasets = ["audio","sift","gist","glove-100","msong","enron"]
# datasets = ["audio"]

basic_ids_e=["1","16","17","18","19","8","2","3","5","1"]
query_ids_e=["1","3_1","3_2","3_3","3_4","4","5_1","5_2","5_3","5_4"]
# basic_ids_e=["2"]
# query_ids_e=["5_1"]
# basic_ids_c=["1_9-15","1-8"]
# query_ids_c=["2_1","2_2"]

# Define the components of the command
dataname = "sift"  
K = 10
alpha = 1.2 
T_s = 1 
T_m = 16
T_b = 32

#filtered equality
R_e = 96 
L_e = 90
#stitched equality
s_R_e = 32
s_L_e = 100
s_SR_e = 64
 
L_values_e = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,600, 650, 700, 750, 800, 850, 900, 950, 1000,1100,1200,1300, 1400]

#filtered containment
R_c = 128 
L_c = 180
#stitched containment
s_R_c = 48
s_L_c = 200
s_SR_c = 96

L_values_c = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,600, 650, 700, 750, 800, 850, 900, 950, 1000,1100,1200,1300, 1400]


def run_commands_e(dataset, basic_id, query_id): 
           
    # Run build_filtered_index
    print(f"Starting to build filtered index for {dataset}...")
    # Ensure the directory exists before running the command
    index_path = f"/data/filter-yjy/diskann/index/{dataset}_{query_id}_filtered/"
    os.makedirs(index_path, exist_ok=True)  # This will create the directory if it doesn't exist
    result_path =f"/data/filter-yjy/diskann/result/{dataset}_{query_id}_filtered/"
    os.makedirs(result_path, exist_ok=True)  # This will create the directory if it doesn't exist

    # subprocess.run([
    #     "build/apps/build_memory_index",
    #     "--data_type", "float",
    #     "--dist_fn", "l2",
    #     "--data_path", f"/data/filter_data/bin/{dataset}_base.bin",
    #     "--index_path_prefix", f"{index_path}_",
    #     "-R", f"{R_e}",
    #     "--FilteredLbuild", f"{L_e}",
    #     "--alpha", f"{alpha}",
    #     "-T", f"{T_b}",
    #     "--label_file", f"/data/filter_data/label/{dataset}_label/comma_basic_{basic_id}.txt"
    # ])
    # filtered单线程搜索 循环遍历 L_values_e
    for L in L_values_e:
        print(f"Starting to search filtered index for {dataset} with L={L} (single thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
            "build/apps/search_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", f"{index_path}_",  # 使用当前 L
            "--query_file", f"/data/filter_data/bin/{dataset}_query.bin",
            "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
            "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
            "-K", f"{K}",
            "-T", f"{T_s}",
            "-L", f"{L}",
            "--result_path", f"{result_path}1_"  # 添加结果路径
        ])
    # filtered16线程搜索 循环遍历 L_values_e
    for L in L_values_e:
        print(f"Starting to search filtered index for {dataset} with L={L} (16 thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
            "build/apps/search_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", f"{index_path}_",  # 使用当前 L
            "--query_file", f"/data/filter_data/bin/{dataset}_query.bin",
            "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
            "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
            "-K", f"{K}",
            "-T", f"{T_m}",
            "-L", f"{L}",
            "--result_path", f"{result_path}16_"  # 添加结果路径
        ])

    #Run build_stitched_index
    print(f"\nStarting to build stitched index for {dataset}...")
    # Ensure the directory exists before running the command
    index_path = f"/data/filter-yjy/diskann/index/{dataset}_{query_id}_stitched/"
    os.makedirs(index_path, exist_ok=True)  # This will create the directory if it doesn't exist
    result_path =f"/data/filter-yjy/diskann/result/{dataset}_{query_id}_stitched/"
    os.makedirs(result_path, exist_ok=True)  # This will create the directory if it doesn't exist

    # subprocess.run([
    #     "build/apps/build_stitched_index",
    #     "--data_type", "float",
    #     "--data_path", f"/data/filter_data/bin/{dataset}_base.bin",
    #     "--index_path_prefix", f"{index_path}_",
    #     "-R", f"{s_R_e}",
    #     "-L", f"{s_L_e}",
    #     "--stitched_R", f"{s_SR_e}",
    #     "--alpha",  f"{alpha}",
    #     "-T", f"{T_b}",
    #     "--label_file", f"/data/filter_data/label/{dataset}_label/comma_basic_{basic_id}.txt"
    # ])

    # stitched单线程搜索 循环遍历 L_values_e
    for L in L_values_e:
        print(f"Starting to search filtered index for {dataset} with L={L} (single thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
        "build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", f"{index_path}_" ,
        "--query_file",f"/data/filter_data/bin/{dataset}_query.bin",
        "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
        "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
        "-K", f"{K}",
        "-T", f"{T_s}",
        "-L", f"{L}",
        "--result_path", f"{result_path}1_"
        ])
     # stitched16线程搜索 循环遍历 L_values_e
    for L in L_values_e:
        print(f"Starting to search filtered index for {dataset} with L={L} (16 thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
        "build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", f"{index_path}_" ,
        "--query_file",f"/data/filter_data/bin/{dataset}_query.bin",
        "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
        "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
        "-K", f"{K}",
        "-T", f"{T_m}",
        "-L", f"{L}",
        "--result_path", f"{result_path}16_"
        ])
   
def run_commands_c(dataset, basic_id, query_id):       
    #  # Run build_filtered_index
    # print(f"Starting to build filtered index for {dataset}...")
    # # Ensure the directory exists before running the command
    # index_path = f"/data/filter-yjy/diskann/index/{dataset}_{query_id}_filtered/"
    # os.makedirs(index_path, exist_ok=True)  # This will create the directory if it doesn't exist
    # result_path =f"/data/filter-yjy/diskann/result/{dataset}_{query_id}_filtered/"
    # os.makedirs(result_path, exist_ok=True)  # This will create the directory if it doesn't exist

    # subprocess.run([
    #     "build/apps/build_memory_index",
    #     "--data_type", "float",
    #     "--dist_fn", "l2",
    #     "--data_path", f"/data/filter_data/bin/{dataset}_base.bin",
    #     "--index_path_prefix", f"{index_path}_",
    #     "-R", f"{R_c}",
    #     "--FilteredLbuild", f"{L_c}",
    #     "--alpha", f"{alpha}",
    #     "-T", f"{T_b}",
    #     "--label_file", f"/data/filter_data/label/{dataset}_label/comma_basic_{basic_id}.txt"
    # ])
    # # filtered单线程搜索 循环遍历 L_values_c
    # for L in L_values_c:
    #     print(f"Starting to search filtered index for {dataset} with L={L} (single thread)...")
    #     # 调用 build/apps/search_memory_index
    #     subprocess.run([
    #         "build/apps/search_memory_index",
    #         "--data_type", "float",
    #         "--dist_fn", "l2",
    #         "--index_path_prefix", f"{index_path}_",  # 使用当前 L
    #         "--query_file", f"/data/filter_data/bin/{dataset}_query.bin",
    #         "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
    #         "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
    #         "-K", f"{K}",
    #         "-T", f"{T_s}",
    #         "-L", f"{L}",
    #         "--result_path", f"{result_path}1_"  # 添加结果路径
    #     ])
    # # filtered16线程搜索 循环遍历 L_values_c
    # for L in L_values_c:
    #     print(f"Starting to search filtered index for {dataset} with L={L} (16 thread)...")
    #     # 调用 build/apps/search_memory_index
    #     subprocess.run([
    #         "build/apps/search_memory_index",
    #         "--data_type", "float",
    #         "--dist_fn", "l2",
    #         "--index_path_prefix", f"{index_path}_",  # 使用当前 L
    #         "--query_file", f"/data/filter_data/bin/{dataset}_query.bin",
    #         "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
    #         "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
    #         "-K", f"{K}",
    #         "-T", f"{T_m}",
    #         "-L", f"{L}",
    #         "--result_path", f"{result_path}16_"  # 添加结果路径
    #     ])
     #Run build_stitched_index
    print(f"\nStarting to build stitched index for {dataset}...")
    # Ensure the directory exists before running the command
    index_path = f"/data/filter-yjy/diskann/index/{dataset}_{query_id}_stitched/"
    os.makedirs(index_path, exist_ok=True)  # This will create the directory if it doesn't exist
    result_path =f"/data/filter-yjy/diskann/result/{dataset}_{query_id}_stitched/"
    os.makedirs(result_path, exist_ok=True)  # This will create the directory if it doesn't exist

    subprocess.run([
        "build/apps/build_stitched_index",
        "--data_type", "float",
        "--data_path", f"/data/filter_data/bin/{dataset}_base.bin",
        "--index_path_prefix", f"{index_path}_",
        "-R", f"{s_R_c}",
        "-L", f"{s_L_c}",
        "--stitched_R", f"{s_SR_c}",
        "--alpha",  f"{alpha}",
        "-T", f"{T_b}",
        "--label_file", f"/data/filter_data/label/{dataset}_label/comma_basic_{basic_id}.txt"
    ])

    # stitched单线程搜索 循环遍历 L_values_c
    for L in L_values_c:
        print(f"Starting to search filtered index for {dataset} with L={L} (single thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
        "build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", f"{index_path}_" ,
        "--query_file",f"/data/filter_data/bin/{dataset}_query.bin",
        "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
        "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
        "-K", f"{K}",
        "-T", f"{T_s}",
        "-L", f"{L}",
        "--result_path", f"{result_path}1_"
        ])
    # stitched16线程搜索 循环遍历 L_values_c
    for L in L_values_c:
        print(f"Starting to search filtered index for {dataset} with L={L} (16 thread)...")
        # 调用 build/apps/search_memory_index
        subprocess.run([
        "build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", f"{index_path}_" ,
        "--query_file",f"/data/filter_data/bin/{dataset}_query.bin",
        "--gt_file", f"/data/filter-yjy/diskann/gt/{dataset}_{query_id}.bin",
        "--query_filters_file", f"/data/filter_data/label/{dataset}_label/comma_query_query_set_{query_id}.txt",
        "-K", f"{K}",
        "-T", f"{T_m}",
        "-L", f"{L}",
        "--result_path", f"{result_path}16_"
        ])
   

# Main function
def main():
    # Loop over datasets and run commands
    for dataset in datasets:
        for basic_id, query_id in zip(basic_ids_e, query_ids_e):
            run_commands_e(dataset, basic_id, query_id)
    # for dataset in datasets:
    #     for basic_id, query_id in zip(basic_ids_c, query_ids_c):
    #         run_commands_c(dataset, basic_id, query_id)

if __name__ == "__main__":
    main()