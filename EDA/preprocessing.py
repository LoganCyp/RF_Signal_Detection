import os

# For now we will exclude the UAV Controllers from the data
exclude_path = [r"D:\CARDRF\CARDRF\LOS\Train\UAV_Controller", r"D:\CARDRF\CARDRF\LOS\Test\UAV_Controller"]

def iterate_directory(dir_path):
    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)
        
        if full_path in exclude_path:
            continue

        if os.path.isfile(full_path):
            print(f"File: {full_path}")

        elif os.path.isdir(full_path):
            print(f"Directory: {full_path}")
            iterate_directory(full_path) 

directory_path = r"D:\CARDRF\CARDRF\LOS"
iterate_directory(directory_path)