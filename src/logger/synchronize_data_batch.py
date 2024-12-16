import os

path = input("Insert path of parent directory:")

folders = os.listdir(path)

for folder in folders:
    folder_path = os.path.join(path, folder)

    os.system('python3 syncronize_data.py ' + folder_path + ' --sampling_freq 20 --resize_to "(224,224)"')
    