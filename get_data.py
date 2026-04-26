import os
import shutil

def generate_filenames(numbers, prefix="frame_", suffix=".jpg"):
    """
    根据给定的数字列表生成文件名列表。
    """
    filenames = []
    for number in numbers:
        filename = f"{prefix}{number:05d}{suffix}"  # 将数字格式化为5位数，前面补零
        filenames.append(filename)
    return filenames

def move_images(source_folder, destination_folder, filenames):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in filenames:
        # 构造完整的源文件和目标文件路径
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        # 检查源文件是否存在
        if os.path.exists(source_path):
            print(f"Copying {source_path} to {destination_path}")
            shutil.copy(source_path, destination_path)
        else:
            print(f"File {source_path} does not exist.") 

dataset = 'lerf_ovs'
# data = 'waldo_kitchen'
# data = 'ramen'
# data = 'figurines'
data = 'teatime'
source_path = '/home/hu997372/code/langsplat/data'
source_folder = os.path.join(source_path, dataset, data, 'images')
destination_folder = './data/{}/images'.format(data)
# numbers_to_move = [40, 53, 80, 89]  # waldo_kitchen
# numbers_to_move = [6, 24, 60, 119]  # ramen
# numbers_to_move = [41, 105, 152, 195]  # figurines
numbers_to_move = [2, 25, 107, 140]  # teatime
filenames_to_move = generate_filenames(numbers_to_move)

move_images(source_folder, destination_folder, filenames_to_move)