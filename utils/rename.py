import os #导入模块







filename = './data1_0102' #文件地址
list_path = os.listdir(filename)  #读取文件夹里面的名字
number = 1
for index in list_path:  #list_path返回的是一个列表   通过for循环遍历提取元素
    
    path = filename + '/' + index  # 原本文件名
    new_path = filename + '/' + f'20220331_{number}.bag'
    print(new_path)
    os.rename(path, new_path)
    number += 1
    
    

print('修改完成')


