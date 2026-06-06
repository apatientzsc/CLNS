import os

def clear_all_txt_files(folder_path):
    """ss
    清空指定文件夹下所有txt文件的内容
    
    参数:
        folder_path (str): 目标文件夹路径
        
    返回:
        int: 成功清空的文件数量
    """
    count = 0
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return 0
    
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 以写入模式打开文件，这会自动清空文件内容
                with open(file_path, 'w', encoding='utf-8') as file:
                    pass  # 不需要实际写入内容，打开后关闭就会清空
                count += 1
                print(f"已清空: {file_path}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    print(f"操作完成，共清空了 {count} 个txt文件")
    return count

# 使用示例
# clear_all_txt_files('C:/path/to/your/folder')