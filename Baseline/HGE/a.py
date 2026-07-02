import re

def extract_numbers(line):
    # 提取方括号内的数字
    match = re.search(r'\[(.*?)\]', line)
    if match:
        numbers_str = match.group(1)
        # 分割数字字符串
        numbers = numbers_str.split(',')
        result = []
        for num in numbers:
            # 去除 tensor() 包装
            num = re.sub(r'tensor\((.*?)\)', r'\1', num).strip()
            if num:
                try:
                    # 转换为浮点数并保留四位小数
                    num_float = float(num)
                    num_rounded = round(num_float, 4)
                    result.append(num_rounded)
                except ValueError:
                    continue
        return result
    return []

def read_file_and_process(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # 提取标签
                label = line.split(':')[0].strip()
                # 提取数字列表
                numbers = extract_numbers(line)
                data[label] = numbers
    return data

# 示例使用
file_path = 'no-reg.txt'
result = read_file_and_process(file_path)

# 打印结果
for label, numbers in result.items():
    print(f"{label}: {numbers}")