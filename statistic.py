import os

def count_lines_of_code(dirs, extensions):
    total_lines = 0
    for dir in dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                file_line = 0
                for ext in extensions:
                    if file.endswith(ext):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_line += sum(1 for _ in f)
                total_lines += file_line
                if file_line != 0:
                    print("file:%s, line:%d" %(file, file_line))
    return total_lines

directory = ['./src', './include', './python']  # 当前目录
extensions = ['.cpp','.h','.cu','.cuh','.py', '.cc', '.hh']  # 扩展名
total_lines = count_lines_of_code(directory, extensions)
print(f"Total lines of code in {extensions} files: {total_lines}")