import os

def print_tree(root_dir, indent=0, output_lines=None):
    if output_lines is None:
        output_lines = []
    items = sorted(os.listdir(root_dir))
    for name in items:
        path = os.path.join(root_dir, name)
        prefix = ' ' * (indent * 2)
        if os.path.isdir(path):
            output_lines.append(f"{prefix}[D] {name}")
            print_tree(path, indent + 1, output_lines)
        else:
            output_lines.append(f"{prefix}[F] {name}")
    return output_lines

if __name__ == "__main__":
    root = "."  # 当前目录（脚本所在目录）即项目根
    lines = print_tree(root)
    with open("directory_structure.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))