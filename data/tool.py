input_file = "/home/hjz/AI漫改/novel/YiShouMiCheng.txt"
output_dir = "/home/hjz/AI漫改/data"

import os

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 八卷标题
volume_titles = [
    "第一卷",
    "第二卷",
    "第三卷",
    "第四卷",
    "第五卷",
    "第六卷",
    "第七卷",
    "第八卷",
]

# 找到每一卷标题的位置
positions = []
for i, title in enumerate(volume_titles):
    pos = content.find(title)
    if pos != -1:
        positions.append((i + 1, pos, title))

# 如果一本都没找到
if not positions:
    print("未找到任何卷标题，请检查文本中的卷名格式。")
else:
    # 按位置切分每一卷
    for idx in range(len(positions)):
        chapter_num, start_pos, title = positions[idx]

        if idx < len(positions) - 1:
            end_pos = positions[idx + 1][1]
            chapter_content = content[start_pos:end_pos]
        else:
            chapter_content = content[start_pos:]  # 最后一卷到文件结尾

        output_file = os.path.join(output_dir, f"chapter{chapter_num}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chapter_content)

        print(f"{title} 已保存到: {output_file}")

print("全部切分完成！")