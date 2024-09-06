list1 = [(1, 12), (1, 13), (1, 14), (2, 1), (2, 2), (2, 3), (1, 15), (1, 16), (1, 17), (2, 4)]
list2 = [(1, 12), (1, 13), (1, 14), (2, 1), (2, 2), (2, 3), (1, 11), (1, 15), (1, 16), (1, 17)]

# 转换为集合并计算交集
set1 = set(list1)
set2 = set(list2)
common_elements = set1 & set2

# 输出结果
print(f"共同的元组数量: {len(common_elements)}")
print(f"共同的元组: {common_elements}")
