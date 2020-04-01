import pandas as pd

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# data = df.head()#默认读取前5行的数据
# print(data)#格式化输出

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# data=df.iloc[0].values #0表示第一行 这里读取数据并不包含表头，要注意哦！
# print(data)


# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# data = df.iloc[1, 2]#读取第2行第3列的值，这里不需要嵌套列表
# print(data)
#
# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# data = df.iloc[:, [1, 2]].values  # 读所有行的第2列和第3列的值，这里需要嵌套列表
# print("读取指定行的数据：\n{0}".format(data))

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# list = df['Time'].values
# a = str(list[0])
# print(a)
# if a[8] is not '2':
#     a[6] = '1'
#     print(a)

# for i in range(len(list)):
#     print(list[i])     # 获取指定列Time的值

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# data = df.iloc[[0, 1], [1, 2]].values  # 读1,2行的第2列和第3列的值，这里需要嵌套列表
# print("读取指定行的数据：\n{0}".format(data))

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# print("输出行号列表", df.index.values)

# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# print("输出列标题", df.columns.values)


# df = pd.read_excel("D:/python_workplace/csv_mag/data.xlsx")
# print("输出值", df.sample(2).values)  # 获取指定前两行的值

# df = pd.DataFrame({'Data1': [2012, 2013]})
# write = pd.ExcelWriter("D:/python_workplace/csv_mag/data.xlsx")
# df.to_excel(write, startcol=1, index=True)
#


# writer = pd.ExcelWriter('D:/python_workplace/csv_mag/data.xlsx')
# df1 = pd.DataFrame(data={'City': ['ssss', 'aaa']})
# df1.to_excel(writer, 'Data')
# writer.save()
