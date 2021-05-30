import matplotlib.pyplot as plt
import numpy as np

file = open("c:/users/hunterj/desktop/实验数据/starry_night/styleloss.txt", "r")
x = []
for line in file.readlines():
    x.append(float(line.strip('\n')))
print(min(x))
plt.title("Loss")
plt.xlabel("update times/100")
plt.ylabel("variance updated every 100")
line1, = plt.plot(range(1, len(x) + 1), x, color='r', linestyle='--')
file.close()



file = open("c:/users/hunterj/desktop/实验数据/Fire/starry_night/styleloss.txt", "r")
x = []
for line in file.readlines():
    x.append(float(line.strip('\n')))
print(min(x))
line2, = plt.plot(range(1, len(x) + 1), x, color='b')
file.close()
plt.legend([line1, line2], ["Normal", "Fire"], loc='upper left')#添加图例

# file = open("c:/users/hunterj/desktop/实验数据/starry_night/styleloss.txt", "r")
# x = []
# for line in file.readlines():
#     x.append(float(line.strip('\n')))
# print(min(x))
# plt.figure(3)
# plt.subplot(312)
# plt.title("styleloss")
# plt.plot(range(1, len(x) + 1), x)
# file.close()
#
# file = open("c:/users/hunterj/desktop/实验数据/starry_night/totalloss.txt", "r")
# x = []
# for line in file.readlines():
#     x.append(float(line.strip('\n')))
# print(min(x))
# plt.figure(3)
# plt.subplot(313)
# plt.title("totalloss")
# plt.plot(range(1, len(x) + 1), x)
# file.close()

plt.show()
