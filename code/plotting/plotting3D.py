# script permettant de tracer les courbes de conductivités en fonction du volume occupé par le domaine diffusif (cas 3D)
import matplotlib.pyplot as plt

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

yd, se1, se2, se3, si1, si2, si3 = [], [], [], [], [], [], []
for line in open("data_main.dat", "r"):
    lines = [i for i in line.split()]
    yd.append(lines[0])
    se1.append(lines[1])
    se2.append(lines[2])
    se3.append(lines[3])
    si1.append(lines[4])
    si2.append(lines[5])
    si3.append(lines[6])
yd = [float(x) for x in yd]
se1 = [float(x) for x in se1]
se2 = [float(x) for x in se2]
se3 = [float(x) for x in se3]
si1 = [float(x) for x in si1]
si2 = [float(x) for x in si2]
si3 = [float(x) for x in si3]

figure, axis = plt.subplots(2, 2)

axis[1, 0].plot(yd, se1, '-bo', label="sig_e*11")
axis[1, 0].plot(yd, se2, '-go', label="sig_e*22")
axis[1, 0].plot(yd, se3, '-yo', label="sig_e*33")
axis[1, 0].set(xlabel='YD Volume', ylabel='Extracellular conductivities')
axis[1, 0].legend(loc='upper right')

axis[0, 0].plot(yd, si1, '-ro', label="sig_i*11")
axis[0, 0].plot(yd, si2, '-co', label="sig_i*22")
axis[0, 0].plot(yd, si3, '-ko', label="sig_i*33")
axis[0, 0].set(xlabel='YD Volume', ylabel='Intracellular conductivities')
axis[0, 0].legend(loc='upper right')

axis[1, 1].plot(yd, [a/b for a, b in zip(se1, se2)], '-mo', label="sig_e*11/sig_e*22")
axis[1, 1].plot(yd, [a/b for a, b in zip(se1, se3)], '-ro', label="sig_e*11/sig_e*33")
axis[1, 1].set(xlabel='YD Volume', ylabel='Extracellular anisotropy ratio')
axis[1, 1].legend(loc='upper right')

axis[0, 1].plot(yd, [a/b for a, b in zip(si1, si2)], '-ko', label="sig_i*11/sig_i*22")
axis[0, 1].plot(yd, [a/b for a, b in zip(si1, si3)], '-go', label="sig_i*11/sig_i*33")
axis[0, 1].set(xlabel='YD Volume', ylabel='Intracellular anisotropy ratio')
axis[0, 1].legend(loc='upper right')

plt.show()
