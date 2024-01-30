from copy import copy
import numpy as np
import mesh as m  # Mesh and P1 finite elements
import fem_p1 as p1
from scipy.sparse.linalg import spsolve, factorized, cg, bicgstab  # direct and iterative sparse solvers

path = '/home/carmen/VB/code/meshes/CASE1_3D/'
# circular domain (2D)
# meshes = ['circle1.msh', 'circle1_5.msh', 'circle2.msh', 'circle3.msh', 'circle4.msh', 'circle5.msh', 'circle6.msh',
#          'circle7.msh', 'circle8.msh', 'circle9.msh', 'circle9_5.msh', 'circle10.msh']

# elliptic domain (2D)
#meshes = ['elipse1.msh', 'elipse1_2.msh', 'elipse1_5.msh', 'elipse2.msh', 'elipse3.msh', 'elipse4.msh', 'elipse5.msh', 'elipse6.msh', 'elipse7.msh',
#          'elipse8.msh', 'elipse9.msh', 'elipse10.msh', 'elipse11.msh', 'elipse12.msh', 'elipse13.msh'
#          , 'elipse14.msh']

# cubic domain (3D)
meshes = ['cube1.msh', 'cube1_5.msh', 'cube2.msh', 'cube3.msh', 'cube4.msh', 'cube5.msh', 'cube6.msh', 'cube7.msh',
          'cube8.msh', 'cube9.msh', 'cube10.msh', 'cube11.msh']


def main(sigmad, mesh_file):

    "-------------- BUILDING AND READING MESH  --------------"
    Th = m.Mesh()
    Th.import_from_gmsh(mesh_file)

    if Th.dim == 2:
        Th2 = copy(Th)
        idx = np.argwhere(Th2.triangle_label == 600).ravel()
        Th2.triangle = Th2.triangle[idx]
        Th2.n_triangles = idx.size
        Th2.triangle_label = Th2.triangle_label[idx]

        "-------------- CONDUCTIVITY TENSORS  --------------"
        Sd = np.array([[sigmad, 0.0], [0.0, sigmad]])
        Si = np.array([[1.741, 0.0], [0.0, 0.1934]])
        Se = np.array([[3.904, 0.0], [0.0, 1.970]])

        dict_sigma_intra = {700: Sd, 600: Si}  # dictionaries containing conductivities and their tags
        dict_sigma_extra = {700: Sd, 600: Se}

        def sigma(y):
            labels = Th.triangle_label
            if labels[y] == 700:
                return Sd
            elif labels[y] == 600:
                return Se

        "-------------- ASSEMBLY AND FACTORIZATION  --------------"
        A = p1.assemblyStiffP1base(Th, dict_sigma_extra)  # assembly of stiffness matrix in sparse CSC format
        b1 = p1.assemblySecondMember(Th, Se - Sd, 1)
        b2 = p1.assemblySecondMember(Th, Se - Sd, 2)

        B = p1.assemblyStiffP1base(Th2, dict_sigma_intra)
        b3 = p1.assemblySecondMember(Th2, Si, 1)
        b4 = p1.assemblySecondMember(Th2, Si, 2)

        solve_A = factorized(A)  # LU Factorization of A
        x1 = solve_A(b1)
        x2 = solve_A(b2)
        dict_w = {0: x1, 1: x2}

        solve_B = factorized(B)  # LU Factorization of B
        x3 = solve_B(b3)
        x4 = solve_B(b4)
        dict_wi = {0: x3, 1: x4}

        "-------------- COMPUTATION OF THE NEW CONDUCTIVITY TENSORS  --------------"

        def homog_e(Th):
            d = Th.dim
            dof = Th.dof
            vols = Th.computeEltVolumes()
            volume_tot = 0
            elt = Th.triangle
            sigstar_e = np.zeros((d, d))

            for i in np.arange(Th.n_triangles):
                volume_tot += vols[i]

            for k in np.arange(d):
                for j in np.arange(d):
                    for i in np.arange(Th.n_triangles):
                        p = Th.node[elt[i]]
                        vect_dof = dof[elt[i]]
                        G = p1.gradientLocal(p)

                        sigstar_e[k, j] += vols[i] * sigma(i)[k, j] \
                                            + vols[i] * sigma(i)[k, 0] * np.dot(dict_w[j][vect_dof], np.dot(G.T, np.eye(2)[0]))\
                                            + vols[i] * sigma(i)[k, 1] * np.dot(dict_w[j][vect_dof], np.dot(G.T, np.eye(2)[1]))
            return sigstar_e, volume_tot

        def homog_i(Th2):
            d = Th2.dim
            dof = Th2.dof
            vols = Th2.computeEltVolumes()
            volume_b = 0
            elt = Th2.triangle
            sigstar_i = np.zeros((d, d))

            for i in np.arange(Th2.n_triangles):
                volume_b += vols[i]

            for k in np.arange(d):
                for j in np.arange(d):
                    for i in np.arange(Th2.n_triangles):
                        p = Th2.node[elt[i]]
                        vect_dof = dof[elt[i]]
                        G = p1.gradientLocal(p)

                        sigstar_i[k, j] += vols[i] * Si[k, j] \
                                            + vols[i] * Si[k, 0] * np.dot(dict_wi[j][vect_dof], np.dot(G.T, np.eye(2)[0]))\
                                            + vols[i] * Si[k, 1] * np.dot(dict_wi[j][vect_dof], np.dot(G.T, np.eye(2)[1]))
            return sigstar_i, volume_b

        sigstar_e, volume_tot = homog_e(Th)
        sigstar_i, volume_b = homog_i(Th2)

        return volume_tot - volume_b, sigstar_e[0, 0], sigstar_e[1, 1], sigstar_i[0, 0], sigstar_i[1, 1], \
               sigstar_e[0, 0]/sigstar_e[1, 1], sigstar_i[0, 0]/sigstar_i[1, 1]

    elif Th.dim == 3:
        Th2 = copy(Th)
        idx = np.argwhere(Th2.tetra_label == 600).ravel()
        Th2.tetra = Th2.tetra[idx]
        Th2.n_tetras = idx.size
        Th2.tetra_label = Th2.tetra_label[idx]

        "-------------- CONDUCTIVITY TENSORS  --------------"
        Sd = np.array([[sigmad, 0.0, 0.0], [0.0, sigmad, 0.0], [0.0, 0.0, sigmad]])
        Si = np.array([[1.741, 0.0, 0.0], [0.0, 0.1934, 0.0], [0.0, 0.0, 0.1934]])
        Se = np.array([[3.904, 0.0, 0.0], [0.0, 1.970, 0.0], [0.0, 0.0, 1.970]])

        dict_sigma_intra = {700: Sd, 600: Si}  # dictionaries containing conductivities and their tags
        dict_sigma_extra = {700: Sd, 600: Se}

        def sigma(y):
            labels = Th.tetra_label
            if labels[y] == 700:
                return Sd
            elif labels[y] == 600:
                return Se

        "-------------- ASSEMBLY AND FACTORIZATION  --------------"
        A = p1.assemblyStiffP1base(Th, dict_sigma_extra)  # assembly of stiffness matrix in sparse CSC format
        b1 = p1.assemblySecondMember(Th, Se - Sd, 1)
        b2 = p1.assemblySecondMember(Th, Se - Sd, 2)
        b5 = p1.assemblySecondMember(Th, Se - Sd, 3)

        B = p1.assemblyStiffP1base(Th2, dict_sigma_intra)
        b3 = p1.assemblySecondMember(Th2, Si, 1)
        b4 = p1.assemblySecondMember(Th2, Si, 2)
        b6 = p1.assemblySecondMember(Th2, Si, 3)

        solve_A = factorized(A)  # LU Factorization of A
        x1 = solve_A(b1)
        x2 = solve_A(b2)
        x5 = solve_A(b5)
        dict_w = {0: x1, 1: x2, 2: x5}

        solve_B = factorized(B)  # LU Factorization of B
        x3 = solve_B(b3)
        x4 = solve_B(b4)
        x6 = solve_B(b6)
        dict_wi = {0: x3, 1: x4, 2: x6}

        "-------------- COMPUTATION OF THE NEW CONDUCTIVITY TENSORS  --------------"

        def homog_e(Th):
            d = Th.dim
            dof = Th.dof
            vols = Th.computeEltVolumes()
            volume_tot = 0
            elt = Th.tetra
            sigstar_e = np.zeros((d, d))

            for i in np.arange(Th.n_tetras):
                volume_tot += vols[i]

            for k in np.arange(d):
                for j in np.arange(d):
                    for i in np.arange(Th.n_tetras):
                        p = Th.node[elt[i]]
                        vect_dof = dof[elt[i]]
                        G = p1.gradientLocal(p)

                        sigstar_e[k, j] += vols[i] * sigma(i)[k, j] \
                                           + vols[i] * sigma(i)[k, 0] * np.dot(dict_w[j][vect_dof],
                                                                               np.dot(G.T, np.eye(3)[0])) \
                                           + vols[i] * sigma(i)[k, 1] * np.dot(dict_w[j][vect_dof],
                                                                               np.dot(G.T, np.eye(3)[1])) \
                                           + vols[i] * sigma(i)[k, 2] * np.dot(dict_w[j][vect_dof],
                                                                               np.dot(G.T, np.eye(3)[2]))
            return sigstar_e, volume_tot

        def homog_i(Th2):
            d = Th2.dim
            dof = Th2.dof
            vols = Th2.computeEltVolumes()
            volume_b = 0
            elt = Th2.tetra
            sigstar_i = np.zeros((d, d))

            for i in np.arange(Th2.n_tetras):
                volume_b += vols[i]

            for k in np.arange(d):
                for j in np.arange(d):
                    for i in np.arange(Th2.n_tetras):
                        p = Th2.node[elt[i]]
                        vect_dof = dof[elt[i]]
                        G = p1.gradientLocal(p)

                        sigstar_i[k, j] += vols[i] * Si[k, j] \
                                           + vols[i] * Si[k, 0] * np.dot(dict_wi[j][vect_dof],
                                                                         np.dot(G.T, np.eye(3)[0])) \
                                           + vols[i] * Si[k, 1] * np.dot(dict_wi[j][vect_dof],
                                                                         np.dot(G.T, np.eye(3)[1])) \
                                           + vols[i] * Si[k, 2] * np.dot(dict_wi[j][vect_dof],
                                                                         np.dot(G.T, np.eye(3)[2]))
            return sigstar_i, volume_b

        sigstar_e, volume_tot = homog_e(Th)
        sigstar_i, volume_b = homog_i(Th2)

        return volume_tot - volume_b, sigstar_e[0, 0], sigstar_e[1, 1], sigstar_e[2, 2], sigstar_i[0, 0], \
        sigstar_i[1, 1], sigstar_i[2, 2], sigstar_e[0, 0] / sigstar_e[1, 1], sigstar_i[0, 0] / sigstar_i[1, 1], \
        sigstar_e[0, 0] / sigstar_e[2, 2], sigstar_i[0, 0] / sigstar_i[2, 2]


"-------------- COMPUTATION FOR DIFFERENT MESHES  --------------"
# 2D case
#with open("data_main.dat", "w") as f:
    #for i in meshes:
        #valeurs = "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n" % (main(0.2, path + i))
        #f.write(valeurs)
# 3D case
with open("data_main.dat", "w") as f:
    for i in meshes:
        valeurs = "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n" % (main(3.0, path + i))
        f.write(valeurs)
