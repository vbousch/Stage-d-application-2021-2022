import mesh as m
import numpy as np
from scipy import sparse
from scipy import linalg


# we are here building the stiffness and mass matrices with two methods
# the usual method laying on a loop on the number of elements
# and another 'vectorized' method more efficient in terms of computation time

def assemblyMassP1LumpedOptV(Th):
    """Assembly of the Mass Matrix with lumping by :math:`P_1`-Lagrange
    finite elements using a vectorized version.

    With mass lumping the matrix is diagonal.

    param Th: a Mesh class (see mesh.py)
    returns: A *Scipy* CSC sparse matrix.

    """
    if Th.dim == 2:
        n_elts = Th.n_triangles
        elt = Th.triangle  # in 2D, the elements are the triangles
    if Th.dim == 3:
        n_elts = Th.n_tetras
        elt = Th.tetra  # in 3D, the elements are the tetraedras

    # the mass matrix has coefficients E_ij = \int_T phi_j phi_i dx, with
    # phi_k affine functions such that phi_k(x_l) = delta_{kl}. Using the
    # simplest quadrature rule \int_T f(x)dx ~ |T|/(d+1)*\sum_{k=1..d+1}
    # f(x_k), we finally have thge lumped coefficients E_ij = |T|/(d+1) *
    # \delta_{ij}, where \delta_{ij} is the Kronecker symbol.

    vols = Th.computeEltVolumes() / (Th.dim + 1)
    dof = Th.dof
    # We have to build the plain arrays I,J, and E used to construct the
    # CSC matrix.
    E = np.zeros((n_elts, Th.dim + 1), dtype=np.float64)  # All the diagonal terms
    I = np.zeros((n_elts, Th.dim + 1), dtype=np.int32)  # and their indices
    for iLoc in np.arange(Th.dim + 1):
        # Coeff ii of the elementary lumped mass matrix over all the
        # elements: E_ii = vol/(d+1) are the only nonzero entries
        E[:, iLoc] = vols[:]
        I[:, iLoc] = dof[elt[:, iLoc]]

    M = sparse.csc_matrix((E.ravel(), (I.ravel(), I.ravel())),
                          shape=(Th.n_dof, Th.n_dof))
    return M


def gradientLocal(simplex):
    """Compute the gradient of the P1 basis functions in the d-simplicial
    element which coordinates are given in the (d+1)xd array.

    simplex is an array of coordinates of size (d+1)xd

    """
    # On récupère la dimension spatiale = le nombre de coordonnées
    d = simplex.shape[1]
    # cas 2d pour le moment

    # on remplit la matrice G de l'élement de reférence avec les gradients des fonctions de base
    # cas 2D : matrice 2lignesx3col

    G_ref = np.zeros((d, d + 1))
    G_ref[:, 0] = -1.
    G_ref[:, 1:] = np.eye(d, d)

    # on a par calcul que P^T * G = G_ref

    G = np.zeros_like(G_ref)
    # P^T = simplex_k - simplex_0 pour 1<=k<=d
    Pt = simplex[1:, :] - np.tile(simplex[0, :], (d, 1))

    # Et il reste à résoudre le système linéaire
    G = np.linalg.solve(Pt, G_ref)

    return G


def elemStiffP1(simplex, vol, sigma):  # a deplacer
    """Returns the elementary stiffness matrix in the simplex.
    simplex is an array of coordinates of size (d+1) x d
    vol is its precomputed volume
    """

    G = gradientLocal(simplex)  # each column is a gradient, G is d x (d+1)

    E = vol * np.matmul(np.matmul(sigma, G).transpose(), G)
    return E


def assemblyStiffP1base(Th, dict_sigma):  # a deplacer
    """
    param Th: a Mesh class
    returns: A *Scipy* CSC sparse matrix.

    """
    if Th.dim == 2:
        n_elts = Th.n_triangles
        elt = Th.triangle  # in 2D, the elements are the triangles
        labels = Th.triangle_label
    if Th.dim == 3:
        n_elts = Th.n_tetras
        elt = Th.tetra  # in 3D, the elements are the tetraedras
        labels = Th.tetra_label

    M = sparse.lil_matrix((Th.n_dof, Th.n_dof))

    vols = Th.computeEltVolumes()
    dof = Th.dof

    for k in np.arange(n_elts):
        sigma = dict_sigma[labels[k]]
        E = elemStiffP1(Th.node[elt[k]], vols[k], sigma)

        for iLoc in np.arange(Th.dim + 1):
            i = dof[elt[k, iLoc]]

            for jLoc in np.arange(Th.dim + 1):
                j = dof[elt[k, jLoc]]

                M[i, j] += E[iLoc, jLoc]

    return M.tocsc()


def gradientVec(Th):
    """Compute the gradients of the P1 basis functions in all the elements
    of the mesh. Returns an array of size n_elts x (d+1) x d

    """
    if Th.dim == 2:
        n_elts = Th.n_triangles
        elt = Th.triangle  # in 2D, the elements are the triangles
    if Th.dim == 3:
        n_elts = Th.n_tetras
        elt = Th.tetra  # in 3D, the elements are the tetraedras

    # L'élément de référence est le simplexe T_ref = {z \in (0,1)^d tel
    # que z_0+...+z_{d-1} < 1}. L'élément simplex est obtenu par le
    # changement de coordonnées x = x_0 + P*z où x_0 est le point 0 de
    # simplex, et P = [x_1-x_0, ... x_d-x_0] est une matrice dxd.
    #
    # Dans l'elt de référence, les fonctions de forme sont la fonction
    # Z_0 = 1-z_0-...-z_{d-1} dont le gradient est le vecteur [-1...-1]
    # dans R^d, et les fonctions coordonnées Z_{k+1} = z_k pour k=0 à d-1
    # dont les gradients dont les vecteurs e_k de la base
    # canoninque. Leur gradient est donc donné (par colonnes) par la
    # matrice
    G_ref = np.zeros((Th.dim, Th.dim + 1))
    G_ref[:, 0] = -1.
    G_ref[:, 1:] = np.eye(Th.dim, Th.dim)  # G_ref is the matrix of
    # gradients in the ref element,
    # of size (d+1)xd

    # Les fonctions de forme P1 sont les fonction phi_k pour k=0 à d
    # telles que phi_k(x) = z_k(z). Le gradient de phi_k est donc donné
    # par la formule de dérivation des fonctions composées:
    # P^T * d_x(phi_k) = d_z(z_k)
    # où d_z(z_k) est la matrice G_ref

    # We have to solve the n_elts linear systems P_k^T G = G_ref, where
    # P_k = [x1-x0,... xd-x0] is a dxd matrix. Python (Numpy, Scipy) do
    # not have such functions. Anyway, the function numpy.linalg.inv is
    # able to inverse several linear systems at once.
    #
    # First, let's build the matrices P^T as a global array of size n_elts x dxd.
    Pt = np.zeros((n_elts, Th.dim, Th.dim))
    for k in np.arange(Th.dim):
        Pt[:, k, :] = Th.node[elt[:, 1 + k]] - Th.node[elt[:, 0]]

    # On calcule les inverse des matrices Pt
    PtInv = np.linalg.inv(Pt)
    G = np.matmul(PtInv, G_ref)  # Multiplies each of the Pt^{-1} with G_ref

    return G


def assemblyStiffP1OptV(Th, dict_sigma):
    """Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite
    elements using a vectorized version.

    param Th: a Mesh class (see mesh.py)
    returns: A *Scipy* CSC sparse matrix.

    """
    if Th.dim == 2:
        n_elts = Th.n_triangles
        elt = Th.triangle  # in 2D, the elements are the triangles
    if Th.dim == 3:
        n_elts = Th.n_tetras
        elt = Th.tetra  # in 3D, the elements are the tetraedras

    dof = Th.dofs
    G = gradientVec(Th)  # All the gradients in the each of the mesh elts,
    # this is a 3D array of size n_elts x d x (d+1)
    vols = Th.computeEltVolumes()
    sigmas = np.array([dict_sigma[l] for l in Th.triangle_label])

    # We have to compute the G_k^T*G_k -> array n_elts x (d+1) x
    # (d+1). We loop over the (i,j) in the last 2 dimensions and
    # vectorize along the first one
    E = np.zeros((n_elts, Th.dim + 1, Th.dim + 1), dtype=np.float64)  # All the elementary matrices
    I = np.zeros((n_elts, Th.dim + 1, Th.dim + 1), dtype=np.int32)  # and their indices
    J = np.zeros((n_elts, Th.dim + 1, Th.dim + 1), dtype=np.int32)
    for iLoc in np.arange(Th.dim + 1):
        for jLoc in np.arange(Th.dim + 1):
            # Coeff ij of the elementary stiffness matrix over all the
            # elements
            # E[:,iLoc,jLoc] = vols * np.sum( G[:,:,jLoc] * G[:,:,iLoc], axis=1)
            E[:, iLoc, jLoc] = vols * np.multiply(sigmas, np.sum(G[:, :, jLoc] * G[:, :, iLoc], axis=1))
            I[:, iLoc, jLoc] = dof[:, iLoc]
            J[:, iLoc, jLoc] = dof[:, jLoc]

    M = sparse.csc_matrix((E.ravel(), (I.ravel(), J.ravel())),
                          shape=(Th.n_nodes, Th.n_nodes))
    return M


def assemblySecondMember(Th, sigma, j):
    """Assembly of the second member for the cell problem
       param : Th a Mesh class (see mesh.py), sigma : an array (see main_homogeneisation.py),
               j : an integer representing the local direction (1,2,3)                """

    n_dofs = Th.n_dof
    dof = Th.dof

    if Th.dim == 2:
        n_edges_inter = Th.n_edges_inter
        edge_inter = Th.edge_inter
        e_j = np.zeros(2)
        e_j[j - 1] = 1

        b = np.zeros(n_dofs)

        for k in np.arange(n_edges_inter):

            # for each edge, its normal vector is calculated
            pt_a, pt_b = Th.node[edge_inter[k][[0, 1]]]  # coordinates linked to the ends of the edge are extracted
            tang = pt_b - pt_a
            n = np.array([tang[1], -tang[0]])
            lenght = np.linalg.norm(n)
            n /= lenght
            v = np.array([0.5, 0.5]) - pt_a
            if np.dot(n, v) < 0.:
                n *= -1

            val = np.dot(np.dot(sigma, e_j), n)
            edge_contrib = 0.5 * lenght * val * np.ones(2)

            for iLoc in np.arange(Th.dim):
                i = dof[edge_inter[k, iLoc]]
                b[i] -= edge_contrib[iLoc]

    if Th.dim == 3:
        n_tri_inter = Th.n_tri_inter
        tri_inter = Th.tri_inter
        tri_label = Th.triangle_label
        e_j = np.zeros(3)
        e_j[j - 1] = 1

        b = np.zeros(n_dofs)

        for k in np.arange(n_tri_inter):
            # for each triangle belonging to the interface, its normal vector is calculated
            pt_a, pt_b, pt_c = Th.node[tri_inter[k][[0, 1, 2]]]  # coordinates linked to the 3 ends
                                                                 # of the triangle are extracted
            temp1 = pt_a - pt_b
            temp2 = pt_a - pt_c
            n = np.cross(temp1, temp2)
            lenght = np.linalg.norm(n)
            n /= lenght
            # we check the orientation of the normal vector (cubic case)
            if tri_label[k] == 1000:
                if np.dot(n, np.array([-1.0, 0.0, 0.0])) < 0.:
                    n *= -1
            elif tri_label[k] == 2000:
                if np.dot(n, np.array([1.0, 0.0, 0.0])) < 0.:
                    n *= -1
            elif tri_label[k] == 3000:
                if np.dot(n, np.array([0.0, 1.0, 0.0])) < 0.:
                    n *= -1
            elif tri_label[k] == 4000:
                if np.dot(n, np.array([0.0, -1.0, 0.0])) < 0.:
                    n *= -1
            elif tri_label[k] == 5000:
                if np.dot(n, np.array([0.0, 0.0, 1.0])) < 0.:
                    n *= -1
            elif tri_label[k] == 6000:
                if np.dot(n, np.array([0.0, 0.0, -1.0])) < 0.:
                    n *= -1
            val = np.dot(np.dot(sigma, e_j), n)
            tri_contrib = 1/3 * lenght * val * np.ones(3)

            for iLoc in np.arange(Th.dim):
                i = dof[tri_inter[k, iLoc]]
                b[i] -= tri_contrib[iLoc]

    return b
