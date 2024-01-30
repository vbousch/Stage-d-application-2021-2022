"Premier jet sur un code important les différentes informations liées au maillage d'un domaine, on se limite ici à un maillage 2D utilisant le format medit"

"Ce code se base sur les travaux de Yves Coudière, Jérôme Fehrenbach et Lisl Weynans"
# -*- coding: utf-8 -*-
"""Created on Thu Jul  9 11:58:36 2020

@author: Yves Coudière
@author: Jérôme Fehrenbach
@author: Lisl Weynans

Librairie d'utilités pour les maillages.

fonctionnalités :

implémentation d'une structure de donnée (class) Mesh et des méthodes
listées ci-dessous

- import_from_mesh :: lecture du format mesh (Medit)
- export_to_vtk :: écriture sous format vtk (Legacy, v3.0)
- computeMaxLengthEdges :: longueur de la plus grande arète
- lecture d'un maillage medit .mesh 2D (plan) ou 3D (volume)
- lecture d'un maillage gmsh en 2D (surface maillée) et 3D
- lecture d'un maillage vtk
- lecture d'un maillage en {node,ele}

Le maillage contient les structures utilisées dans Medit et MMG
(https://www.ljll.math.upmc.fr/frey/logiciels/Docmedit.dir/index.html)

Si un maillage de dimension 3 ne contient pas de tetraèdres, alors c'est
une surface. Si toutes les coordonnées z sont nulles alors c'est un
maillage 2D, sinon, c'est une surface 2D plongée dans R^3. Le paramètre
dim fait référence à l'espace complet, ie le omnbre de coordonnées des
sommets -- dim=3 dans le cas d'une surface plongée dans R^3.

"""
import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.text as txt
import scipy.io as scio


class Mesh:
    """Data structure for a mesh like they are used in MMG or Medit, see
    this page: described here
    https://www.ljll.math.upmc.fr/frey/logiciels/Docmedit.dir/index.html

    Dimension = {2,3}
    Vertices = nodes with dim coordinates and an integer label
    Triangles = sets of 3 nodes and a label
    Quadrilaterlas = sets of 4 nodes and a label -- not used
    Tetrahedra = sets of 4 nodes and a label
    Hexaedra = sets of 8 nodes and a label -- not used
    Edges = sets of 2 nodes and a label
    Singularities (corners, ridges, etc) -- not used
    
    """

    def __init__(self,
                 dim=0,
                 is_surface=False,
                 n_nodes=0,
                 node=np.empty((0, 0), dtype=np.float64),
                 node_label=np.empty((0), dtype=np.int32),
                 n_triangles=0,
                 triangle=np.empty((0, 0), dtype=np.int32),
                 triangle_label=np.empty((0), dtype=np.int32),
                 n_tetras=0,
                 tetra=np.empty((0, 0), dtype=np.int32),
                 tetra_label=np.empty((0), dtype=np.int32),
                 n_edges=0,
                 n_edges_inter=0,
                 n_tri_inter=0,
                 edge=np.empty((0, 0), dtype=np.int32),
                 edge_inter=np.empty((0, 0), dtype=np.int32),
                 tri_inter=np.empty((0, 0, 0), dtype=np.int32),
                 edge_label=np.empty((0), dtype=np.int32),
                 n_per_nodes=0,
                 per_relations=np.empty((0, 0), dtype=np.int32),
                 ):
        """Default constructor
        
        """  # Shape
        self.dim = dim
        self.is_surface = False  # True if only triangles
        # and the nodes need 3
        # coordinates
        self.n_nodes = n_nodes
        self.node = node  # n_nodes x dim
        self.node_label = node_label  # n_nodes
        self.dof, self.n_dof = dof(per_relations, n_nodes)
        self.n_triangles = n_triangles
        self.triangle = triangle  # n_triangles x 3
        self.triangle_label = triangle_label  # n_triangles
        self.n_tri_inter = n_tri_inter
        self.tri_inter = tri_inter

        self.n_tetras = n_tetras
        self.tetra = tetra  # n_tetras x 4
        self.tetra_label = tetra_label  # n_tetras

        self.n_edges = n_edges
        self.n_edges_inter = n_edges_inter
        self.edge = edge  # n_edges x 2
        self.edge_inter = edge_inter
        self.edge_label = edge_label  # n_edges

        self.n_interfaces = 0
        self.interface = np.empty((0, 0), dtype=np.int32)
        self.neighbour = np.empty((0, 0), dtype=np.int32)
        self.n_per_nodes = n_per_nodes
        self.per_relations = per_relations  # n_per_nodes  x 2
        return

    def __repr__(self):
        str = ""
        if self.dim > 0:
            if self.dim == 2:
                str += "Mesh of a 2D domain\n"
            elif self.dim == 3:
                if self.is_surface:
                    str = "Mesh of a 2D surface in R^3\n"
                else:
                    str = "Mesh of a 3D volume\n"
            str += "Nombre de sommets   : {}".format(self.n_nodes)
            if self.n_nodes > 0:
                str += " -- labels {} à {}\n".format(
                    np.min(self.node_label), np.max(self.node_label))
            else:
                str += "\n"
            if self.n_tetras > 0:
                str += "Nombre de tetras    : {}".format(self.n_tetras)
                str += " -- labels {} à {}\n".format(
                    np.min(self.tetra_label), np.max(self.tetra_label))
                for label in np.arange(np.min(self.tetra_label), np.max(self.tetra_label) + 1):
                    n = np.count_nonzero(self.tetra_label == label)
                    if n > 0:
                        str += "    avec label {} : {} tetras\n".format(label, n)
            if self.n_triangles > 0:
                str += "Nombre de triangles : {}".format(self.n_triangles)
                str += " -- labels {} à {}\n".format(
                    np.min(self.triangle_label), np.max(self.triangle_label))
            if self.n_edges > 0:
                str += "Nombre d'arètes     : {}".format(self.n_edges)
                str += " -- labels {} à {}\n".format(
                    np.min(self.edge_label), np.max(self.edge_label))
            if self.n_interfaces > 0:
                str += "Nombre d'interfaces : {}".format(self.n_interfaces)
                n = np.count_nonzero(self.interface[:, -1] + 1)
                str += " -- dont {} au bord\n".format(self.n_interfaces - n)
            D = self.computeDiameter()
            h = self.computeMaxLengthEdges()
            str += "Diamètre du domaine : {:4e}\n".format(D)
            str += "Longueur de la plus grande arète {:4e} soit {:3.1f}% du diamètre".format(h, h / D * 100.)
        return str

    def import_from_gmsh(self, mshfile):
        """Read a Gmsh .msh file.
        
        """
        # We will read only edges, triangles and tetras
        elm_types = [1, 2, 4]  # from gmsh element types: MSH_LIN_2,MSH_TRI_3,MSH_TET_4
        n_vert = [2, 3, 0, 4]
        # We will read edges, triangles and tets in lists before casting
        # them to numpy arrays, because the size of the arrays is not
        # known initially
        edg, tri, tet, tri_labels, tetra_labels, edg_inter, tri_inter = [], [], [], [], [], [], []

        print("Reading {}".format(mshfile))

        with open(mshfile) as f:
            for line in f:
                words = line.split()
                if line.startswith('$MeshFormat'):
                    words = f.readline().split()  # read next line that contains info
                    if len(words) == 3:
                        x, file_type, data_size = (np.float64(words[0]),
                                                   np.int32(words[1]),
                                                   np.int32(words[2]))
                        version = int(x)  # get major version number (1,2,4)
                        assert version in [1, 2, 4], "unkown version number {}".format(x)
                        assert file_type == 0, "this is a Binary file, not readable yet"
                        print('{} format, version {}'.format(
                            ('ASCII', 'Binary')[file_type], version))

                if line.startswith('$Nodes') or line.startswith('$NOD'):
                    words = f.readline().split()  # read next line that contains info
                    if version in [1, 2]:
                        self.n_nodes = np.int32(words[0])
                        self.node = np.zeros((self.n_nodes, 3), dtype=np.float64)
                        # in versions 1 and 2, one node on each row, with
                        # its index at the begining of the row.
                        for i in np.arange(self.n_nodes):
                            line = f.readline()
                            numbers = np.fromstring(line, sep=' ', dtype=np.float64, count=4)
                            ind = np.int32(numbers[0])  # index, starting from 1
                            self.node[ind - 1] = numbers[1:]
                        # TODO find a smart way to do that w/o an explicit loop
                        # The nodes may be numbered in a sparse manner,
                        # which will cause an error here.
                    elif version == 4:
                        numEntityBlocks = np.int32(words[0])
                        self.n_nodes = np.int32(words[1])
                        minNodeTag = np.int32(words[2])
                        maxNodeTag = np.int32(words[3])
                        nNodeTags = 1 + maxNodeTag - minNodeTag
                        # Check that mesh numbering is not sparse
                        assert nNodeTags == self.n_nodes, \
                            "The nodes seem to have sparse indexing"
                        self.node = np.zeros((self.n_nodes, 3), dtype=np.float64)
                        ind = np.zeros(self.n_nodes)
                        for i in np.arange(self.n_nodes):
                            line = f.readline()
                            ind[i] = np.fromstring(line, sep=' ', dtype=np.int32, count=1)
                            print(ind[i])
                        for i in np.arange(self.n_nodes):
                            line = f.readline()
                            self.node[ind[i] - minNodeTag] = np.fromstring(
                                line, sep=' ', dtype=np.float64, count=3)
                        # TODO find a smart way to do that w/o a explicit loops
                    self.node_label = np.zeros(self.n_nodes, dtype=np.int32)

                if line.startswith('$Elements') or line.startswith('$ELM'):
                    if version in [1, 2]:
                        words = f.readline().split()  # read next line that contains info
                        n_elts = np.int32(words[0])  # Total number of elements
                        # v2: elm-number elm-type number-of-tags < tag > … node-number-list
                        # v1: elm-number elm-type reg-phys reg-elem number-of-nodes node-number-list
                        for i in np.arange(n_elts):
                            line = f.readline()
                            numbers = np.fromstring(line, sep=' ', dtype=np.int32, count=-1)
                            n_tags = numbers[2]
                            # the list of vertices is always at the end
                            if numbers[1] == 1:
                                self.n_edges += 1
                                edg.append(numbers[-2:])
                                if numbers[3] == 100:  # 100 correspond to the tag of the interface
                                    self.n_edges_inter += 1
                                    edg_inter.append(numbers[-2:])
                            if numbers[1] == 2:
                                self.n_triangles += 1
                                tri.append(numbers[-3:])
                                tri_labels.append(numbers[-5])
                                if numbers[3] == 1000 or numbers[3] == 2000 or numbers[3] == 3000 or numbers[3] == 4000\
                                        or numbers[3] == 5000 or numbers[3] == 6000:
                                    # correspond to the tags of the sides of the cubic domain
                                    self.n_tri_inter += 1
                                    tri_inter.append(numbers[-3:])
                            if numbers[1] == 4:
                                self.n_tetras += 1
                                tet.append(numbers[-4:])
                                tetra_labels.append(numbers[-6])
                        # finalize the edges, triangles and tetras
                        if self.n_edges > 0:
                            self.edge = np.ascontiguousarray(edg) - 1
                            self.edge_inter = np.ascontiguousarray(edg_inter) - 1
                            self.edge_label = np.zeros(self.n_edges, dtype=np.int32)
                        if self.n_triangles > 0:
                            self.triangle = np.ascontiguousarray(tri) - 1
                            # self.triangle_label = np.zeros(self.n_triangles, dtype=np.int32)
                            self.triangle_label = np.ascontiguousarray(tri_labels)
                            self.tri_inter = np.ascontiguousarray(tri_inter) - 1
                        if self.n_tetras > 0:
                            self.tetra = np.ascontiguousarray(tet) - 1
                            # self.tetra_label = np.zeros(self.n_tetras, dtype=np.int32)
                            self.tetra_label = np.ascontiguousarray(tetra_labels)
                    elif version == 4:
                        # 1st line: nEntityBlocks nElts minTag maxTag
                        # repeat as many times as there are entity blocks
                        # entityDim entityTag eltType nEltsInBlock
                        # elementTag nodeTag
                        line = f.readline()
                        n_blocks, n_elts, tag_min, tag_max = \
                            np.fromstring(line, sep=' ', dtype=np.int32, count=-1)
                        for i_block in np.arange(n_blocks):
                            line = f.readline()
                            dim, tag, e_type, n = \
                                np.fromstring(line, sep=' ', dtype=np.int32, count=-1)
                            if e_type == 1:
                                self.n_edges = n
                                self.edge = np.zeros((n, 2))
                                for i in np.arange(n):
                                    line = f.readline()
                                    numbers = np.fromstring(line, sep=' ',
                                                            dtype=np.float64, count=3)
                                    ind = np.int32(numbers[0])  # index, starting from 1
                                    self.edge[ind - 1] = numbers[1:]
                            if e_type == 2:
                                self.n_triangles = n
                                self.triangle = np.zeros((n, 3))
                                for i in np.arange(n):
                                    line = f.readline()
                                    numbers = np.fromstring(line, sep=' ',
                                                            dtype=np.float64, count=4)
                                    ind = np.int32(numbers[0])  # index, starting from 1
                                    self.triangle[ind - 1] = numbers[1:]
                            if e_type == 4:
                                self.n_tetras = n
                                self.tetra = np.zeros((n, 4))
                                for i in np.arange(n):
                                    line = f.readline()
                                    numbers = np.fromstring(line, sep=' ',
                                                            dtype=np.float64, count=3)
                                    ind = np.int32(numbers[0])  # index, starting from 1
                                    self.tetra[ind - 1] = numbers[1:]

                if line.startswith('$Periodic'):         #TODO
                    words = f.readline().split()
                    n_per_relations = np.int32(words[0])

                    temp1 = np.empty((0, 2), int)
                    temp2 = 0

                    if not tet:         # cas 2D
                        for k in np.arange(n_per_relations):
                            line = f.readline()
                            words = f.readline().split()
                            n = np.int32(words[0])  # number of periodic nodes on the entity

                            per_relations = np.zeros((n, 2))

                            for i in np.arange(n):
                                words = f.readline().split()
                                per_relations[i][0] = np.int32(words[0])
                                per_relations[i][1] = np.int32(words[1])

                            temp1 = np.append(temp1, per_relations)
                            temp2 += n
                            # self.n_per_nodes += n

                            # np.add(relation, per_relations, out=relation, casting="unsafe")
                        self.n_per_nodes = temp2
                        # -1 pour adapter la notation de gmsh à python (1->0)
                        self.per_relations = temp1.reshape((temp2, 2)) - 1
                    else:        # cas 3D
                        for k in np.arange(n_per_relations):
                            line = f.readline()
                            line = f.readline()
                            words = f.readline().split()
                            n = np.int32(words[0])  # number of periodic nodes on the entity

                            per_relations = np.zeros((n, 2))

                            for i in np.arange(n):
                                words = f.readline().split()
                                per_relations[i][0] = np.int32(words[0])
                                per_relations[i][1] = np.int32(words[1])

                            temp1 = np.append(temp1, per_relations)
                            temp2 += n
                            # self.n_per_nodes += n

                            # np.add(relation, per_relations, out=relation, casting="unsafe")
                        self.n_per_nodes = temp2
                        # -1 pour adapter la notation de gmsh à python (1->0)
                        self.per_relations = temp1.reshape((temp2, 2)) - 1

        # in gmsh meshes, the dimension is always 3
        self.dim = 3
        self.test_if_surface()
        self.per_relations = self.per_relations.astype(int)
        self.dof, self.n_dof = dof(self.per_relations, self.n_nodes)
        return

    def import_from_mesh(self, meshfile):
        """Read a .mesh file following the medit and mmg file format. Works for
        meshes in 2D and 3D.

        """
        print("Reading {}".format(meshfile))
        with open(meshfile) as f:
            for line_plain in f:

                line = line_plain.strip()  # Delete the blank chars at the beginning

                if line.startswith('#'):
                    continue  # skip the rest and go directly to the next line

                if line.startswith('Dimension'):
                    # the dimension may be on the same line or on the next
                    # non empty line...
                    self.dim = read_integer_number(f, line)

                elif line.startswith('Vertices'):
                    self.n_nodes = read_integer_number(f, line)
                    self.node, self.node_label = read_floats_and_label(f, self.n_nodes, self.dim)

                elif line.startswith('Edges'):
                    self.n_edges = read_integer_number(f, line)
                    self.edge, self.edge_label = read_ints_and_label(f, self.n_edges, 2)
                    self.edge += -1

                elif line.startswith('Triangles'):
                    self.n_triangles = read_integer_number(f, line)
                    self.triangle, self.triangle_label = read_ints_and_label(f, self.n_triangles, 3)
                    self.triangle += -1

                elif line.startswith('Tetrahedra'):
                    self.n_tetras = read_integer_number(f, line)
                    self.tetra, self.tetra_label = read_ints_and_label(f, self.n_tetras, 4)
                    self.tetra += -1
        self.test_if_surface()
        return

    def export_to_vtk(self, filename, *values):
        """Write the mesh data to a legacy VTK unstructured grid file, and
        optionally any number of numpy arrays that correspond to point
        data or cell data

        """
        f = open(filename, 'w')

        f.write('# vtk DataFile Version 3.0\n')
        f.write('# Export of mesh structure\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')

        # POINTS
        f.write('POINTS {} float\n'.format(self.n_nodes))
        # self.node.tofile(f,sep=" ",format="%g") # fast but unreadable by a human
        if self.dim == 3:
            for i in np.arange(self.n_nodes):
                self.node[i].tofile(f, sep=" ")
                f.write('\n')
        if self.dim == 2:
            x = np.zeros(3)
            for i in np.arange(self.n_nodes):
                x[0:2] = self.node[i]
                x.tofile(f, sep=" ")
                f.write('\n')

        # CELLS
        # Total number if integers to write
        n_cells = 5 * self.n_tetras + 4 * self.n_triangles + 3 * self.n_edges
        f.write('CELLS {} {}\n'.format(self.n_tetras + self.n_triangles + self.n_edges, n_cells))
        # Tetra
        for i in np.arange(self.n_tetras):
            f.write('{} '.format(4))
            self.tetra[i].tofile(f, sep=" ")
            f.write('\n')
        # Triangles
        for i in np.arange(self.n_triangles):
            f.write('{} '.format(3))
            self.triangle[i].tofile(f, sep=" ")
            f.write('\n')
        # Edges
        for i in np.arange(self.n_edges):
            f.write('{} '.format(2))
            self.edge[i].tofile(f, sep=" ")
            f.write('\n')

        # CELL TYPES
        f.write('CELL_TYPES {}\n'.format(self.n_tetras + self.n_triangles + self.n_edges))
        # Tetra -- VTK_TETRA = 10
        if self.n_tetras > 0:
            (10 * np.ones(self.n_tetras, dtype=np.int)).tofile(f, sep="\n")
            f.write('\n')
        # Triangles -- VTK_TRIANGLE = 5
        if self.n_triangles > 0:
            (5 * np.ones(self.n_triangles, dtype=np.int)).tofile(f, sep="\n")
            f.write('\n')
        # Edges -- VTK_LINE = 3
        if self.n_edges > 0:
            (3 * np.ones(self.n_edges, dtype=np.int)).tofile(f, sep="\n")
            f.write('\n')

        # CELL DATA
        f.write('CELL_DATA {}\n'.format(self.n_tetras + self.n_triangles + self.n_edges))
        f.write('SCALARS domaine int\n')
        f.write('LOOKUP_TABLE DEFAULT\n')
        # Tetra
        if self.n_tetras > 0:
            self.tetra_label.tofile(f, sep="\n", format="%d")
            f.write('\n')
        # Triangles
        if self.n_triangles > 0:
            self.triangle_label.tofile(f, sep="\n", format="%d")
            f.write('\n')
        # Edges
        if self.n_edges > 0:
            self.edge_label.tofile(f, sep="\n", format="%d")
            f.write('\n')
        # Go through all *values and write cell_data if any
        for val in values:
            v = val[0]
            name = val[1].replace(" ", "_")  # white spaces not recognized here
            if v.shape[0] == self.n_tetras + self.n_triangles + self.n_edges:
                f.write('SCALARS {} float\n'.format(name))
                f.write('LOOKUP_TABLE DEFAULT\n')
                v.tofile(f, sep="\n", format="%f")

        # POINT DATA
        f.write('POINT_DATA {}\n'.format(self.n_nodes))
        # Go through all *values and write cell_data if any
        for val in values:
            v = val[0]
            name = val[1].replace(" ", "_")  # white spaces not recognized here
            if v.shape[0] == self.n_nodes:
                f.write('SCALARS {} float\n'.format(name))
                f.write('LOOKUP_TABLE DEFAULT\n')
                v.tofile(f, sep="\n", format="%f")

        f.close()
        return

    def computeMaxLengthEdges(self):
        """Compute the maximum length of the edges of the mesh, which elements
        are in me(d+1,nme) and vertices' coordinates in q(d,nq).

        """
        h = 0.
        # We loop over the edges and vectorized the computation over the
        # elements. Each edge will be computed several times (as many as
        # elements it belongs to), but the computation is vectorized
        # along the longest direction
        # For tetra
        for elt in [self.tetra, self.triangle, self.edge]:
            if elt.shape[0] > 0:
                for iLoc in np.arange(elt.shape[1]):
                    for jLoc in np.arange(iLoc + 1, elt.shape[1]):
                        # We are considering the edge with endpoints the
                        # vertices with local indices (iLoc,jLoc) in each
                        # element, with iLoc<jLoc
                        print(self.node[elt[:, iLoc]])
                        print(self.node[elt[:, jLoc]])
                        h = max(h,
                                np.max(np.linalg.norm(
                                    self.node[elt[:, iLoc]] -
                                    self.node[elt[:, jLoc]],
                                    axis=1))
                                )
                        # This computes the max of the 2-norm for each edge
                        # (iLoc,jLoc) of the elements.
        return h

    def computeDiameter(self):
        """Compute the diameter of the mesh

        """
        # Algorithm: choose a node a_0, find the farthest node
        # a_1. Consider the sphere of center c = 0.5*(a_0+a_1). Find the
        # node a_2 the farthest from c. If it belongs to the sphere,
        # then it is finished, otherwise, redo with a_0 := a_1 and a_1
        # := the farthest node just identified.
        a = self.node[0]
        i_max = np.argmax(np.linalg.norm(self.node - a, axis=1))
        b = self.node[i_max]

        c = 0.5 * (a + b)
        d = np.linalg.norm(b - a)  # tentative diameter
        i_max = np.argmax(np.linalg.norm(self.node - b, axis=1))
        a = b
        b = self.node[i_max]

        while np.linalg.norm(b - c) > 0.5 * d:
            c = 0.5 * (a + b)
            d = np.linalg.norm(b - a)  # tentative diameter
            i_max = np.argmax(np.linalg.norm(self.node - b, axis=1))
            a = b
            b = self.node[i_max]
        return d

    def computeEltVolumes(self):
        """Computes the volume of each mesh element

        """
        if self.dim == 2:
            n_elts = self.n_triangles
            elt = self.triangle  # in 2D, the elements are the triangles
        if self.dim == 3:
            n_elts = self.n_tetras
            elt = self.tetra  # in 3D, the elements are the tetraedras

        # With an array p x (n x n), np.linalg.det returns the det of
        # the p matrices of size n x n. The volume of an element is
        # given by |det(x1-x0,...xd-x0)|/d!.
        X = np.zeros((n_elts, self.dim, self.dim))

        for k in np.arange(self.dim):
            X[:, :, k] = self.node[elt[:, k + 1]] - self.node[elt[:, 0]]

        volume = np.abs(np.linalg.det(X)) / np.math.factorial(self.dim)
        return volume

    def extract_surf(self, triangle_list=[], label=[]):
        """Build the mesh corresponding to the triangles which indices in the
        mesh Th are given by the condition.

        Only one of triangle_list and label may be given, not both.

        triangle_list: build a surface mesh with these triangles
        label: build a surface mesh with the triangles having this label.

        Returns the nodes, triangles, and index mappings (direct and
        inverse).

        """
        # Add to the list of triangles the ones that have the required
        # label
        triangle_list = np.asarray(triangle_list)  # Cast to a numpy
        # array, just in case
        if len(label) > 0:
            for k in label:
                triangle_list = np.concatenate(triangle_list,
                                               np.where(self.triangle_label == k)[0])
        # In case some indices are repeated -> array of unique indices
        # of the triangles to be taken from the volume mesh
        triangle_list = np.unique(triangle_list)
        # Actually take the triangles from the original mesh
        triangle_inMesh = self.triangle.take(triangle_list, axis=0)
        # -> array of the unique indices of the nodes to be taken from
        # -> the volume mesh
        surfIndex_to_volIndex = np.unique(triangle_inMesh)
        # the inverse mapping
        volIndex_to_surfIndex = -np.ones(self.n_nodes, dtype=np.int32)
        volIndex_to_surfIndex[surfIndex_to_volIndex] = np.arange(surfIndex_to_volIndex.size)

        # the array of the nodes of the surface mesh
        node = self.node[surfIndex_to_volIndex]

        # Now, we just need to renumber the triangles
        triangle = volIndex_to_surfIndex[triangle_inMesh]

        # Return a Mesh type
        return node, triangle, surfIndex_to_volIndex, volIndex_to_surfIndex

    def test_if_surface(self):
        """Test if the mesh is a 3D volume, a 2D domain, or a 2D surface in
        3D.

        """
        if self.dim == 3 and self.n_tetras == 0:
            if np.all(self.node[:, 2] <= 10 ** (-15)):
                # This is a surface mesh, described by 2D coordinates
                self.dim = 2
                self.node = self.node[:, :2]  # keep only the first 2 coordinates
            else:
                self.is_surface = True
        return

    def construct_interface(self):
        """Construct the array of the interface: an array of size n_itfs x (d+2)
        where d is the number of vertices of each interface, 2 for
        triangular meshes, and 3 for tetras.

        self.interface is an array of size n_itfs x (d+2), where
        self.interface[:,0:d] are the d vertices of the interface, and
        self.interface[:,d] and self.interface[:,d+1] are its
        neighbours. If the interface is on the boundary of the
        computational domain, then self.interface[:,d+1] = -1.

        """
        # Get the elements and the dimension of the interfaces
        if self.dim == 2 or self.is_surface:
            dim = 2
            elt = np.sort(self.triangle, axis=1)
            n_elts = self.n_triangles
        else:
            dim = 3
            elt = np.sort(self.tetra, axis=1)
            n_elts = self.n_tetras
        # Assert that we can treat the problem with int64 integers for
        # the hash keys. hash_key = n1*n_nodes^2 + n2*n_nodes^1 +
        # n3*n_nodes^0 (en 3D), hash_key = n2*n_nodes^1 + n3*n_nodes^0
        # (en 2D). Or, n1, n2, n3 peuvent prendres toutes les valeurs de
        # n_nodes, de telle sorte que la clmé maximum est n^3+n^2+n (3D)
        # ou n^2+n (2D), on doit donc assurer que [n^3 +] n^2 + n <
        # n_max, où n_max = np.iinfo(np.int64). En pratique on vérifie
        # que (n+1)^dim < n_max, soit n+1
        assert n_elts < np.sqrt(np.iinfo(np.int64).max) - 1, \
            "The mesh has {} elts, which is too many for int64 hash keys used here".format(n_elts)

        # Allocate an array that will contain the interfaces (twice the
        # internal ones and once the boundary ones)
        itfs = np.zeros(((dim + 1) * n_elts, dim), dtype=np.int64)
        indices = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int64).reshape([4, 3])
        for i in np.arange(dim + 1):
            itfs[i * n_elts:(i + 1) * n_elts] = elt[:, indices[i, :dim]]
        i_elt = np.tile(np.arange(n_elts), dim + 1)  # ordre des éléments associés au itfs
        # Compute a hash key for each interface
        hash_vect = self.n_nodes ** np.arange(dim - 1, -1, -1, dtype=np.int64)
        hash_key = itfs.dot(hash_vect)
        # Hash key = i0*n_nodes^{d-1} ... + i{d-1}*n_nodes^0. Each
        # interface has a unique hash key because the indices have been
        # sorted The hash key may need int64 because it can be very
        # large. The largest needed is n^d-1, while the largest int64 is
        # np.iinfo(np.uint64)
        # i_max = np.iinfo(np.int64).max
        # n_max = (1+i_max)**(1/dim) --> 2. 10^6 en int64 et 2.5 10^6 en
        # uint64 pour la 3D, 3. 10^9 en int64 et 4.2 10^9 en uint64 pour
        # la 2D
        i_sorted = np.argsort(hash_key)
        tmp_var, indices, n_neigh = np.unique(hash_key[i_sorted],
                                              return_index=True, return_counts=True)
        self.n_interfaces = indices.shape[0]
        print("found {} interfaces".format(self.n_interfaces))
        self.interface = np.empty((self.n_interfaces, dim + 2), np.int32)
        # Vertices of the interfaces
        self.interface[:, :dim] = np.take(itfs, i_sorted[indices], axis=0)
        # Neighbours of the interfaces
        self.interface[:, dim] = np.take(i_elt, i_sorted[indices])
        # use -1 if no neighbour (boundary interface)
        self.interface[:, dim + 1] = np.where(n_neigh == 2, i_elt[i_sorted[indices + 1]], -1)
        # some tests: (dim+1)*n_elts = 2*n_interfaces-n_interface_au_bord
        n_bord = np.count_nonzero(n_neigh == 1)
        assert (dim + 1) * n_elts == 2 * self.n_interfaces - n_bord, \
            "(d+1)*n_elts, {} is not equal to 2*n_interfaces-n_interfaces_bord {}".format(
                (dim + 1) * n_elts, 2 * self.n_interfaces - n_bord)

        return

    def construct_neighbour(self):
        """Construct the array of the d+1 neighbours of the elements"""
        from scipy.sparse import csr_matrix
        # Get the elements and the dimension of the interfaces
        if self.dim == 2 or self.is_surface:
            dim = 2
            elt = self.triangle
            n_elts = self.n_triangles
        else:
            dim = 3
            elt = self.tetra
            n_elts = self.n_tetras
        # Cast the list of neighbours of each interface to some arrays
        # of int64, necessary for meshes such that n_elt +
        # (1+n_elts)*n_elts < n64_max, that is n_elts(2+n_elts)<n64_max,
        # so that n_elts<sqrt(1+n64_max)-1.
        assert n_elts < np.sqrt(np.iinfo(np.int64).max) - 1, \
            "The mesh has {} elts, which is many for int64 hash keys used here".format(n_elts)

        I = np.copy(self.interface[:, dim]).astype(np.int64)
        J = np.copy(self.interface[:, dim + 1]).astype(np.int64)
        # I->J are the interfaces, with I always inside the domain, and
        # J==-1 when it is outside

        # Hence we known that I,J are of size n_interfaces, and such
        # that 0<=I<n_elts and -1<=J<n_elts. We compute a unique
        # hash_key for each interface J->I. This is why the assertion
        # above is needed -- this create bugs if the max is reached.
        hash_key = J + (1 + n_elts) * I
        i_sorted = np.argsort(hash_key)
        tmp_var, indices, count = np.unique(hash_key[i_sorted],
                                            return_index=True, return_counts=True)
        # Each interface is unique except if it has several faces on the
        # boundary. It may have 1 to dim such faces. It is unique if it
        # has 1 face on the boundary.
        if np.any(count == 2):
            J[i_sorted[indices[np.where(count == 2)]]] = -2
        if np.any(count == 3):
            J[i_sorted[indices[np.where(count == 3)]]] = -3
            J[i_sorted[indices[np.where(count == 3)] + 1]] = -2
        J_min = -np.min(J)
        # Now the elts on the boundary have their neighbours labeled
        # -1,-2,-3 depending on the case.  Now -J_min <= J < n_elts and
        # 0 <= I < n_elts, where J_min = maximum number of neighbours
        # that a element can have (1 or 2 for triangles, 1, 2 or 3 for
        # tetras)
        IJ = np.hstack((I, J)) + J_min  # translate to have IJ>=0
        JI = np.hstack((J, I)) + J_min  # translate to have JI>=0
        N = csr_matrix((JI - J_min, (IJ, JI)))
        # The first |J_min| rows of the matrix contain the elements that
        # are along the boundary. Each of the following rows contains
        # the exactly dim+1 neighbours of the element
        self.neighbour = np.reshape(N.data[N.indptr[J_min]:], (n_elts, dim + 1))
        n3 = np.count_nonzero(self.neighbour == -3)  # exactly 3 faces on the boundary
        n2 = np.count_nonzero(self.neighbour == -2)  # more than 2 faces on the boundary
        n1 = np.count_nonzero(self.neighbour == -1)  # more than 1 face on the boundary
        # we have n1+n2+n3 = n_interfaces on the boundary
        print("# of faces on the boundary : {:10d} {:10d} {:10d} => {:>10s}".format(
            1, 2, 3, "total"))
        print("# of such elements         : {:10d} {:10d} {:10d} => {:10d}/{}".format(
            n1 - n2, n2 - n3, n3, n1, n_elts))
        return

    def construct_neighbour_naif(self):
        """Construct the array of the d+1 neighbours of the elements"""
        # Get the elements and the dimension of the interfaces
        if self.dim == 2 or self.is_surface:
            dim = 2
            elt = self.triangle
            n_elts = self.n_triangles
        else:
            dim = 3
            elt = self.tetra
            n_elts = self.n_tetras
        # We use the interfaces
        self.neighbour = -np.ones((n_elts, dim + 1), dtype=np.int32)
        face_on_the_boundary = np.zeros(n_elts)
        for itf in self.interface:
            i_K = itf[dim]  # voisin K
            i_L = itf[dim + 1]  # voisin L (peut-être =-1)
            i_s = itf[:dim]  # Les dim sommets de la face
            s_K = elt[i_K]  # Les dim+1 sommets de l'elt
            for k in np.arange(dim + 1):
                # Loop over the vertices of the elt K
                if s_K[k] not in i_s:
                    self.neighbour[i_K, k] = i_L
                    break
            if i_L > -1:
                s_L = elt[i_L]
                for k in np.arange(dim + 1):
                    # Loop over the vertices of the elt L
                    if s_L[k] not in i_s:
                        self.neighbour[i_L, k] = i_K
                        break
            else:
                face_on_the_boundary[i_K] += 1
        print("{} elts ayant 0 faces au bord".format(
            np.count_nonzero(face_on_the_boundary == 0)))
        print("{} elts ayant 1 faces au bord".format(
            np.count_nonzero(face_on_the_boundary == 1)))
        print("{} elts ayant 2 faces au bord".format(
            np.count_nonzero(face_on_the_boundary == 2)))
        print("{} elts ayant 3 faces au bord".format(
            np.count_nonzero(face_on_the_boundary == 3)))
        return face_on_the_boundary

    def construct_interface_stencil(self):
        """Construct the stencils of the interfaces"""
        pass
        return


def dof(idx, n_nodes):  # return a list of dof giving a list idx of periodic node relations

    # for i in np.arange(len(idx)):
    #     idx[i] = sorted(idx[i])
    # new_r = sorted(idx)  # we first order the list

    dof_id = 0
    dof_idx = -1 * np.ones(shape=n_nodes, dtype=int)

    if len(idx) == 0:
        return np.arange(n_nodes), n_nodes
    for rel in idx:
        i, j = rel
        if dof_idx[i] == -1 and dof_idx[j] == -1:
            dof_idx[i] = dof_idx[j] = dof_id
            dof_id += 1
        elif dof_idx[i] == -1 and dof_idx[j] != -1:
            dof_idx[i] = dof_idx[j]
        elif dof_idx[j] == -1 and dof_idx[i] != -1:
            dof_idx[j] = dof_idx[i]

    n_rest = dof_idx[dof_idx == -1].size
    dof_idx[dof_idx == -1] = dof_id + np.arange(0, n_rest)
    return dof_idx, dof_id + n_rest


# Functions for internal use 
def read_integer_number(f, line):
    """Read a number in file f. The number is either on the current line or
    on next one.
    
    """
    words = line.split()
    if len(words) == 1:
        # then the current line only has the keyword
        line = f.readline()
        n = np.int32(line)
    else:
        # the number is right here
        n = np.int32(words[1])
    return n


def read_floats_and_label(f, n, p):
    """Read an array of size nxp float and an integer label from opened file
    with id f.

    """
    x = np.zeros((n, p), dtype=np.float64)
    label = np.zeros(n, dtype=np.int32)
    for i in np.arange(n):
        line = f.readline()
        numbers = np.fromstring(line, sep=' ', dtype=np.float64, count=p + 1)
        x[i] = numbers[:p]
        label[i] = np.int32(numbers[p])
    return x, label


def read_ints_and_label(f, n, p):
    """Read an array of size nxp integers and an integer label from opened
    file with id f.
    
    """
    x = np.zeros((n, p), dtype=np.int32)
    label = np.zeros(n, dtype=np.int32)
    for i in np.arange(n):
        line = f.readline()
        numbers = np.fromstring(line, sep=' ', dtype=np.int32, count=p + 1)
        x[i] = numbers[:p]
        label[i] = numbers[p]
    return x, label
