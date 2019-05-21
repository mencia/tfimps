import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers
import tensorflow as tf
from itertools import cycle
import random


class Tfimps:
    """
    Infinite Matrix Product State class.
    """

    def __init__(self, phys_d, bond_d, r_prec, uc_size, A_matrices=None, symmetrize=False, hamiltonian=None):
        """
        :param phys_d: Physical dimension of the state e.g. 2 for spin-1/2 systems.
        :param bond_d: Bond dimension, the size of the A matrices.
        :param A_matrices: Square matrices of size `bond_d` forming the Matrix Product State.
        :param symmetrize: Boolean indicating A matrices are symmetrized.
        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two site Hamiltonian
        :param two_site: Boolean indicating whether we a two-site unit cell
        """

        self._session = tf.Session()

        self.r_prec = r_prec
        self.phys_d = phys_d
        self.bond_d = bond_d
        self.hamiltonian = hamiltonian
        self.uc_size = uc_size


        if uc_size is not 1:
            self.mps_manifold = pymanopt.manifolds.Stiefel(phys_d * bond_d, bond_d, k=uc_size)

            if A_matrices is None:
                A_init = tf.reshape(self.mps_manifold.rand(), [uc_size, phys_d, bond_d, bond_d])

            else:
                A_init = A_matrices

            Stiefel_init = tf.reshape(A_init, [uc_size, self.phys_d * self.bond_d, self.bond_d])
            self.Stiefel = tf.get_variable("Stiefel_matrix", initializer=Stiefel_init, trainable=True, dtype=tf.float64)
            self.A = tf.reshape(self.Stiefel, [uc_size, self.phys_d, self.bond_d, self.bond_d])

            # Define the transfer matrix, all eigenvalues and dominant eigensystem.
            # AB
            # self._transfer_matrix_2s = self._add_transfer_matrix_2s()
            # self._right_eigenvector_2s = None

            self._transfer_matrix_cycle = self._add_transfer_matrix_cycle()
            self._right_eigenvector_cycle = None

            # Define the variational energy.
            if hamiltonian is not None:
                # self.variational_energy = self._add_variational_energy_left_canonical_mps_2s(hamiltonian)
                self.variational_energy = self._add_variational_energy_left_canonical_mps_cycle(hamiltonian)

        else:
            self.mps_manifold = pymanopt.manifolds.Stiefel(phys_d * bond_d, bond_d, k=1)


            if A_matrices is None:
                A_init = tf.reshape(self.mps_manifold.rand(), [phys_d, bond_d, bond_d])

            else:
                A_init = A_matrices

            Stiefel_init = tf.reshape(A_init, [self.phys_d * self.bond_d, self.bond_d])
            self.Stiefel = tf.get_variable("Stiefel_matrix", initializer=Stiefel_init, trainable=True, dtype=tf.float64)
            self.A = tf.reshape(self.Stiefel, [self.phys_d, self.bond_d, self.bond_d])

            self._transfer_matrix = None
            self._right_eigenvector = None
            self._all_eig = tf.self_adjoint_eig(self.transfer_matrix)
            self._dominant_eig = None


            self.variational_energy = self._add_variational_energy_left_canonical_mps(hamiltonian)



    # multi-site unit cell MPS

    def _add_transfer_matrix_cycle(self):

        index_list = cycle(list(range(self.uc_size)))
        Lee = []
        cycle_list = []
        for _ in range(2 * self.uc_size - 1):
            Lee.append(next(index_list))

        for i in range(self.uc_size):
            cycle_list.append(Lee[i:i + self.uc_size])

        if self.uc_size == 4:

            T_list = []
            for cy in cycle_list:

                [a,b,c,d] = cy
                A_prod = tf.einsum("zab,ybc,xcd,wde->zyxwae",
                                   self.A[a], self.A[b], self.A[c], self.A[d])
                T_list.append(tf.reshape(tf.einsum("zyxwab,zyxwcd->acbd", A_prod, A_prod),
                                         [self.bond_d ** 2, self.bond_d ** 2]))

        elif self.uc_size == 2:

            T_list = []
            for cy in cycle_list:
                [a, b] = cy
                A_prod = tf.einsum("sab,zbc->szac",
                                   self.A[a], self.A[b])
                T_list.append(tf.reshape(tf.einsum("szab,szcd->acbd", A_prod, A_prod),
                                         [self.bond_d ** 2, self.bond_d ** 2]))

        elif self.uc_size == 8:

            T_list = []
            for cy in cycle_list:
                [a, b, c, d, e, f, g, h] = cy
                A_prod = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi->zyxwvutsai",
                                   self.A[a], self.A[b], self.A[c], self.A[d], self.A[e], self.A[f], self.A[g],
                                   self.A[h])
                T_list.append(tf.reshape(tf.einsum("zyxwvutsab,zyxwvutscd->acbd", A_prod, A_prod),
                                         [self.bond_d ** 2, self.bond_d ** 2]))

        elif self.uc_size == 12:

            T_list = []
            for cy in cycle_list:
                [a, b, c, d, e, f, g, h, i, j, k, l] = cy
                A_prod = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi,rij,qjk,pkl,olm->zyxwvutsrqpoam",
                                   self.A[a], self.A[b], self.A[c], self.A[d], self.A[e], self.A[f], self.A[g],
                                   self.A[h], self.A[i], self.A[j], self.A[k], self.A[l])
                T_list.append(tf.reshape(tf.einsum("zyxwvutsrqpoab,zyxwvutsrqpocd->acbd", A_prod, A_prod),
                                         [self.bond_d ** 2, self.bond_d ** 2]))

        elif self.uc_size == 10:

            T_list = []
            for cy in cycle_list:
                [a, b, c, d, e, f, g, h, i, j] = cy
                A_prod = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi,rij,qjk->zyxwvutsrqak",
                                   self.A[a], self.A[b], self.A[c], self.A[d], self.A[e], self.A[f], self.A[g],
                                   self.A[h], self.A[i], self.A[j])
                T_list.append(tf.reshape(tf.einsum("zyxwvutsrqab,zyxwvutsrqcd->acbd", A_prod, A_prod),
                                         [self.bond_d ** 2, self.bond_d ** 2]))

        return T_list

    def _add_transfer_matrix_2s(self):

        if self.uc_size == 2:

            A_prod = tf.einsum("sab,zbc->szac", self.A[0], self.A[1])
            T = tf.einsum("szab,szcd->acbd", A_prod, A_prod)

        elif self.uc_size == 8:

            A_prod = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi->zyxwvutsai",
                               self.A[0], self.A[1], self.A[2], self.A[3], self.A[4], self.A[5], self.A[6], self.A[7])
            T = tf.einsum("zyxwvutsab,zyxwvutscd->acbd", A_prod, A_prod)

        elif self.uc_size == 12:

            A_prod = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi,rij,qjk,pkl,olm->zyxwvutsrqpoam",
                               self.A[0], self.A[1], self.A[2], self.A[3], self.A[4], self.A[5], self.A[6], self.A[7],
                               self.A[8], self.A[9], self.A[10], self.A[11])
            T = tf.einsum("zyxwvutsrqpoab,zyxwvutsrqpocd->acbd", A_prod, A_prod)

        T = tf.reshape(T, [self.bond_d ** 2, self.bond_d ** 2])
        return T

    @property
    def transfer_matrix_2s(self):
        if self._transfer_matrix_2s is None:
            self._transfer_matrix = self._add_transfer_matrix_2s()
        return self._transfer_matrix

    @property
    def right_eigenvector_2s(self):
        if self._right_eigenvector_2s is None:
            self._right_eigenvector_2s = self._right_eigenvector_power_method(self._transfer_matrix_2s)
            # Normalize using left vector
            left_vec = tf.reshape(tf.eye(self.bond_d, dtype=tf.float64), [self.bond_d ** 2])
            norm = tf.einsum('a,a->', left_vec, self._right_eigenvector_2s)
            self._right_eigenvector_2s = self._right_eigenvector_2s / norm
        return self._right_eigenvector_2s

    @property
    def right_eigenvector_cycle(self):
        if self._right_eigenvector_cycle is None:

            R_list = []
            for i in range(self.uc_size):
                R = self._right_eigenvector_power_method(self._transfer_matrix_cycle[i])
                left_vec = tf.reshape(tf.eye(self.bond_d, dtype=tf.float64), [self.bond_d ** 2])
                norm = tf.einsum('a,a->', left_vec, R)
                R = R / norm
                R_list.append(R)

            self._right_eigenvector_cycle = R_list

        return self._right_eigenvector_cycle

    def _add_variational_energy_left_canonical_mps_2s(self, hamiltonian):

        right_eigenmatrix = tf.reshape(self.right_eigenvector_2s, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A[self.uc_size-2], self.A[self.uc_size-2])
        BBbar_R = tf.einsum("uac,vbd,cd->uvab", self.A[self.uc_size-1], self.A[self.uc_size-1], right_eigenmatrix)
        L_AAbar_BBbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, BBbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_BBbar_R, hamiltonian)

        return h_exp

    def _add_variational_energy_left_canonical_mps_cycle(self, hamiltonian):

        iterator = cycle(list(range(self.uc_size)))

        ind_list = []
        for _ in range(self.uc_size + 1):
            ind_list.append(next(iterator))

        ind_pairs = []
        for i in range(self.uc_size):
            ind_pairs.append(ind_list[i:i + 2])

        ind_pairs_shift = []
        for i in range(self.uc_size-2, 2*self.uc_size-2):
            ind_pairs_shift.append(ind_pairs[i % self.uc_size])

        e_list = []

        for k in range(self.uc_size):

            [i,j] = ind_pairs_shift[k]
            ############################################################
            right_eigenmatrix = tf.reshape(self.right_eigenvector_cycle[k], [self.bond_d, self.bond_d])
            #################################################################


            L_AAbar = tf.einsum("sab,tac->stbc", self.A[i], self.A[i])
            BBbar_R = tf.einsum("uac,vbd,cd->uvab", self.A[j], self.A[j], right_eigenmatrix)
            L_AAbar_BBbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, BBbar_R)
            h_exp = tf.einsum("stuv,stuv->", L_AAbar_BBbar_R, hamiltonian)
            e_list.append(h_exp)

        return tf.reduce_sum(e_list)/self.uc_size

    # 1-site unit cell MPS

    @property
    def transfer_matrix(self):
        if self._transfer_matrix is None:
            T = tf.einsum("sab,scd->acbd", self.A, self.A)
            T = tf.reshape(T, [self.bond_d ** 2, self.bond_d ** 2])
            self._transfer_matrix = T
        return self._transfer_matrix

    @property
    def right_eigenvector(self):
        if self._right_eigenvector is None:
            self._right_eigenvector = self._right_eigenvector_power_method(self.transfer_matrix)
            # Normalize using left vector
            left_vec = tf.reshape(tf.eye(self.bond_d, dtype=tf.float64), [self.bond_d ** 2])
            norm = tf.einsum('a,a->', left_vec, self._right_eigenvector)
            self._right_eigenvector = self._right_eigenvector / norm
        return self._right_eigenvector


    def _add_variational_energy_left_canonical_mps(self, hamiltonian):
        """
        Evaluate the variational energy density for MPS in left canonical form

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, right_eigenmatrix)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)

        return h_exp


    def _right_eigenvector_power_method(self, T):
        dim = T.shape[0]
        vec = tf.ones([dim], dtype=tf.float64)
        next_vec = tf.einsum("ab,b->a", T, vec)
        # CHANGING POWER
        norm_big = lambda v1, v2: tf.reduce_any(
            tf.greater(tf.abs(v1 - v2), tf.constant(self.r_prec, shape=[dim], dtype=tf.float64)))
        increment = lambda v1, v2: (v2, tf.einsum("ab,b->a", T, v2))
        vec, next_vec = tf.while_loop(norm_big, increment, [vec, next_vec])
        # FIXED POWER
        # i = tf.constant(0)
        # norm_big = lambda i, v: tf.less(i, 2000)
        # increment = lambda i, v: (tf.add(i, 1), tf.einsum("ab,b->a", T, v))
        # i_fin, next_vec = tf.while_loop(norm_big, increment, [i, next_vec])
        return next_vec  # Not normalized


if __name__ == "__main__":

    ########################
    # TRANSVERSE FIELD ISING
    ########################

    # seed for generation of random number
    seed = 551

    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 2
    r_prec = 1e-14  # convergence condition for right eigenvector
    uc_size = 1

    # Hamiltonian parameters
    J = 1
    h = 0.48

    # Pauli spin=1/2 matrices. For now we avoid complex numbers
    X = tf.constant([[0, 1], [1, 0]], dtype=tf.float64)
    iY = tf.constant([[0, 1], [-1, 0]], dtype=tf.float64)
    Z = tf.constant([[1, 0], [0, -1]], dtype=tf.float64)

    I = tf.eye(phys_d, dtype=tf.float64)

    XX = tf.einsum('ij,kl->ikjl', X, X)
    YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    X1 = tf.einsum('ij,kl->ikjl', X, I)

    # Heisenberg Hamiltonian
    # My impression is that staggered correlations go hand in hand with nonsymmetric A matrices
    h_xxx = XX + YY + ZZ

    h_zz = tf.constant(J / 4, dtype=tf.float64)
    h_x1 = tf.constant(h / 2, dtype=tf.float64)

    # Ising Hamiltonian (at criticality). Exact energy is -4/pi=-1.27324...
    h_ising = -h_zz * ZZ - h_x1 * X1
    #h_ising = - ZZ - X1
    ham = h_ising

    #################################
    # AKLT
    #################################

    # phys_d = 3
    # bond_d = 2
    # r_prec = 1e-14
    # uc_size = 8
    #
    # # Follow Annals of Physics Volume 326, Issue 1, Pages 96-192.
    # # Note that even though the As are not symmetric, the transfer matrix is.
    # # We normalize these to be in left (and right) canonical form
    #
    # Aplus = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
    # Aminus = np.array([[0, 0], [-1 / np.sqrt(2), 0]])
    # A0 = np.array([[-1 / 2, 0], [0, 1 / 2]])
    # A_matrices = np.array([Aplus, A0, Aminus]) * np.sqrt(4 / 3)
    #
    # # Spin 1 operators.
    #
    # X = tf.constant([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
    # iY = tf.constant([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
    # Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)
    #
    # XX = tf.einsum('ij,kl->ikjl', X, X)
    # YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    # ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    #
    # hberg = XX + YY + ZZ
    # h_aklt = hberg + tf.einsum('abcd,cdef->abef', hberg, hberg) / 3
    # ham = h_aklt

    #######################################################################################
    #######################################################################################

    # Create initial point in Stiefel manifold

    def rand_seed_Stiefel(n, p, k, seed):
        """
        :param n: height = phys_d*bond_d
        :param p: width = bond_d
        :param k: number of manifolds
        :param seed: seed for random number generation
        """


        # CHECK THAT THIS SHOULD BE RANDN INSTEAD RAND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        np.random.seed(seed)
        if k == 1:
            X = np.random.randn(n, p)
            q, r = np.linalg.qr(X)
            return q

        X = np.zeros((k, n, p))
        for i in range(k):
            X[i], r = np.linalg.qr(np.random.randn(n, p))
        return X

    # Initialize the MPS

    imps = Tfimps(phys_d, bond_d, r_prec, uc_size, hamiltonian=ham, symmetrize=False)
    problem = pymanopt.Problem(manifold=imps.mps_manifold, cost=imps.variational_energy,
                               arg=imps.Stiefel)

    mingradnorm = 1e-20
    minstepsize = 1e-20
    inf = float('inf')
    maxiter = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # A_init = np.load("/rscratch/bm485/Code/deeplearning/tfimps/initial/"+"Stiefel_init_d"+str(phys_d)+"_D"+str(bond_d)+
        #                  "_uc_size"+str(uc_size)+".npy")
        # Stiefel_init = np.reshape(A_init, [phys_d * bond_d, bond_d])
        Stiefel_init = rand_seed_Stiefel(phys_d * bond_d, bond_d, uc_size, seed)
        print(problem.cost(Stiefel_init))
        solver = pymanopt.solvers.ConjugateGradient(maxtime=inf, maxiter=maxiter, mingradnorm=mingradnorm,
                                                    minstepsize=minstepsize)

        # THIS PRINTS IN AN OUTER FILE DURING CONVERGENCE AND STORES THE CONVERGED WAVE FUNCTION AND ENERGY IN A FILE
        ##########################################################
        # import sys
        # np.random.seed()
        # rand_number = np.random.rand(1)[0]
        # sys.stdout = open("logging/logging" + "_physd" + str(phys_d) + "_bondD" + str(bond_d) + "_h" + str(h) + "_rprec" +
        #                   str(r_prec) + "_minstepsize" + str(minstepsize) + "_mingradnorm" + str(mingradnorm) + "_pr" +
        #                   str(rand_number)[:5] + "_uc_size" + str(uc_size) + "_seed" + str(seed)+".csv", "w")
        ##########################################################
        Xopt = solver.solve(problem, x=Stiefel_init)
        wlist_1d = np.ravel(Xopt)
        print(wlist_1d)
        for i in range(len(wlist_1d)):
            out_string = ""
            out_string += str(wlist_1d[i])
            out_string += "\n"
            print(out_string)

    # THIS PRINTS IN THE SCREEN DURING CONVERGENCE AND STORES THE CONVERGED WAVE FUNCTION AND ENERGY IN A FILE
    on_wave = 1
    wlist_1d = np.ravel(Xopt)
    with open("logging/logging" + "_physd" + str(phys_d) + "_bondD" + str(bond_d) + "_h" + str(h) + "_rprec" + str(
            r_prec) + "_minstepsize" + str(minstepsize) + "_mingradnorm" + str(mingradnorm) + "_pr" + str(
        np.random.rand(1)[0])[:5] + "_uc_size" +str(uc_size)  + "_seed" + str(seed)+"_maxiter"+str(maxiter)+".csv", "w") as out_file:

        out_string = str(problem.cost(Xopt))
        out_string += "\n"
        out_file.write(out_string)

        if on_wave == 1:

            for i in range(len(wlist_1d)):
                out_string = ""
                out_string += str(wlist_1d[i])
                out_string += "\n"
                out_file.write(out_string)
