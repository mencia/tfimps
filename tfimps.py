import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers
import tensorflow as tf


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



        if uc_size not 1:
            self.mps_manifold = pymanopt.manifolds.Stiefel(phys_d * bond_d, bond_d, k=uc_size)

            if A_matrices is None:
                A_init = tf.reshape(self.mps_manifold.rand(), [uc_size, phys_d, bond_d, bond_d])

            else:
                A_init = A_matrices

            Stiefel_init = tf.reshape(A_init, [uc_size, self.phys_d * self.bond_d, self.bond_d])
            self.Stiefel = tf.get_variable("Stiefel_matrix", initializer=Stiefel_init, trainable=True, dtype=tf.float64)
            self.Stiefel_p = tf.reshape(self.Stiefel, [uc_size, self.phys_d, self.bond_d, self.bond_d])
            self.A = self.Stiefel_p[0]
            self.B = self.Stiefel_p[0]

            # Define the transfer matrix, all eigenvalues and dominant eigensystem.
            # AB
            self._transfer_matrix_2s = self._add_transfer_matrix_2s()
            self._right_eigenvector_2s = None

            # Define the variational energy.
            if hamiltonian is not None:
                self.variational_energy = self._add_variational_energy_left_canonical_mps_2s(hamiltonian)

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

            if hamiltonian is not None:
                if symmetrize:
                    self.variational_energy = self._add_variational_energy_symmetric_mps(hamiltonian)
                else:
                    self.variational_energy = self._add_variational_energy_left_canonical_mps(hamiltonian)


        if symmetrize:
            self.A = self._symmetrize(self.A)
            self.B = self._symmetrize(self.B)


    # 2-site unit cell MPS

    def _add_transfer_matrix_2s(self, n_sites):

        A1 = self.A
        A2 = self.B

        A_prod_2 = tf.einsum("sab,zbc->szac", A1, A2)

        #8
        A_prod_8 = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi->zyxwvutsrqpoai",
                              A1, A2, A3, A4, A5, A6, A7, A8)

        # 12
        A_prod_12 = tf.einsum("zab,ybc,xcd,wde,vef,ufg,tgh,shi,rij,qjk,pkl,olm->zyxwvutsrqpoam",
                              A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)

        T = tf.einsum("szab,szcd->acbd", A_prod_2, A_prod_2)
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

    def _add_variational_energy_left_canonical_mps_2s(self, hamiltonian):
        """
        Evaluate the variational energy density for MPS in left canonical form

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        right_eigenmatrix = tf.reshape(self.right_eigenvector_2s, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        BBbar_R = tf.einsum("uac,vbd,cd->uvab", self.B, self.B, right_eigenmatrix)
        L_AAbar_BBbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, BBbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_BBbar_R, hamiltonian)

        return h_exp

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

    @property
    def dominant_eig(self):
        if self._dominant_eig is None:
            eigvals, eigvecs = self._all_eig
            idx = tf.cast(tf.argmax(tf.abs(eigvals)), dtype=np.int32)
            self._dominant_eig = eigvals[idx], eigvecs[:, idx]  # Note that eigenvectors are given in columns, not rows!
        return self._dominant_eig

    def _symmetrize(self, M):
        # Symmetrize -- sufficient to guarantee transfer matrix is symmetric (but not necessary)
        M_lower = tf.matrix_band_part(M, -1, 0)
        M_diag = tf.matrix_band_part(M, 0, 0)
        return M_lower + tf.matrix_transpose(M_lower) - M_diag

    def _add_variational_energy_symmetric_mps(self, hamiltonian):
        """
        Evaluate the variational energy density for symmetric MPS (not using canonical form)

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        dom_eigval, dom_eigvec = self.dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, dom_eigmat)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)

        return h_exp / tf.square(dom_eigval)

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

    def _add_variational_energy_left_canonical_mps_onsite_and_NN(self, h_NN, h_onsite):
        """
        Evaluate the variational energy density for MPS in left canonical form

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian: h_NN + h_onsite,
        e.g. Transverse Field Ising. Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, right_eigenmatrix)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp_NN = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, h_NN)

        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        h_exp_onsite = tf.einsum("sab,tac,bc,st->", self.A, self.A, right_eigenmatrix, h_onsite)

        return h_exp_NN + h_exp_onsite

    def _right_eigenvector_power_method(self, T):
        dim = T.shape[0]
        vec = tf.ones([dim], dtype=tf.float64)
        next_vec = tf.einsum("ab,b->a", T, vec)
        norm_big = lambda v1, v2: tf.reduce_any(
            tf.greater(tf.abs(v1 - v2), tf.constant(self.r_prec, shape=[dim], dtype=tf.float64)))
        increment = lambda v1, v2: (v2, tf.einsum("ab,b->a", T, v2))
        vec, next_vec = tf.while_loop(norm_big, increment, [vec, next_vec])
        return next_vec  # Not normalized


if __name__ == "__main__":

    ########################
    # TRANSVERSE FIELD ISING
    ########################

    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 16
    r_prec = 1e-14  # convergence condition for right eigenvector
    two_site = False

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

    #################################
    # AKLT
    #################################

    # phys_d = 3
    # bond_d = 2
    # r_prec = 1e-14

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

    #######################################################################################
    #######################################################################################

    # Initialize the MPS

    imps = Tfimps(phys_d, bond_d, r_prec, two_site, hamiltonian=h_ising, symmetrize=False)
    problem = pymanopt.Problem(manifold=imps.mps_manifold, cost=imps.variational_energy,
                               arg=imps.Stiefel)

    mingradnorm = 1e-20
    minstepsize = 1e-20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solver = pymanopt.solvers.ConjugateGradient(maxtime=float('inf'), maxiter=100000, mingradnorm=mingradnorm,
                                                    minstepsize=minstepsize)
        # import sys
        # sys.stdout = open("logging/logging" + "_physd" + str(phys_d) + "_bondD" + str(bond_d) + "_h" + str(h) + "_rprec" +
        #                   str(r_prec) + "_minstepsize" + str(minstepsize) + "_mingradnorm" + str(mingradnorm) + "_pr" +
        #                   str(np.random.rand(1)[0])[:5] + "_2site" +str(two_site) +".csv", "w")

        Xopt = solver.solve(problem)
        # print(Xopt)
        # print(problem.cost(Xopt))

    # on_wave = 1
    # wlist_1d = np.ravel(Xopt)
    # with open("logging" + "_physd" + str(phys_d) + "_bondD" + str(bond_d) + "_h" + str(h) + "_rprec" + str(
    #         r_prec) + "_minstepsize" + str(minstepsize) + "_mingradnorm" + str(mingradnorm) + "_pr" + str(
    #     np.random.rand(1)[0])[:5] + "_2site" + str(two_site) +".csv", "w") as out_file:
    #
    #     out_string = str(problem.cost(Xopt))
    #     out_string += "\n"
    #     out_file.write(out_string)
    #
    #     if on_wave == 1:
    #
    #         for i in range(len(wlist_1d)):
    #             out_string = ""
    #             out_string += str(wlist_1d[i])
    #             out_string += "\n"
    #             out_file.write(out_string)