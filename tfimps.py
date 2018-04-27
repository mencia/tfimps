import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers
import tensorflow as tf


#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

class Tfimps:
    """
    Infinite Matrix Product State class.
    """

    # TODO Allow for two-site unit cell and average energy between A_1 A_2 and A_2 A_1 ordering
    def __init__(self, phys_d, bond_d, A_matrices=None, symmetrize=True, hamiltonian=None):
        """
        :param phys_d: Physical dimension of the state e.g. 2 for spin-1/2 systems.
        :param bond_d: Bond dimension, the size of the A matrices.
        :param A_matrices: Square matrices of size `bond_d` forming the Matrix Product State.
        :param symmetrize: Boolean indicating A matrices are symmetrized.
        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two site Hamiltonian
        """

        self._session = tf.Session()

        self.phys_d = phys_d
        self.bond_d = bond_d
        self.hamiltonian = hamiltonian

        self.mps_manifold = pymanopt.manifolds.Stiefel(phys_d * bond_d, bond_d)

        if A_matrices is None:
            A_init = tf.reshape(self.mps_manifold.rand(), [phys_d, bond_d, bond_d])
            # A_init = self._symmetrize(np.random.rand(phys_d, bond_d, bond_d))

        else:
            A_init = A_matrices

        self.A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)

        if symmetrize:
            self.A = self._symmetrize(self.A)

        self._transfer_matrix = None
        self._right_eigenvector = tf.ones([self.bond_d ** 2], dtype=tf.float64)

        self._all_eig = tf.self_adjoint_eig(self.transfer_matrix)
        self._dominant_eig = None

        self._variational_energy = None

        if hamiltonian is not None:
            if symmetrize:
                self.variational_energy = self._add_variational_energy_symmetric_mps(hamiltonian)
            else:
                self.variational_energy = self._add_variational_energy_left_canonical_mps(hamiltonian)

    def correlator(self, operator, range):
        """
        Evaluate the correlation function of `operator` up to `range` sites.

        :param operator: Tensor of shape [phys_d, phys_d] giving single site operator.
        :param range: Maximum separation at which correlations required
        :return: Correlation function
        """
        dom_eigval, dom_eigvec = self.dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        #
        eigval, eigvec = self._all_eig
        eigtens = tf.reshape(tf.transpose(eigvec), [self.bond_d**2, self.bond_d, self.bond_d])
        #
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        L_AAbar_Rk = tf.einsum("stcd,kcd->kst", L_AAbar, eigtens)
        L_AAbar_Rk_Z = tf.einsum("kst,st->k", L_AAbar_Rk, operator)
        #
        AAbar_R = tf.einsum("sac,tbd,cd->stab", self.A, self.A, dom_eigmat)
        Lk_AAbar_R = tf.einsum("kab,stab->kst", eigtens, AAbar_R)
        Lk_AAbar_R_Z = tf.einsum("kst,st->k", Lk_AAbar_R, operator)
        #
        ss_list = []
        for n in np.arange(1,range):
            delta = (n-1) * tf.ones([self.bond_d ** 2], tf.float64)
            we = tf.reduce_sum(L_AAbar_Rk_Z * Lk_AAbar_R_Z * tf.pow(eigval, delta)) / dom_eigval ** (n + 1)
            ss_list.append(we)

        return ss_list

    @property
    def entanglement_spectrum(self):
        """
        Calculate the spectrum of eigenvalues of the reduced density matrix for a bipartition of
        an infinite system into two semi-infinite subsystems.

        :return: The `bond_d` eigenvalues of the reduced density matrix
        """
        pass

    # TODO Calculation of entanglement spectrum

    @property
    def transfer_matrix(self):
        if self._transfer_matrix is None:
            T = tf.einsum("sab,scd->acbd", self.A, self.A)
            T = tf.reshape(T, [self.bond_d**2, self.bond_d**2])
            self._transfer_matrix = T
        return self._transfer_matrix

    @property
    def right_eigenvector(self):

        feed_dict = {vec: guess}
        return self._session.run(objective, feed_dict)

        T = self.transfer_matrix
        vec = self._right_eigenvector
        next_vec = tf.einsum("ab,b->a", T, vec)
        norm_big = lambda vec, next: tf.greater(tf.norm(vec - next), 1e-6)
        increment = lambda vec, next: (next, tf.einsum("ab,b->a", T, next))
        vec, next_vec = tf.while_loop(norm_big, increment, [vec, next_vec])
        # Normalize using left vector
        left_vec = tf.reshape(tf.eye(self.bond_d, dtype=tf.float64), [self.bond_d ** 2])
        norm = tf.einsum('a,a->', left_vec, next_vec)
        self._right_eigenvector =  next_vec / norm




    @property
    def dominant_eig(self):
        if self._dominant_eig is None:
            eigvals, eigvecs = self._all_eig
            # We use cast to make the number an integer
            idx = tf.cast(tf.argmax(tf.abs(eigvals)), dtype=np.int32)# Why do abs?
            self._dominant_eig = eigvals[idx], eigvecs[:,idx] # Note that eigenvectors are given in columns, not rows!
        return self._dominant_eig

    def _symmetrize(self, M):
        # Symmetrize -- sufficient to guarantee transfer matrix is symmetric (but not necessary)
        M_lower = tf.matrix_band_part(M, -1, 0) #takes the lower triangular part of M (including the diagonal)
        return (M_lower + tf.matrix_transpose(M_lower)) / 2


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
        right_eigenmatrix = tf.reshape(self.right_eigenvector(), [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, right_eigenmatrix)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)

        return h_exp


if __name__ == "__main__":
    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 4

    # Pauli matrices. For now we avoid complex numbers
    X = tf.constant([[0,1],[1,0]], dtype=tf.float64)
    iY = tf.constant([[0,1],[-1,0]], dtype=tf.float64)
    Z = tf.constant([[1,0],[0,-1]], dtype=tf.float64)

    I = tf.eye(phys_d, dtype=tf.float64)

    XX = tf.einsum('ij,kl->ikjl', X, X)
    YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    X1 = (tf.einsum('ij,kl->ikjl', X, I) + tf.einsum('ij,kl->ikjl', I, X)) / 2

    # Heisenberg Hamiltonian
    # My impression is that staggered correlations go hand in hand with nonsymmetric A matrices
    h_xxx = XX + YY + ZZ

    # Ising Hamiltonian (at criticality). Exact energy is -4/pi=-1.27324...
    h_ising = - ZZ - X1

    # Initialize the MPS


    imps = Tfimps(phys_d, bond_d, hamiltonian=h_ising, symmetrize=False)
    problem = pymanopt.Problem(manifold=imps.mps_manifold, cost=imps.variational_energy, arg=imps.A)

    with tf.Session() as sess:
        point = sess.run(tf.reshape(imps.mps_manifold.rand(), [phys_d, bond_d, bond_d]))
        sess.run(tf.global_variables_initializer())
        print(sess.run(imps.variational_energy))

    print(problem.grad(point))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(imps.A))
    #
    #     print(problem.cost(imps.A))
    # # solver = pymanopt.solvers.ConjugateGradient()

    # Xopt = solver.solve(problem)

    # imps = Tfimps(phys_d, bond_d, symmetrize=True, hamiltonian=h_ising)
    #
    # train_op = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(imps.variational_e)
    #
    # with tf.Session() as sess:
    #
    #     sess.run(tf.global_variables_initializer())
    #
    #     for i in range(100):
    #         print(sess.run([imps.variational_e, train_op])[0])