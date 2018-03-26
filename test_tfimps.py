import numpy as np
import tensorflow as tf
import tfimps

class TestTfimps(tf.test.TestCase):

    def testTransferMatrixForIdentity(self):
        phys_d = 2
        bond_d = 2

        A1 = A0 =  np.identity(phys_d)
        bond_matrices = np.array([A0, A1])

        imps = tfimps.Tfimps(phys_d, bond_d, bond_matrices)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps._transfer_matrix)
            self.assertAllClose(phys_d * np.identity(4), actual)

    def testDominantEigenvectorIsEigenvector(self):
        phys_d = 3
        bond_d = 5
        imps = tfimps.Tfimps(phys_d, bond_d)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            T = sess.run(imps._transfer_matrix)
            val, vec = sess.run(imps._dominant_eig)
            self.assertAllClose(T@vec, val*vec)

    def testIdentityHamiltonianHasEnergyOneDiagonalMPS(self):
        phys_d = 2
        bond_d = 5

        A0 = np.diag(np.random.rand(bond_d))
        A1 = np.diag(np.random.rand(bond_d))
        bond_matrices = np.array([A0, A1])

        imps = tfimps.Tfimps(phys_d, bond_d, bond_matrices)
        I = tf.eye(phys_d, dtype=tf.float64)
        h = tf.einsum('ij,kl->ikjl', I, I)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps.variational_e(h))
            self.assertAllClose(1, actual)

    def testIdentityHamiltonianHasEnergyOneRandomMPS(self):
        phys_d = 3
        bond_d = 5
        imps = tfimps.Tfimps(phys_d, bond_d)
        I = tf.eye(phys_d, dtype=tf.float64)
        h = tf.einsum('ij,kl->ikjl', I, I)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps.variational_e(h))
            self.assertAllClose(1, actual)

    def testAKLTStateHasCorrectEnergy(self):
        phys_d = 3
        bond_d = 2

        # Follow Annals of Physics Volume 326, Issue 1, January 2011, Pages 96-192
        # Note that even though the As are not symmetric, the transfer matrix is
        # TODO Failing because I've symmetrized the matrices
        Aplus = np.array([[0, 1/np.sqrt(2)], [0, 0]])
        Aminus = np.array([[0, 0], [-1/np.sqrt(2), 0]])
        A0 = np.array([[-1/2, 0], [0, 1/2]])
        bond_matrices = np.array([Aplus, A0, Aminus])

        aklt = tfimps.Tfimps(phys_d, bond_d, bond_matrices)

        # Spin 1 operators
        X = tf.constant([[0, 1, 0 ], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        iY = tf.constant([[0, -1, 0 ], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)

        XX = tf.einsum('ij,kl->ikjl', X, X)
        YY = - tf.einsum('ij,kl->ikjl', iY, iY)
        ZZ = tf.einsum('ij,kl->ikjl', Z, Z)

        hberg = XX + YY + ZZ
        h_aklt = hberg + tf.einsum('abcd,cdef->abef', hberg, hberg) / 3

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            aklt_energy = sess.run(aklt.variational_e(h_aklt))
            self.assertAllClose(-2/3, aklt_energy)
