import unittest
import numpy as np


def scalar_finite_diff(f, x, h):
    '''
    TODO(Q2): Implement the scalar finite difference function. Running this file should pass the unit test.

    Input:
    f is a function that returns a scalar output
    x is the input around which the finite difference is computed
    h has same dimension as x, and it contains the magnitude of deltas for computing the finite difference


    Output:
    return the gradients of f at x, which is a vector of the same dimension as x
    '''
    grad = np.zeros(x.shape)

    for i in range(len(x)):
        v = np.zeros(x.shape)
        v[i] = 1
        grad[i] = (f(x+h*v) - f(x-h*v))/(2*h)[i]

    return grad


class ScalarFiniteDiffTest(unittest.TestCase):
    """Tests for `finite_diff.py`."""

    def setUp(self):
        self._n = 5
        self._h = np.ones(self._n) * 1e-3


    def test_linear(self):
        for _ in range(10):
            m, x, k = np.random.random((3, self._n))
            f = lambda x : m @ (x + k) # np.dot(m,(x+k))

            grad_f_x = m
            print('grad_f_x', grad_f_x)
            fd_f_x = scalar_finite_diff(f, x, self._h)

            self.assertTrue(np.allclose(grad_f_x, fd_f_x))


    def test_trig(self):
        for _ in range(10):
            m, x, k, p = np.random.random((4, self._n))

            f = lambda x : p @ (np.sin(m * x) + k)

            grad_f_x = p * (np.cos(m * x) * m)
            fd_f_x = scalar_finite_diff(f, x, self._h)

            self.assertTrue(np.allclose(grad_f_x, fd_f_x))


if __name__ == '__main__':
    unittest.main()
