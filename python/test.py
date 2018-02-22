import numpy as np


def test_figtree():
    from figtree import figtree
    X = np.array([[1., 1.],
                  [-1., -1.],
                  [1., 1.]])
    h = 1.
    Q = np.array([1., 1., 1.])
    Y = np.array([[1., 1.],
                  [-1., -1.]])
    epsilon = 10 ** -5
    ret = figtree(X, h, Q, Y, epsilon)
    ans00 = np.exp(0) + np.exp(-(2**2 + 2**2)) + np.exp(0)
    ans01 = np.exp(-(2**2 + 2**2)) + np.exp(0) + np.exp(-(2**2 + 2**2))
    ans = np.array([[ans00, ans01]])
    assert ret is not None
    assert_str = "ans:{}, ret:{}".format(ans, ret)
    assert np.allclose(ret, ans, rtol=0, atol=epsilon), assert_str
    return ret
