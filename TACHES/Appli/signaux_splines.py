"""
Creation de signaux aleatoires.
"""

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

def stationary_signal(shape, regularity, noise_func=lambda x: x, seed=None):
    """
    Create samples from stationary signal.

    Parameters
    ----------
    shape : tuple
        Number of samples in 1D or 2D.
    regularity : float
        Scalar number between 0 and 1 to control the regularity of
        the sampled signal; 0 being close to random whereas 1 being
        spatially highly correlated.
    noise_func : function
        Add noise to samples according to function, defaut is no
        noise added.
	seed : integer
		Seed of the random generator, defaut is a random seed.

    Returns
    -------
    pts : tuple of numpy vectors
        x and y coordinates of the samples.
    Zi : numpy array
        Array of samples with size shape.
    f : scipy interpolation function
        analytic spline function which outputs the samples.

    Examples
    --------
    # 1d case example with 20 non noisy samples at 0.6 regularity
    >>> x, z, f = stationary_signal((20,), 0.6)
    # plot results
    >>> plt.plot(x, z, 'bx', label='samples')
    >>> xi = np.arange(0, 1+0.01, 0.01)
    >>> plt.plot(xi, f(xi), 'r', label='analytic function')
    >>> plt.legend()
    """
    # init
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.rand()
        seed = int(str(seed)[2:10])
        np.random.seed(seed)
    shape = (shape[0], 1) if len(shape) == 1 else shape

    # node points value of spline surface
    shape_node = (int(4 + (1 - regularity) ** 2 * shape[0]),
                  int(4 + (1 - regularity) ** 2 * shape[1]))
    Z = np.random.randn(*shape_node)

    # interpolation
    x, y = np.arange(0, shape_node[1]) / (max(shape_node) - 1), \
           np.arange(0, shape_node[0]) / (max(shape_node) - 1)
    xi, yi = np.linspace(0, shape_node[1] - 1, shape[1]) / (max(shape_node) - 1), \
             np.linspace(0, shape_node[0] - 1, shape[0]) / (max(shape_node) - 1)
    f = sp.interpolate.RectBivariateSpline(y, x, Z)
    Zi = f(yi, xi)

    # reshape
    if shape[0] == 1 or shape[1] == 1:
        pts = xi if shape[0] == 1 else yi
        Zi = Zi.flatten()
        f = sp.interpolate.interp1d(pts, Zi, kind='cubic',
                                    fill_value = "extrapolate")
    else:
        pts = (xi, yi)

    return pts, noise_func(Zi), f, seed


def non_stationary_signal(shape, walk_prob=0.02, switch_prob=0.1, noise_func=lambda x: x, seed=None):
    """
    Create stationary signal samples.

    Parameters
    ----------
    shape : tuple
        Number of samples in 1D or 2D.
    walk_prob : float
        Positive scalar number to control smoothness in stationary
        states.
    switch_prob : float
        Positive scalar number between 0 and 1 to tune frequency
        of transition between stationary states.
    noise_func : function
        Add noise to samples according to function, defaut is no
        noise added.
	seed : integer
		Seed of the random generator, defaut is a random seed.

    Returns
    -------
    pts : tuple of numpy vectors
        x and y coordinates of the samples.
    Z : numpy array
        Array of samples with size shape.
    f : scipy interpolation function
        analytic spline function which outputs the samples.

    Examples
    --------
    # 1d case example with 40 non noisy samples
    >>> x, z, f = non_stationary_signal((40,), switch_prob=0.2)
    # plot results
    >>> plt.plot(x, z, 'bx', label='samples')
    >>> xi = np.arange(0, 1+0.01, 0.01)
    >>> plt.plot(xi, f(xi), 'r', label='analytic function')
    >>> plt.legend()

    """
    # init
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.rand()
        seed = int(str(seed)[2:10])
        np.random.seed(seed)
    shape = (shape[0], 1) if len(shape) == 1 else shape

    # first dimension
    z_x = np.zeros(shape[0])
    z_x[0] = np.random.randn()
    for x in range(1, shape[0]):
        w = int(np.random.rand() < switch_prob)
        z_x[x] = (1 - w) * (z_x[x - 1] + walk_prob * np.random.randn()) + w * np.random.randn()

    # second dimension
    z_y = np.zeros(shape[1])
    z_y[0] = np.random.randn()
    for y in range(1, shape[1]):
        w = int(np.random.rand() < switch_prob)
        z_y[y] = (1 - w) * (z_y[y - 1] + walk_prob * np.random.randn()) + w * np.random.randn()

    # outer product and spline function
    Z = np.outer(z_x, z_y)
    x, y = np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0])

    if shape[0] == 1 or shape[1] == 1:
        Z = Z.flatten()
        pts = np.linspace(0, 1, Z.size)
        f = sp.interpolate.interp1d(pts, Z, kind='cubic')
    else:
        pts = (x, y)
        f = sp.interpolate.RectBivariateSpline(y, x, Z)

    return pts, noise_func(Z), f, seed


def add_bivariate_noise(x, std1, ratio=20, prob=0.15):
    """
    Add bivariate normal noise to x.

    Parameters
    ----------
    x : numpy array
        Samples to which noise is added.
    std1 : float
        Positive scalar value, standard deviation of first distribution.
    ratio : float
        Positive scalar value, ratio between standard deviations of second
        and first distributions.
    prob : float
        Scalar number between 0 and 1, switching probability value between
        the two distributions.

    Returns
    -------
    numpy array of noisy samples.


    Examples
    --------
    # 1d case example with 20 non noisy samples at 0.6 regularity
    >>> x, z, f = stationary_signal((20,), 0.6)
    # add noise with std1=0.1
    >>> zn = add_bivariate_noise(z, 0.1)
    # plot results
    >>> plt.plot(x, z, 'bx', label='samples')
    >>> plt.plot(x, zn, 'ko', label='noisy samples')
    >>> xi = np.arange(0, 1+0.01, 0.01)
    >>> plt.plot(xi, f(xi), 'r', label='analytic function')
    >>> plt.legend()
    """
    # first distribution random values
    noise1 = std1 * np.random.randn(*x.shape)

    # second distribution random values
    noise2 = ratio * std1 * np.random.randn(*x.shape)

    # boolean draw and bivariate noise
    W = np.random.rand(*x.shape) > prob
    noise = noise1 * W + noise2 *~W

    return x + noise


def add_student_noise(x, std, df=1.5):
    """
    Add to x noise drawn with student's t distribution.

    Parameters
    ----------
    x : numpy array
        Samples to which noise is added.
    std : float
        Positive scalar value, standard deviation of the distribution.
    df : float
        Positive scalar value, degree of freedom. Distribution gets closer
        to normal as df goes to infinity.

    Returns
    -------
    numpy array of noisy samples.


    Examples
    --------
    # 1d case example with 20 non noisy samples at 0.6 regularity
    >>> x, z, f = stationary_signal((20,), 0.6)
    # add noise with std1=0.1
    >>> zn = add_student_noise(z, 0.1)
    # plot results
    >>> plt.plot(x, z, 'bx', label='samples')
    >>> plt.plot(x, zn, 'ko', label='noisy samples')
    >>> xi = np.arange(0, 1+0.01, 0.01)
    >>> plt.plot(xi, f(xi), 'r', label='analytic function')
    >>> plt.legend()
    """
    # distribution random values
    noise = std * np.random.standard_t(df, size=x.shape)

    return x + noise

if __name__ == "__main__":

    # STATIONNAIRE

    # definition de la fonction de bruit avec ecart type à 0.05 et probabilité de switch à 15%
    nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)

    # signal stationnaire bruité de 30 points et régularité à 0.9
    x1, z1, f1 = stationary_signal((30, ), 0.9, noise_func=nfunc)
    # signal stationnaire bruité de 30 points et régularité à 0.5
    x2, z2, f2 = stationary_signal((30, ), 0.5, noise_func=nfunc)

    # affichage
    xi = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x1, z1, 'rx', label='mesures')
    axs[0].plot(xi, f1(xi), 'b', label='signal')
    axs[0].legend()
    axs[1].plot(x2, z2, 'rx', label='mesures')
    axs[1].plot(xi, f2(xi), 'b', label='signal')
    axs[1].legend()

    # NON STATIONNAIRE

    # definition de la fonction de bruit avec ecart type à 0.01
    nfunc = lambda x: add_student_noise(x, 0.03)

    # signal non stationnaire bruité de 30 points
    x1, z1, f1 = non_stationary_signal((30, ), switch_prob=0.1, noise_func=nfunc)
    # signal non stationnaire bruité de 30 points et probabilité de switch plus grande
    x2, z2, f2 = non_stationary_signal((30, ), switch_prob=0.2, noise_func=nfunc)

    # affichage
    xi = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x1, z1, 'rx', label='mesures')
    axs[0].plot(xi, f1(xi), 'b', label='signal')
    axs[0].legend()
    axs[1].plot(x2, z2, 'rx', label='mesures')
    axs[1].plot(xi, f2(xi), 'b', label='signal')
    axs[1].legend()

