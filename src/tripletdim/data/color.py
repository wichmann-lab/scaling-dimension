import numpy as np
from sklearn.utils import Bunch
from scipy import spatial


def _piecewise_gaussian(x: np.ndarray, alpha: float, mu: float, sigma1: float, sigma2: float) -> np.ndarray:
    square_root = (x - mu) / np.where(x < mu, sigma1, sigma2)
    return alpha * np.exp(-(square_root * square_root) / 2)


def _rgb_from_wavelength(wavelength: np.ndarray) -> np.ndarray:
    """ Approximate RGB colors for wavelengths.

    The perceived colors of wavelengths are approximated in
    the CIE 1931 XYZ color space by three sums of Gaussian functions [1].
    Then the XYZ color space is transformed to the RGB color space and gamma corrected [2][3].

    See `Wikipedia CIE 1931 color space`_ for an introduction to the color space and its approximation.

    Args:
        wavelength: Array of n wavelengths in Angstrom unit (0.1 nanometer).
    Returns:
        rgb: Array of RGB colors, shape (n, 3)


    .. _`Wikipedia CIE 1931 color space`: https://en.wikipedia.org/wiki/CIE_1931_color_space

    References
    ----------
    .. [1] Chris Wyman, Peter-Pike Sloan, and Peter Shirley. "Simple Analytic Approximations to the CIE XYZ Color
           Matching Functions", Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1-11, 2013
    .. [2] Michael Stokes, Matthew Anderson, Srinivasan Chandraseka, and Ricardo Motta.
          "A Standard Default Color Space for the Internet – sRGB, Version 1.10". November 5, 1996
    .. [3] "IEC 61966-2-1:1999". IEC Webstore. International Electrotechnical Commission. Retrieved 3 March 2017.
    """
    # CIE 1931 approximation of Wyman et al. (2013)
    # Parameters from Wikipedia: https://en.wikipedia.org/wiki/CIE_1931_color_space#Analytical_approximation
    x = (_piecewise_gaussian(wavelength, 1.056, 5998, 379, 310)
         + _piecewise_gaussian(wavelength, 0.362, 4420, 160, 267)
         + _piecewise_gaussian(wavelength, -0.065, 5011, 204, 262))
    y = (_piecewise_gaussian(wavelength, 0.821, 5688, 469, 405)
         + _piecewise_gaussian(wavelength, 0.286, 5309, 163, 311))
    z = (_piecewise_gaussian(wavelength, 1.217, 4370, 118, 360)
         + _piecewise_gaussian(wavelength, 0.681, 4590, 260, 138))
    xyz = np.c_[x, y, z]

    # Transformation and correction from sRGB specification (IEC 61966-2-1:1999), based on Stokes et al (1996).
    # Parameters from Wikipedia: https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    xyz2srgb_transform = np.array([
        [3.24096994, -1.53738318, -0.49861076],
        [-0.96924364, 1.8759675, 0.04155506],
        [0.05563008, -0.20397696, 1.05697151]])
    rgb = (xyz2srgb_transform @ xyz.T).T.clip(0, 1)
    rgb_gamma_correced = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb**(1 / 2.4) - 0.055)
    return rgb_gamma_correced


_EKMAN_WAVELENGTHS = np.asarray([434, 445, 465, 472, 490, 504, 537, 555, 584, 600, 610, 628, 651, 674])
_EKMAN_SIMILARITIES = np.array([0.86, 0.42, 0.42, 0.18, 0.06, 0.07, 0.04, 0.02, 0.07, 0.09, 0.12,
                                0.13, 0.16, 0.5, 0.44, 0.22, 0.09, 0.07, 0.07, 0.02, 0.04, 0.07,
                                0.11, 0.13, 0.14, 0.81, 0.47, 0.17, 0.1, 0.08, 0.02, 0.01, 0.02,
                                0.01, 0.05, 0.03, 0.54, 0.25, 0.1, 0.09, 0.02, 0.01, 0., 0.01,
                                0.02, 0.04, 0.61, 0.31, 0.26, 0.07, 0.02, 0.02, 0.01, 0.02, 0.,
                                0.62, 0.45, 0.14, 0.08, 0.02, 0.02, 0.02, 0.01, 0.73, 0.22, 0.14,
                                0.05, 0.02, 0.02, 0., 0.33, 0.19, 0.04, 0.03, 0.02, 0.02, 0.58,
                                0.37, 0.27, 0.2, 0.23, 0.74, 0.5, 0.41, 0.28, 0.76, 0.62, 0.55,
                                0.85, 0.68, 0.76])


def load_ekman_colors() -> Bunch:
    """ Load similarity of colors, based on human ratings.

    Similarities of 14 colors with wavelenth from 434nm to 674 nm
    were collected by Ekman (1954) in rating experiments.
    Subjects (n=31) rated the similarity of a color pair
    on a 5-point scale from 0=no similarity to 4=identical.
    The similarities were averaged across subjects and standardized to unit intervals.

    =================   ==============
    Objects                         14
    Dimensionality                   2
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <color_similarity>`.

    Returns:
        data: Dictionary-like object, with the following attributes.

            data: The similarity matrix of shape (14, 14).
            feature_names: The names of the similarity columns/rows.
            wavelength: The wavelengths of colors in nanometer.
            color: Approximation of the perceived colors in CIE RGB color space and hex format.


    >>> data = load_ekman_colors()
    >>> data.data.shape
    (14, 14)
    >>> data.feature_names[0], data.feature_names[-1]
    ('434nm', '674nm')


    .. topic:: References

        - Ekman, G. (1954). Dimensions of color vision. Journal of Psychology, 38, 467-474.

    """
    similarities = spatial.distance.squareform(_EKMAN_SIMILARITIES)
    color_matrix = _rgb_from_wavelength(_EKMAN_WAVELENGTHS * 10)  # input in Angstrom (0.1nm)
    colors = ["#{0:02x}{1:02x}{2:02x}".format(*rgb)
              for rgb in np.rint(255 * color_matrix).astype(np.int)]
    return Bunch(data=similarities,
                 feature_names=list(map("{}nm".format, _EKMAN_WAVELENGTHS)),
                 wavelength=_EKMAN_WAVELENGTHS,
                 color=colors)
