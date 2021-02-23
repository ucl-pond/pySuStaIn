#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>


namespace py = pybind11;

// Actual C++ snippet. See `docstr` below for argument info.
template <typename T>
py::array_t<T> kernel_sum (const py::array_t<T>& X,
                           const py::array_t<T>& Y,
                           const py::array_t<T>& invbw,
                           const py::array_t<T>& norm) {
    // Gather input shape info
    auto bX = X.template unchecked<2>();
    auto bY = Y.template unchecked<2>();
    auto lenX = bX.shape(0);
    auto ndim = bX.shape(1);
    auto lenY = bY.shape(0);
    auto ndimY = bY.shape(1);
    auto binvbw = invbw.template unchecked<1>();
    auto bnorm = norm.template unchecked<1>();

    // Check data integrity
    if (ndim != ndimY) {
        throw std::runtime_error(
            "Points in X and Y must have the same number of dimensions.");
    }
    if (lenX != binvbw.shape(0) || lenX != bnorm.shape(0)) {
        throw std::runtime_error(
            "X, invbw and norm must have the same lengths.");
    }

    // Set up output: shape (lenY)
    py::array_t<T> eval(lenY);
    auto beval = eval.template mutable_unchecked<1>();

    // Get squared distances for all in Y to all in X and sum up the PDF value
    // from the gaussian kernel PDF for each Y.
    T diff_ij;
    T dist2_ij;
    for (unsigned int i = 0; i < lenY; ++i) {  // Loop over Y elements
        beval(i) = 0.;
        for (unsigned int j = 0; j < lenX; ++j) {  // Loop over X elements
            dist2_ij = 0.;
            for (unsigned int k = 0; k < ndim; ++k) {  // Loop over features
                diff_ij = bY(i, k) - bX(j, k);
                dist2_ij += diff_ij * diff_ij;
            }
            // PDF_i = sum_{j in X} (norm_j * exp(-0.5 * dist2_ij * invbw_j^2)
            beval(i) += bnorm(j) * std::exp(-0.5 * dist2_ij *
                                            binvbw(j) * binvbw(j));
        }
    }

    return eval;
}


PYBIND11_PLUGIN(backend) {
    py::module m("backend", R"pbdoc(
        Pybind11 C++ backend for awkde
        ------------------------------

        .. currentmodule:: awkde_backend

        .. autosummary::
           :toctree: _generate

           kernel_sum
    )pbdoc");

    auto docstr = R"pbdoc(
                    kernel_sum

                    Takes an array of kernel points `X` and points `Y` to
                    evaluate the KDE at and returns the KDE PDF values for
                    each point in `Y`.

                    Parameters
                    ----------
                    X : double array, shape (len(X), ndim)
                        Data points defining each kernel position. Each row is
                        a point, each column is a feature.
                    Y : double array, shape (len(Y), ndim)
                        Data points we want to evaluate the KDE at. Each row is
                        a point, each column is a feature.
                    invbw : double array, shape (len(X))
                        Inverse kernel bandwidth, acting as :math:`1 / sigma^2`.
                    norm : double array, shape (len(X))
                        Kernel gaussian norm for `ndim` dimensions.

                    Returns
                    -------
                    eval : float array, shape (len(Y))
                        The probability from the KDE PDF for each point in `Y`.
                  )pbdoc";

    // Define the actual template typess
    m.def("kernel_sum", &kernel_sum<double>, docstr,
          py::arg("X"), py::arg("Y"), py::arg("invbw"), py::arg("norm"));
    m.def("kernel_sum", &kernel_sum<float>, "",
          py::arg("X"), py::arg("Y"), py::arg("invbw"), py::arg("norm"));

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
    return m.ptr();
}
