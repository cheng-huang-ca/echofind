// pybind11 module _ef_edge: the edge front end and compiled classifiers as NumPy functions.
// Built with -DEF_PYBIND=ON (CI on Linux; on a Raspberry Pi with python3-dev). On Windows the
// ctypes C ABI (src/capi.cpp) is used instead, since MinGW-built extensions do not load in an
// MSVC-built CPython.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "echofind/capi.h"
#include "echofind/edge.hpp"

namespace py = pybind11;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using carr = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

static ef::Sos to_sos(const arr& a) {
  if (a.ndim() != 2 || a.shape(1) != 6) throw std::invalid_argument("SOS must be (n, 6)");
  ef::Sos s(size_t(a.shape(0)));
  auto r = a.unchecked<2>();
  for (py::ssize_t i = 0; i < a.shape(0); ++i)
    for (py::ssize_t j = 0; j < 6; ++j) s[size_t(i)][size_t(j)] = r(i, j);
  return s;
}

static carr to_np(const ef::cvec<double>& v) {
  carr out(py::ssize_t(v.size()));
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

PYBIND11_MODULE(_ef_edge, m) {
  m.doc() = "EchoFind edge DSP (C++17 port of echofind.dsp) and generated classifiers";
  m.def("demodulate", [](const arr& x, double fs, double fc, const arr& bp, const arr& lp,
                         int decimate, double t0) {
    std::vector<double> v(x.data(), x.data() + x.size());
    return to_np(ef::demodulate<double>(v, fs, fc, to_sos(bp), to_sos(lp), decimate, t0));
  }, py::arg("x"), py::arg("fs"), py::arg("fc"), py::arg("bp"), py::arg("lp"),
        py::arg("decimate"), py::arg("t0") = 0.0);
  m.def("matched_filter", [](const carr& x, const carr& rep) {
    ef::cvec<double> xv(x.data(), x.data() + x.size()), rv(rep.data(), rep.data() + rep.size());
    ef::MatchedFilter<double> f(rv, xv.size());
    return to_np(f.run(xv));
  });
  m.def("cfar_threshold", [](const arr& p, int kind, int n_ref, int n_guard, int k, int stride,
                             double pfa) {
    ef::CfarConfig c;
    c.kind = ef::CfarKind(kind);
    c.n_ref = n_ref; c.n_guard = n_guard; c.k = k; c.stride = stride; c.pfa = pfa;
    auto t = ef::cfar_threshold(std::vector<double>(p.data(), p.data() + p.size()), c);
    return arr(py::ssize_t(t.size()), t.data());
  });
  m.def("model_names", []() {
    std::vector<std::string> n;
    for (int i = 0; i < ef_model_count(); ++i) n.emplace_back(ef_model_name(i));
    return n;
  });
  m.def("model_score", [](int i, const arr& X) {
    if (X.ndim() != 2 || X.shape(1) != ef_model_n_features(i))
      throw std::invalid_argument("X must be (n, n_features)");
    arr out(X.shape(0));
    for (py::ssize_t r = 0; r < X.shape(0); ++r)
      out.mutable_data()[r] = ef_model_score_f64(i, X.data() + r * X.shape(1));
    return out;
  });
}
