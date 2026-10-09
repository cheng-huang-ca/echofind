// Plain C ABI over edge.hpp, so Python (ctypes) can check parity on any platform and compiler,
// with no Python headers at build time. Complex arrays are interleaved (re, im) pairs.
// Each entry point exists for double (_f64, reference) and float (_f32, device).
#include "echofind/capi.h"

#include <cstring>

#include "echofind/edge.hpp"

namespace {

ef::Sos to_sos(const double* s, int n) {
  ef::Sos out(size_t(n > 0 ? n : 0));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 6; ++j) out[size_t(i)][size_t(j)] = s[6 * i + j];
  return out;
}

template <class T>
ef::cvec<T> to_cvec(const T* reim, int n) {
  ef::cvec<T> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[size_t(i)] = {reim[2 * i], reim[2 * i + 1]};
  return v;
}

template <class T>
void from_cvec(const ef::cvec<T>& v, T* reim) {
  for (size_t i = 0; i < v.size(); ++i) { reim[2 * i] = v[i].real(); reim[2 * i + 1] = v[i].imag(); }
}

ef::CfarConfig cfg(int kind, int n_ref, int n_guard, int k, int stride, double pfa) {
  ef::CfarConfig c;
  c.kind = ef::CfarKind(kind);
  c.n_ref = n_ref; c.n_guard = n_guard; c.k = k; c.stride = stride; c.pfa = pfa;
  return c;
}

template <class T>
int demod(const T* x, int n, double fs, double fc, const double* bp, int nbp, const double* lp,
          int nlp, int dec, double t0, T* out) {
  try {
    std::vector<T> xv(x, x + n);
    auto y = ef::demodulate<T>(xv, fs, fc, to_sos(bp, nbp), to_sos(lp, nlp), dec, t0);
    from_cvec(y, out);
    return int(y.size());
  } catch (...) { return -1; }
}

template <class T>
int mf(const T* x, int n, const T* rep, int m, T* out) {
  try {
    ef::MatchedFilter<T> f(to_cvec(rep, m), size_t(n));
    from_cvec(f.run(to_cvec(x, n)), out);
    return n;
  } catch (...) { return -1; }
}

template <class T>
int cfar(const T* p, int n, int kind, int n_ref, int n_guard, int k, int stride, double pfa,
         T* thr) {
  auto t = ef::cfar_threshold(std::vector<T>(p, p + n), cfg(kind, n_ref, n_guard, k, stride, pfa));
  std::memcpy(thr, t.data(), sizeof(T) * size_t(n));
  return n;
}

}  // namespace

extern "C" {

EF_API int ef_demodulate_f64(const double* x, int n, double fs, double fc, const double* bp,
                             int nbp, const double* lp, int nlp, int dec, double t0, double* out) {
  return demod<double>(x, n, fs, fc, bp, nbp, lp, nlp, dec, t0, out);
}
EF_API int ef_demodulate_f32(const float* x, int n, double fs, double fc, const double* bp,
                             int nbp, const double* lp, int nlp, int dec, double t0, float* out) {
  return demod<float>(x, n, fs, fc, bp, nbp, lp, nlp, dec, t0, out);
}
EF_API int ef_matched_filter_f64(const double* x, int n, const double* rep, int m, double* out) {
  return mf<double>(x, n, rep, m, out);
}
EF_API int ef_matched_filter_f32(const float* x, int n, const float* rep, int m, float* out) {
  return mf<float>(x, n, rep, m, out);
}
EF_API void ef_tvg_f64(double* p, int n, double fs, double c, double t0, double alpha,
                       double spreading, double r_min) {
  std::vector<double> v(p, p + n);
  ef::apply_tvg(v, fs, c, t0, alpha, spreading, r_min);
  std::memcpy(p, v.data(), sizeof(double) * size_t(n));
}
EF_API double ef_scale_factor(int kind, int n_ref, int k, double pfa) {
  return ef::scale_factor(cfg(kind, n_ref, 0, k, 1, pfa));
}
EF_API int ef_cfar_threshold_f64(const double* p, int n, int kind, int n_ref, int n_guard, int k,
                                 int stride, double pfa, double* thr) {
  return cfar<double>(p, n, kind, n_ref, n_guard, k, stride, pfa, thr);
}
EF_API int ef_cfar_threshold_f32(const float* p, int n, int kind, int n_ref, int n_guard, int k,
                                 int stride, double pfa, float* thr) {
  return cfar<float>(p, n, kind, n_ref, n_guard, k, stride, pfa, thr);
}

}  // extern "C"
