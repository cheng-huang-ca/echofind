// EchoFind edge DSP: C++17 port of echofind.dsp (W2) for one ping on a handheld device.
//
// Chain, as in src/echofind/dsp/frontend.py:
//   passband -> band-pass (zero-phase SOS) -> mix to baseband -> low-pass (zero-phase SOS)
//   -> decimate -> matched filter (FFT correlation, unit-energy replica) -> |.|^2
//   -> CFAR (CA / GO / OS, strided reference cells) -> TVG -> 1-D candidates.
//
// Every function mirrors a Python function and is checked against it to <= 1e-5 relative error
// (tests/test_edge_parity.py). Filter coefficients (SOS) are designed offline with
// scipy.signal.butter and passed in: a device ships fixed coefficients, not a filter designer.
// All templates work in float (device) or double (reference).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ef {

template <class T> using cplx = std::complex<T>;
template <class T> using cvec = std::vector<cplx<T>>;
using Section = std::array<double, 6>;  // b0 b1 b2 a0 a1 a2 (scipy SOS row, a0 = 1)
using Sos = std::vector<Section>;

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------- zero-phase SOS filtering
// scipy.signal.sosfiltfilt with padtype="odd": odd extension of 3 * ntaps samples at each end,
// initial state sosfilt_zi(sos) scaled by the first sample, forward pass, backward pass.

// lfilter_zi for one biquad: solve (I - A^T) zi = b[1:] - a[1:] b0 (companion form).
inline std::array<double, 2> biquad_zi(const Section& s) {
  const double b0 = s[0], b1 = s[1], b2 = s[2], a1 = s[4] / s[3], a2 = s[5] / s[3];
  const double B0 = b1 / s[3] - a1 * b0 / s[3], B1 = b2 / s[3] - a2 * b0 / s[3];
  // [[1 + a1, -1], [a2, 1]] zi = [B0, B1]
  const double det = (1 + a1) + a2;
  return {(B0 + B1) / det, ((1 + a1) * B1 - a2 * B0) / det};
}

inline std::vector<std::array<double, 2>> sosfilt_zi(const Sos& sos) {
  std::vector<std::array<double, 2>> zi(sos.size());
  double scale = 1.0;
  for (size_t i = 0; i < sos.size(); ++i) {
    auto z = biquad_zi(sos[i]);
    zi[i] = {scale * z[0], scale * z[1]};
    const auto& s = sos[i];
    scale *= (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5]);
  }
  return zi;
}

// Accumulator type: biquad state and arithmetic run in double even for float signals. Six
// cascaded narrow-band sections lose about 1e-5 of peak in float32 state (golden test), and a
// Cortex-A FPU does scalar double at float speed, so only the stored signal stays float.
template <class V> struct Acc { using type = double; };
template <class T> struct Acc<std::complex<T>> { using type = std::complex<double>; };

// Direct form II transposed, in place, all sections in cascade, initial state zi * x0.
template <class V, class R>
void sosfilt_inplace(const Sos& sos, std::vector<V>& x, const std::vector<std::array<double, 2>>& zi,
                     V x0) {
  using A = typename Acc<V>::type;
  for (size_t k = 0; k < sos.size(); ++k) {
    const double b0 = sos[k][0], b1 = sos[k][1], b2 = sos[k][2];
    const double a1 = sos[k][4], a2 = sos[k][5];
    A z0 = zi[k][0] * A(x0), z1 = zi[k][1] * A(x0);
    for (auto& v : x) {
      const A xin = A(v);
      const A y = b0 * xin + z0;
      z0 = b1 * xin - a1 * y + z1;
      z1 = b2 * xin - a2 * y;
      v = V(y);
    }
  }
}

template <class V, class R = V>
std::vector<V> sosfiltfilt(const Sos& sos, const std::vector<V>& x) {
  int ntaps = 2 * int(sos.size()) + 1;
  int zb2 = 0, za2 = 0;
  for (const auto& s : sos) { zb2 += s[2] == 0.0; za2 += s[5] == 0.0; }
  ntaps -= std::min(zb2, za2);
  const size_t edge = size_t(3 * ntaps), n = x.size();
  if (n <= edge) throw std::invalid_argument("signal shorter than filter padding");
  std::vector<V> ext(n + 2 * edge);
  for (size_t i = 0; i < edge; ++i) ext[i] = R(2) * x[0] - x[edge - i];
  std::copy(x.begin(), x.end(), ext.begin() + edge);
  for (size_t i = 0; i < edge; ++i) ext[edge + n + i] = R(2) * x[n - 1] - x[n - 2 - i];
  const auto zi = sosfilt_zi(sos);
  sosfilt_inplace<V, R>(sos, ext, zi, ext.front());
  std::reverse(ext.begin(), ext.end());
  sosfilt_inplace<V, R>(sos, ext, zi, ext.front());
  std::reverse(ext.begin(), ext.end());
  return std::vector<V>(ext.begin() + edge, ext.begin() + edge + n);
}

// ---------------------------------------------------------------- IQ demodulation
// demodulate() in frontend.py: band-pass, then mix by 2 exp(-j 2 pi fc t), low-pass, decimate.
// Demodulator precomputes the mixing table for a fixed ping length once, as a device would;
// the phase is evaluated in double, since 2 pi fc t grows large over a 68 ms ping.
template <class T>
class Demodulator {
 public:
  Demodulator(size_t n, double fs, double fc, Sos bp, Sos lp, int decimate, double t0 = 0.0)
      : bp_(std::move(bp)), lp_(std::move(lp)), dec_(decimate), mix_(n) {
    for (size_t i = 0; i < n; ++i) {
      const double ph = -2.0 * kPi * fc * (t0 + double(i) / fs);
      mix_[i] = cplx<T>(T(2 * std::cos(ph)), T(2 * std::sin(ph)));
    }
  }
  cvec<T> run(const std::vector<T>& x) const {
    if (x.size() != mix_.size()) throw std::invalid_argument("demodulator: wrong input length");
    std::vector<T> xb = bp_.empty() ? x : sosfiltfilt<T, T>(bp_, x);
    cvec<T> mixed(xb.size());
    for (size_t i = 0; i < xb.size(); ++i) mixed[i] = xb[i] * mix_[i];
    cvec<T> bb = sosfiltfilt<cplx<T>, T>(lp_, mixed);
    cvec<T> out;
    out.reserve(bb.size() / size_t(dec_) + 1);
    for (size_t i = 0; i < bb.size(); i += size_t(dec_)) out.push_back(bb[i]);
    return out;
  }

 private:
  Sos bp_, lp_;
  int dec_;
  cvec<T> mix_;
};

template <class T>
cvec<T> demodulate(const std::vector<T>& x, double fs, double fc, const Sos& bp, const Sos& lp,
                   int decimate, double t0 = 0.0) {
  return Demodulator<T>(x.size(), fs, fc, bp, lp, decimate, t0).run(x);
}

// ---------------------------------------------------------------- FFT (radix 2, in place)
template <class T>
class FFT {
 public:
  explicit FFT(size_t n) : n_(n), tw_(n / 2), rev_(n) {
    if (n == 0 || (n & (n - 1))) throw std::invalid_argument("FFT size must be a power of 2");
    for (size_t k = 0; k < n / 2; ++k)
      tw_[k] = cplx<T>(T(std::cos(-2 * kPi * double(k) / double(n))),
                       T(std::sin(-2 * kPi * double(k) / double(n))));
    size_t bits = 0;
    while ((size_t(1) << bits) < n) ++bits;
    for (size_t i = 0; i < n; ++i) {
      size_t r = 0;
      for (size_t b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev_[i] = r;
    }
  }
  size_t size() const { return n_; }
  void forward(cvec<T>& a) const { run(a, false); }
  void inverse(cvec<T>& a) const {  // scaled by 1/n
    run(a, true);
    const T s = T(1) / T(n_);
    for (auto& v : a) v *= s;
  }

 private:
  void run(cvec<T>& a, bool inv) const {
    for (size_t i = 0; i < n_; ++i)
      if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
    for (size_t len = 2; len <= n_; len <<= 1) {
      const size_t half = len / 2, step = n_ / len;
      for (size_t i = 0; i < n_; i += len)
        for (size_t j = 0; j < half; ++j) {
          cplx<T> w = tw_[j * step];
          if (inv) w = std::conj(w);
          const cplx<T> u = a[i + j], v = a[i + j + half] * w;
          a[i + j] = u + v;
          a[i + j + half] = u - v;
        }
    }
  }
  size_t n_;
  cvec<T> tw_;
  std::vector<size_t> rev_;
};

inline size_t next_pow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ---------------------------------------------------------------- matched filter
// matched_filter() in matched_filter.py: correlate with the unit-energy replica, aligned so an
// echo starting at sample k peaks at sample k. The replica spectrum is computed once (init),
// as it would be on the device.
template <class T>
class MatchedFilter {
 public:
  MatchedFilter(const cvec<T>& replica, size_t n_samples)
      : m_(replica.size()), n_(n_samples), fft_(next_pow2(n_samples + replica.size() - 1)),
        H_(fft_.size()), buf_(fft_.size()) {
    double e = 0;
    for (const auto& v : replica) e += std::norm(std::complex<double>(v));
    const T g = T(1.0 / std::sqrt(e));
    std::fill(H_.begin(), H_.end(), cplx<T>(0));
    for (size_t i = 0; i < m_; ++i) H_[i] = std::conj(replica[m_ - 1 - i]) * g;
    fft_.forward(H_);
  }
  cvec<T> run(const cvec<T>& x) {
    if (x.size() != n_) throw std::invalid_argument("matched filter: wrong input length");
    std::fill(buf_.begin(), buf_.end(), cplx<T>(0));
    std::copy(x.begin(), x.end(), buf_.begin());
    fft_.forward(buf_);
    for (size_t i = 0; i < buf_.size(); ++i) buf_[i] *= H_[i];
    fft_.inverse(buf_);
    return cvec<T>(buf_.begin() + (m_ - 1), buf_.begin() + (m_ - 1) + n_);
  }

 private:
  size_t m_, n_;
  FFT<T> fft_;
  cvec<T> H_, buf_;
};

// ---------------------------------------------------------------- power and TVG
template <class T>
std::vector<T> power(const cvec<T>& y) {
  std::vector<T> p(y.size());
  for (size_t i = 0; i < y.size(); ++i) p[i] = std::norm(y[i]);
  return p;
}

// tvg.py: gain dB = spreading log10(max(r, r_min)) + 2 alpha r, r = c (t0 + i / fs) / 2.
template <class T>
void apply_tvg(std::vector<T>& p, double fs, double c, double t0, double alpha_db_per_m,
               double spreading = 40.0, double r_min = 1.0) {
  for (size_t i = 0; i < p.size(); ++i) {
    const double r = c * (t0 + double(i) / fs) / 2;
    const double g = spreading * std::log10(std::max(r, r_min)) + 2 * alpha_db_per_m * r;
    p[i] = T(double(p[i]) * std::pow(10.0, g / 10.0));
  }
}

// ---------------------------------------------------------------- CFAR
// cfar.py. Reference cells on each side at offsets G + 1 + m j, j = 0..n-1 (n = N/2, m = stride);
// cells whose full window does not fit get NaN thresholds and are never detections.
enum class CfarKind { CA = 0, GO = 1, SO = 2, OS = 3 };

struct CfarConfig {
  CfarKind kind = CfarKind::OS;
  int n_ref = 32, n_guard = 2, k = -1, stride = 1;
  double pfa = 1e-4;
  int os_rank() const { return k > 0 ? k : int(std::lround(0.75 * n_ref)); }
};

inline double pfa_ca(double a, int N) { return std::pow(1 + a / N, -N); }
inline double pfa_so(double t, int n) {
  double s = 0;
  for (int k = 0; k < n; ++k)
    s += std::exp(std::lgamma(n + k) - std::lgamma(k + 1) - std::lgamma(n)) * std::pow(2 + t, -(n + k));
  return 2 * s;
}
inline double pfa_go(double t, int n) { return 2 * std::pow(1 + t, -n) - pfa_so(t, n); }
inline double pfa_os(double a, int N, int k) {
  double l = 0;
  for (int i = 0; i < k; ++i) l += std::log(double(N - i)) - std::log(double(N - i) + a);
  return std::exp(l);
}

inline double pfa_of_scale(const CfarConfig& c, double s) {
  const int n = c.n_ref / 2;
  switch (c.kind) {
    case CfarKind::CA: return pfa_ca(s, c.n_ref);
    case CfarKind::SO: return pfa_so(s, n);
    case CfarKind::GO: return pfa_go(s, n);
    default: return pfa_os(s, c.n_ref, c.os_rank());
  }
}

// Scale factor giving cfg.pfa in exponential noise: closed form for CA, bisection otherwise
// (Pfa falls monotonically with the scale).
inline double scale_factor(const CfarConfig& c) {
  if (c.kind == CfarKind::CA) return c.n_ref * (std::pow(c.pfa, -1.0 / c.n_ref) - 1);
  auto f = [&](double s) { return std::log(pfa_of_scale(c, s)) - std::log(c.pfa); };
  double lo = 1e-12, hi = 1.0;
  while (f(hi) > 0) hi *= 2;
  for (int it = 0; it < 200 && hi - lo > 1e-14 * hi; ++it) {
    const double mid = 0.5 * (lo + hi);
    (f(mid) > 0 ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// `scale` < 0 means "compute it": pass scale_factor(c) once when calling every ping.
template <class T>
std::vector<T> cfar_threshold(const std::vector<T>& p, const CfarConfig& c, double scale = -1) {
  const int n = c.n_ref / 2, g = c.n_guard, m = c.stride;
  const long L = long(p.size()), R = g + 1 + m * (n - 1);
  std::vector<T> thr(p.size(), std::numeric_limits<T>::quiet_NaN());
  if (L <= 2 * R) return thr;
  const T s = T(scale >= 0 ? scale : scale_factor(c));
  const int kk = c.os_rank();
  std::vector<T> win(size_t(2 * n));
  for (long i = R; i < L - R; ++i) {
    if (c.kind == CfarKind::OS) {
      for (int j = 0; j < n; ++j) {
        const long o = g + 1 + long(m) * j;
        win[size_t(j)] = p[size_t(i - o)];
        win[size_t(n + j)] = p[size_t(i + o)];
      }
      std::nth_element(win.begin(), win.begin() + (kk - 1), win.end());
      thr[size_t(i)] = s * win[size_t(kk - 1)];
      continue;
    }
    T lag = 0, lead = 0;
    for (int j = 0; j < n; ++j) {
      const long o = g + 1 + long(m) * j;
      lag += p[size_t(i - o)];
      lead += p[size_t(i + o)];
    }
    if (c.kind == CfarKind::CA) thr[size_t(i)] = s * (lag + lead) / T(c.n_ref);
    else if (c.kind == CfarKind::GO) thr[size_t(i)] = s * std::max(lag, lead);
    else thr[size_t(i)] = s * std::min(lag, lead);
  }
  return thr;
}

// ---------------------------------------------------------------- 1-D candidates
// candidates.py for one beam: each run of detections is one candidate, described at its peak.
struct Candidate {
  int peak;            // sample index of the peak
  double range_m, level_db, snr_db, extent_m;
  int n_cells;
};

template <class T>
std::vector<Candidate> candidates_1d(const std::vector<T>& p, const std::vector<T>& thr,
                                     double scale, double fs, double c, double t0,
                                     double min_snr_db = -1e300) {
  std::vector<Candidate> out;
  const size_t L = p.size();
  const double dr = c / (2 * fs);
  size_t i = 0;
  while (i < L) {
    if (!(std::isfinite(double(thr[i])) && p[i] > thr[i])) { ++i; continue; }
    size_t j = i, pk = i;
    while (j < L && std::isfinite(double(thr[j])) && p[j] > thr[j]) {
      if (p[j] > p[pk]) pk = j;
      ++j;
    }
    const double noise = double(thr[pk]) / scale;
    Candidate cd{int(pk), c * (t0 + double(pk) / fs) / 2, 10 * std::log10(double(p[pk])),
                 10 * std::log10(double(p[pk]) / noise), double(j - i) * dr, int(j - i)};
    if (cd.snr_db >= min_snr_db) out.push_back(cd);
    i = j;
  }
  return out;
}

}  // namespace ef
