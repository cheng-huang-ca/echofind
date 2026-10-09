// Golden-vector parity: run each case of tests/golden/manifest.txt (written by
// scripts/w5_export.py from echofind.dsp) in double and in float, and compare with the Python
// output. Error is max |c++ - python| / max |python| (relative to the array's peak); NaN
// positions (CFAR cells without a full window) must match exactly.
// Pass: double <= 1e-9, float <= 1e-5. Exit code 1 on any failure.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "echofind/edge.hpp"

namespace {

std::string dir;

std::vector<double> load(const std::string& f) {
  std::ifstream in(dir + "/" + f, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + f);
  in.seekg(0, std::ios::end);
  const auto n = size_t(in.tellg()) / sizeof(double);
  in.seekg(0);
  std::vector<double> v(n);
  in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(double)));
  return v;
}

ef::Sos sos(const std::vector<double>& v) {
  ef::Sos s(v.size() / 6);
  for (size_t i = 0; i < s.size(); ++i)
    for (size_t j = 0; j < 6; ++j) s[i][j] = v[6 * i + j];
  return s;
}

template <class T>
ef::cvec<T> cv(const std::vector<double>& v) {
  ef::cvec<T> c(v.size() / 2);
  for (size_t i = 0; i < c.size(); ++i) c[i] = {T(v[2 * i]), T(v[2 * i + 1])};
  return c;
}

template <class T>
std::vector<double> flat(const ef::cvec<T>& c) {
  std::vector<double> v(2 * c.size());
  for (size_t i = 0; i < c.size(); ++i) { v[2 * i] = c[i].real(); v[2 * i + 1] = c[i].imag(); }
  return v;
}

template <class T>
std::vector<double> flat(const std::vector<T>& c) { return std::vector<double>(c.begin(), c.end()); }

// returns relative error, or +inf if sizes or NaN positions differ
double rel_err(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) != std::isnan(b[i])) return INFINITY;
    if (std::isnan(b[i])) continue;
    num = std::max(num, std::fabs(a[i] - b[i]));
    den = std::max(den, std::fabs(b[i]));
  }
  return den > 0 ? num / den : num;
}

template <class T>
std::vector<double> run(const std::string& kind, std::istringstream& ss) {
  if (kind == "demod") {
    double fs, fc, t0; int dec; std::string fx, fbp, flp, fo;
    ss >> fs >> fc >> dec >> t0 >> fx >> fbp >> flp >> fo;
    auto x = load(fx);
    return flat(ef::demodulate<T>(std::vector<T>(x.begin(), x.end()), fs, fc, sos(load(fbp)),
                                  sos(load(flp)), dec, t0));
  }
  if (kind == "mf") {
    std::string fx, fr, fo;
    ss >> fx >> fr >> fo;
    auto x = cv<T>(load(fx));
    ef::MatchedFilter<T> m(cv<T>(load(fr)), x.size());
    return flat(m.run(x));
  }
  if (kind == "cfar") {
    ef::CfarConfig c; int k; std::string fp, fo;
    ss >> k >> c.n_ref >> c.n_guard >> c.k >> c.stride >> c.pfa >> fp >> fo;
    c.kind = ef::CfarKind(k);
    auto p = load(fp);
    return flat(ef::cfar_threshold(std::vector<T>(p.begin(), p.end()), c));
  }
  if (kind == "tvg") {
    double fs, c, t0, a, sp, rm; std::string fp, fo;
    ss >> fs >> c >> t0 >> a >> sp >> rm >> fp >> fo;
    auto p = load(fp);
    std::vector<T> v(p.begin(), p.end());
    ef::apply_tvg(v, fs, c, t0, a, sp, rm);
    return flat(v);
  }
  throw std::runtime_error("unknown case " + kind);
}

}  // namespace

int main(int argc, char** argv) {
  dir = argc > 1 ? argv[1] : "tests/golden";
  std::ifstream man(dir + "/manifest.txt");
  if (!man) { std::cerr << "no manifest in " << dir << "\n"; return 2; }
  std::string line;
  int fails = 0, n = 0;
  while (std::getline(man, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream head(line);
    std::string name, kind;
    head >> name >> kind;
    std::string rest;
    std::getline(head, rest);
    std::string expect_file = rest.substr(rest.find_last_of(' ') + 1);
    auto expect = load(expect_file);
    std::istringstream s64(rest), s32(rest);
    const double e64 = rel_err(run<double>(kind, s64), expect);
    const double e32 = rel_err(run<float>(kind, s32), expect);
    const bool ok = e64 <= 1e-9 && e32 <= 1e-5;
    fails += !ok;
    ++n;
    std::printf("%-28s %-6s f64 %.2e  f32 %.2e  %s\n", name.c_str(), kind.c_str(), e64, e32,
                ok ? "ok" : "FAIL");
  }
  std::printf("%d/%d cases pass\n", n - fails, n);
  return fails ? 1 : 0;
}
