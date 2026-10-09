// ef_bench: time one ping through the edge chain, per stage, pinned to one core.
//
//   ef_bench <config.txt> [--pings N] [--core K] [--f64]
//
// The config (written by scripts/w5_export.py) names the passband ping, the SOS filters, the
// replica, the CFAR settings and a table of candidate feature rows for the classifiers. Each
// ping runs: demodulate -> matched filter -> power -> CFAR -> TVG -> candidates, then scores
// `cands_per_ping` candidates with every generated model. Output: one JSON object with p50,
// p95 and max per stage (microseconds), peak resident memory and the machine description.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "echofind/capi.h"
#include "echofind/edge.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sched.h>
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::vector<double> load(const std::string& f) {
  std::ifstream in(f, std::ios::binary);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", f.c_str()); std::exit(2); }
  in.seekg(0, std::ios::end);
  std::vector<double> v(size_t(in.tellg()) / sizeof(double));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(double)));
  return v;
}

bool pin(int core) {
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << core) != 0 &&
         SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(core, &s);
  return sched_setaffinity(0, sizeof(s), &s) == 0;
#endif
}

double peak_rss_mb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  return double(pmc.PeakWorkingSetSize) / 1048576.0;
#else
  rusage r;
  getrusage(RUSAGE_SELF, &r);
#if defined(__APPLE__)
  return double(r.ru_maxrss) / 1048576.0;
#else
  return double(r.ru_maxrss) / 1024.0;
#endif
#endif
}

struct Stats { double p50, p95, max; };
Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double p) { return v[std::min(v.size() - 1, size_t(p * double(v.size())))]; };
  return {q(0.5), q(0.95), v.back()};
}

template <class T>
int run(std::map<std::string, std::string>& cfg, int pings, const std::string& base) {
  auto path = [&](const char* k) { return base + "/" + cfg[k]; };
  auto num = [&](const char* k) { return std::stod(cfg[k]); };
  const double fs = num("fs"), fc = num("fc"), c = num("c"), alpha = num("alpha");
  const int dec = int(num("decimate"));
  ef::Sos bp, lp;
  for (auto* which : {"bp", "lp"}) {
    auto v = load(path(which));
    auto& s = std::strcmp(which, "bp") == 0 ? bp : lp;
    s.resize(v.size() / 6);
    for (size_t i = 0; i < s.size(); ++i)
      for (size_t j = 0; j < 6; ++j) s[i][j] = v[6 * i + j];
  }
  auto xd = load(path("ping"));
  std::vector<T> x(xd.begin(), xd.end());
  auto rd = load(path("replica"));
  ef::cvec<T> rep(rd.size() / 2);
  for (size_t i = 0; i < rep.size(); ++i) rep[i] = {T(rd[2 * i]), T(rd[2 * i + 1])};
  ef::CfarConfig cc;
  cc.kind = ef::CfarKind(int(num("cfar_kind")));
  cc.n_ref = int(num("n_ref")); cc.n_guard = int(num("n_guard")); cc.stride = int(num("stride"));
  cc.pfa = num("pfa");
  ef::CfarConfig ca = cc;
  ca.kind = ef::CfarKind::CA;
  const double scale = ef::scale_factor(cc), fs_bb = fs / dec;
  auto feats = load(path("features"));
  const int n_cands = int(num("cands_per_ping"));

  // set up once, as the device would: replica spectrum and FFT plan
  const size_t n_bb = (x.size() + size_t(dec) - 1) / size_t(dec);
  ef::MatchedFilter<T> mf(rep, n_bb);
  const ef::Demodulator<T> demod(x.size(), fs, fc, bp, lp, dec);
  const double scale_ca = ef::scale_factor(ca);

  const int n_models = ef_model_count();
  std::vector<std::vector<double>> t_stage(6), t_model(static_cast<size_t>(n_models));
  std::vector<double> t_total;
  size_t n_found = 0;
  volatile double sink = 0;
  for (int it = 0; it < pings; ++it) {
    auto t0 = Clock::now();
    auto iq = demod.run(x);
    auto t1 = Clock::now();
    auto y = mf.run(iq);
    auto t2 = Clock::now();
    auto p = ef::power(y);
    auto thr = ef::cfar_threshold(p, cc, scale);
    auto t3 = Clock::now();
    auto thr_ca = ef::cfar_threshold(p, ca, scale_ca);
    auto t4 = Clock::now();
    auto cands = ef::candidates_1d(p, thr, scale, fs_bb, c, 0.0, num("min_snr_db"));
    auto pt = p;
    ef::apply_tvg(pt, fs_bb, c, 0.0, alpha);
    auto t5 = Clock::now();
    n_found += cands.size();
    sink = sink + double(thr_ca[thr_ca.size() / 2]) + double(pt[pt.size() / 3]);
    auto us = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    t_stage[0].push_back(us(t0, t1));
    t_stage[1].push_back(us(t1, t2));
    t_stage[2].push_back(us(t2, t3));
    t_stage[3].push_back(us(t3, t4));
    t_stage[4].push_back(us(t4, t5));
    double t_cls = 0;
    for (int m = 0; m < n_models; ++m) {
      const int nf = ef_model_n_features(m);
      const size_t rows = feats.size() / 64;  // feature table: R2 order, padded to 64 columns
      std::vector<T> row(static_cast<size_t>(nf));
      auto a = Clock::now();
      for (int k = 0; k < n_cands; ++k) {
        const double* r = &feats[64 * (size_t(k + it) % rows)];
        for (int f = 0; f < nf; ++f) row[size_t(f)] = T(r[f]);
        if constexpr (std::is_same<T, float>::value) { const double v = ef_model_score_f32(m, row.data()); if (std::isfinite(v)) sink = sink + v; }
        else { const double v = ef_model_score_f64(m, row.data()); if (std::isfinite(v)) sink = sink + v; }
      }
      const double dt = us(a, Clock::now());
      t_model[size_t(m)].push_back(dt);
      if (std::string(ef_model_name(m)) == cfg["deploy_model"]) t_cls = dt;
    }
    t_stage[5].push_back(t_cls);
    t_total.push_back(us(t0, t5) + t_cls);
  }
  const char* names[] = {"demodulate", "matched_filter", "power_os_cfar", "ca_cfar_extra",
                         "candidates_tvg", "classifier"};
  std::printf("{\n  \"precision\": \"%s\", \"pings\": %d, \"samples_passband\": %zu, "
              "\"samples_baseband\": %zu, \"cands_per_ping\": %d,\n",
              sizeof(T) == 4 ? "float32" : "float64", pings, x.size(), n_bb, n_cands);
  std::printf("  \"mean_cfar_candidates\": %.2f, \"peak_rss_mb\": %.2f,\n",
              double(n_found) / pings, peak_rss_mb());
  for (int s = 0; s < 6; ++s) {
    auto st = stats(t_stage[size_t(s)]);
    std::printf("  \"%s_us\": [%.1f, %.1f, %.1f],\n", names[s], st.p50, st.p95, st.max);
  }
  for (int m = 0; m < n_models; ++m) {
    auto st = stats(t_model[size_t(m)]);
    std::printf("  \"model_%s_us\": [%.2f, %.2f, %.2f],\n", ef_model_name(m), st.p50, st.p95,
                st.max);
  }
  auto st = stats(t_total);
  std::printf("  \"total_us\": [%.1f, %.1f, %.1f], \"sink\": %g\n}\n", st.p50, st.p95, st.max,
              double(sink));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ef_bench <config.txt> [--pings N] [--core K] [--f64]\n");
    return 2;
  }
  std::string cfg_path = argv[1];
  int pings = 300, core = 2;
  bool f64 = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--pings") && i + 1 < argc) pings = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--core") && i + 1 < argc) core = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--f64")) f64 = true;
  }
  std::map<std::string, std::string> cfg;
  std::ifstream in(cfg_path);
  std::string k, v;
  while (in >> k >> v) cfg[k] = v;
  const auto slash = cfg_path.find_last_of("/\\");
  const std::string base = slash == std::string::npos ? "." : cfg_path.substr(0, slash);
  if (!pin(core)) std::fprintf(stderr, "warning: could not pin to core %d\n", core);
  return f64 ? run<double>(cfg, pings, base) : run<float>(cfg, pings, base);
}
