/* C ABI of the EchoFind edge library (see src/capi.cpp) and of the generated classifiers
 * (generated/models.c). Complex arrays are interleaved (re, im). */
#ifndef ECHOFIND_CAPI_H
#define ECHOFIND_CAPI_H

#if defined(_WIN32)
#define EF_API __declspec(dllexport)
#else
#define EF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

EF_API int ef_demodulate_f64(const double* x, int n, double fs, double fc, const double* bp,
                             int nbp, const double* lp, int nlp, int dec, double t0, double* out);
EF_API int ef_demodulate_f32(const float* x, int n, double fs, double fc, const double* bp,
                             int nbp, const double* lp, int nlp, int dec, double t0, float* out);
EF_API int ef_matched_filter_f64(const double* x, int n, const double* rep, int m, double* out);
EF_API int ef_matched_filter_f32(const float* x, int n, const float* rep, int m, float* out);
EF_API void ef_tvg_f64(double* p, int n, double fs, double c, double t0, double alpha,
                       double spreading, double r_min);
EF_API double ef_scale_factor(int kind, int n_ref, int k, double pfa);
EF_API int ef_cfar_threshold_f64(const double* p, int n, int kind, int n_ref, int n_guard, int k,
                                 int stride, double pfa, double* thr);
EF_API int ef_cfar_threshold_f32(const float* p, int n, int kind, int n_ref, int n_guard, int k,
                                 int stride, double pfa, float* thr);

/* Generated classifiers: raw score (log-odds) from one candidate's feature vector, in the
 * feature order of generated/models.json. */
EF_API int ef_model_count(void);
EF_API const char* ef_model_name(int i);
EF_API int ef_model_n_features(int i);
EF_API double ef_model_score_f64(int i, const double* x);
EF_API float ef_model_score_f32(int i, const float* x);

#ifdef __cplusplus
}
#endif
#endif
