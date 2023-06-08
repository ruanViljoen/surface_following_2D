/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

int BAbt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int BAbt_alloc_mem(void);
int BAbt_init_mem(int mem);
void BAbt_free_mem(int mem);
int BAbt_checkout(void);
void BAbt_release(int mem);
void BAbt_incref(void);
void BAbt_decref(void);
casadi_int BAbt_n_in(void);
casadi_int BAbt_n_out(void);
casadi_real BAbt_default_in(casadi_int i);
const char* BAbt_name_in(casadi_int i);
const char* BAbt_name_out(casadi_int i);
const casadi_int* BAbt_sparsity_in(casadi_int i);
const casadi_int* BAbt_sparsity_out(casadi_int i);
int BAbt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define BAbt_SZ_ARG 5
#define BAbt_SZ_RES 1
#define BAbt_SZ_IW 0
#define BAbt_SZ_W 18
int bk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int bk_alloc_mem(void);
int bk_init_mem(int mem);
void bk_free_mem(int mem);
int bk_checkout(void);
void bk_release(int mem);
void bk_incref(void);
void bk_decref(void);
casadi_int bk_n_in(void);
casadi_int bk_n_out(void);
casadi_real bk_default_in(casadi_int i);
const char* bk_name_in(casadi_int i);
const char* bk_name_out(casadi_int i);
const casadi_int* bk_sparsity_in(casadi_int i);
const casadi_int* bk_sparsity_out(casadi_int i);
int bk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define bk_SZ_ARG 5
#define bk_SZ_RES 1
#define bk_SZ_IW 0
#define bk_SZ_W 15
int RSQrqtI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqtI_alloc_mem(void);
int RSQrqtI_init_mem(int mem);
void RSQrqtI_free_mem(int mem);
int RSQrqtI_checkout(void);
void RSQrqtI_release(int mem);
void RSQrqtI_incref(void);
void RSQrqtI_decref(void);
casadi_int RSQrqtI_n_in(void);
casadi_int RSQrqtI_n_out(void);
casadi_real RSQrqtI_default_in(casadi_int i);
const char* RSQrqtI_name_in(casadi_int i);
const char* RSQrqtI_name_out(casadi_int i);
const casadi_int* RSQrqtI_sparsity_in(casadi_int i);
const casadi_int* RSQrqtI_sparsity_out(casadi_int i);
int RSQrqtI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqtI_SZ_ARG 8
#define RSQrqtI_SZ_RES 1
#define RSQrqtI_SZ_IW 0
#define RSQrqtI_SZ_W 1973
int RSQrqtIGN(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqtIGN_alloc_mem(void);
int RSQrqtIGN_init_mem(int mem);
void RSQrqtIGN_free_mem(int mem);
int RSQrqtIGN_checkout(void);
void RSQrqtIGN_release(int mem);
void RSQrqtIGN_incref(void);
void RSQrqtIGN_decref(void);
casadi_int RSQrqtIGN_n_in(void);
casadi_int RSQrqtIGN_n_out(void);
casadi_real RSQrqtIGN_default_in(casadi_int i);
const char* RSQrqtIGN_name_in(casadi_int i);
const char* RSQrqtIGN_name_out(casadi_int i);
const casadi_int* RSQrqtIGN_sparsity_in(casadi_int i);
const casadi_int* RSQrqtIGN_sparsity_out(casadi_int i);
int RSQrqtIGN_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqtIGN_SZ_ARG 8
#define RSQrqtIGN_SZ_RES 1
#define RSQrqtIGN_SZ_IW 0
#define RSQrqtIGN_SZ_W 1448
int rqI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int rqI_alloc_mem(void);
int rqI_init_mem(int mem);
void rqI_free_mem(int mem);
int rqI_checkout(void);
void rqI_release(int mem);
void rqI_incref(void);
void rqI_decref(void);
casadi_int rqI_n_in(void);
casadi_int rqI_n_out(void);
casadi_real rqI_default_in(casadi_int i);
const char* rqI_name_in(casadi_int i);
const char* rqI_name_out(casadi_int i);
const casadi_int* rqI_sparsity_in(casadi_int i);
const casadi_int* rqI_sparsity_out(casadi_int i);
int rqI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define rqI_SZ_ARG 5
#define rqI_SZ_RES 1
#define rqI_SZ_IW 0
#define rqI_SZ_W 340
int RSQrqt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqt_alloc_mem(void);
int RSQrqt_init_mem(int mem);
void RSQrqt_free_mem(int mem);
int RSQrqt_checkout(void);
void RSQrqt_release(int mem);
void RSQrqt_incref(void);
void RSQrqt_decref(void);
casadi_int RSQrqt_n_in(void);
casadi_int RSQrqt_n_out(void);
casadi_real RSQrqt_default_in(casadi_int i);
const char* RSQrqt_name_in(casadi_int i);
const char* RSQrqt_name_out(casadi_int i);
const casadi_int* RSQrqt_sparsity_in(casadi_int i);
const casadi_int* RSQrqt_sparsity_out(casadi_int i);
int RSQrqt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqt_SZ_ARG 8
#define RSQrqt_SZ_RES 1
#define RSQrqt_SZ_IW 0
#define RSQrqt_SZ_W 1973
int RSQrqtGN(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqtGN_alloc_mem(void);
int RSQrqtGN_init_mem(int mem);
void RSQrqtGN_free_mem(int mem);
int RSQrqtGN_checkout(void);
void RSQrqtGN_release(int mem);
void RSQrqtGN_incref(void);
void RSQrqtGN_decref(void);
casadi_int RSQrqtGN_n_in(void);
casadi_int RSQrqtGN_n_out(void);
casadi_real RSQrqtGN_default_in(casadi_int i);
const char* RSQrqtGN_name_in(casadi_int i);
const char* RSQrqtGN_name_out(casadi_int i);
const casadi_int* RSQrqtGN_sparsity_in(casadi_int i);
const casadi_int* RSQrqtGN_sparsity_out(casadi_int i);
int RSQrqtGN_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqtGN_SZ_ARG 8
#define RSQrqtGN_SZ_RES 1
#define RSQrqtGN_SZ_IW 0
#define RSQrqtGN_SZ_W 1448
int rqk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int rqk_alloc_mem(void);
int rqk_init_mem(int mem);
void rqk_free_mem(int mem);
int rqk_checkout(void);
void rqk_release(int mem);
void rqk_incref(void);
void rqk_decref(void);
casadi_int rqk_n_in(void);
casadi_int rqk_n_out(void);
casadi_real rqk_default_in(casadi_int i);
const char* rqk_name_in(casadi_int i);
const char* rqk_name_out(casadi_int i);
const casadi_int* rqk_sparsity_in(casadi_int i);
const casadi_int* rqk_sparsity_out(casadi_int i);
int rqk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define rqk_SZ_ARG 5
#define rqk_SZ_RES 1
#define rqk_SZ_IW 0
#define rqk_SZ_W 340
int LI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int LI_alloc_mem(void);
int LI_init_mem(int mem);
void LI_free_mem(int mem);
int LI_checkout(void);
void LI_release(int mem);
void LI_incref(void);
void LI_decref(void);
casadi_int LI_n_in(void);
casadi_int LI_n_out(void);
casadi_real LI_default_in(casadi_int i);
const char* LI_name_in(casadi_int i);
const char* LI_name_out(casadi_int i);
const casadi_int* LI_sparsity_in(casadi_int i);
const casadi_int* LI_sparsity_out(casadi_int i);
int LI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define LI_SZ_ARG 5
#define LI_SZ_RES 1
#define LI_SZ_IW 0
#define LI_SZ_W 97
int Lk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int Lk_alloc_mem(void);
int Lk_init_mem(int mem);
void Lk_free_mem(int mem);
int Lk_checkout(void);
void Lk_release(int mem);
void Lk_incref(void);
void Lk_decref(void);
casadi_int Lk_n_in(void);
casadi_int Lk_n_out(void);
casadi_real Lk_default_in(casadi_int i);
const char* Lk_name_in(casadi_int i);
const char* Lk_name_out(casadi_int i);
const casadi_int* Lk_sparsity_in(casadi_int i);
const casadi_int* Lk_sparsity_out(casadi_int i);
int Lk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define Lk_SZ_ARG 5
#define Lk_SZ_RES 1
#define Lk_SZ_IW 0
#define Lk_SZ_W 97
int RSQrqtF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqtF_alloc_mem(void);
int RSQrqtF_init_mem(int mem);
void RSQrqtF_free_mem(int mem);
int RSQrqtF_checkout(void);
void RSQrqtF_release(int mem);
void RSQrqtF_incref(void);
void RSQrqtF_decref(void);
casadi_int RSQrqtF_n_in(void);
casadi_int RSQrqtF_n_out(void);
casadi_real RSQrqtF_default_in(casadi_int i);
const char* RSQrqtF_name_in(casadi_int i);
const char* RSQrqtF_name_out(casadi_int i);
const casadi_int* RSQrqtF_sparsity_in(casadi_int i);
const casadi_int* RSQrqtF_sparsity_out(casadi_int i);
int RSQrqtF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqtF_SZ_ARG 8
#define RSQrqtF_SZ_RES 1
#define RSQrqtF_SZ_IW 0
#define RSQrqtF_SZ_W 1
int RSQrqtFGN(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int RSQrqtFGN_alloc_mem(void);
int RSQrqtFGN_init_mem(int mem);
void RSQrqtFGN_free_mem(int mem);
int RSQrqtFGN_checkout(void);
void RSQrqtFGN_release(int mem);
void RSQrqtFGN_incref(void);
void RSQrqtFGN_decref(void);
casadi_int RSQrqtFGN_n_in(void);
casadi_int RSQrqtFGN_n_out(void);
casadi_real RSQrqtFGN_default_in(casadi_int i);
const char* RSQrqtFGN_name_in(casadi_int i);
const char* RSQrqtFGN_name_out(casadi_int i);
const casadi_int* RSQrqtFGN_sparsity_in(casadi_int i);
const casadi_int* RSQrqtFGN_sparsity_out(casadi_int i);
int RSQrqtFGN_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define RSQrqtFGN_SZ_ARG 8
#define RSQrqtFGN_SZ_RES 1
#define RSQrqtFGN_SZ_IW 0
#define RSQrqtFGN_SZ_W 1
int rqF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int rqF_alloc_mem(void);
int rqF_init_mem(int mem);
void rqF_free_mem(int mem);
int rqF_checkout(void);
void rqF_release(int mem);
void rqF_incref(void);
void rqF_decref(void);
casadi_int rqF_n_in(void);
casadi_int rqF_n_out(void);
casadi_real rqF_default_in(casadi_int i);
const char* rqF_name_in(casadi_int i);
const char* rqF_name_out(casadi_int i);
const casadi_int* rqF_sparsity_in(casadi_int i);
const casadi_int* rqF_sparsity_out(casadi_int i);
int rqF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define rqF_SZ_ARG 5
#define rqF_SZ_RES 1
#define rqF_SZ_IW 0
#define rqF_SZ_W 1
int LF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int LF_alloc_mem(void);
int LF_init_mem(int mem);
void LF_free_mem(int mem);
int LF_checkout(void);
void LF_release(int mem);
void LF_incref(void);
void LF_decref(void);
casadi_int LF_n_in(void);
casadi_int LF_n_out(void);
casadi_real LF_default_in(casadi_int i);
const char* LF_name_in(casadi_int i);
const char* LF_name_out(casadi_int i);
const casadi_int* LF_sparsity_in(casadi_int i);
const casadi_int* LF_sparsity_out(casadi_int i);
int LF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define LF_SZ_ARG 5
#define LF_SZ_RES 1
#define LF_SZ_IW 0
#define LF_SZ_W 1
int GgtI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int GgtI_alloc_mem(void);
int GgtI_init_mem(int mem);
void GgtI_free_mem(int mem);
int GgtI_checkout(void);
void GgtI_release(int mem);
void GgtI_incref(void);
void GgtI_decref(void);
casadi_int GgtI_n_in(void);
casadi_int GgtI_n_out(void);
casadi_real GgtI_default_in(casadi_int i);
const char* GgtI_name_in(casadi_int i);
const char* GgtI_name_out(casadi_int i);
const casadi_int* GgtI_sparsity_in(casadi_int i);
const casadi_int* GgtI_sparsity_out(casadi_int i);
int GgtI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define GgtI_SZ_ARG 4
#define GgtI_SZ_RES 1
#define GgtI_SZ_IW 0
#define GgtI_SZ_W 4
int gI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gI_alloc_mem(void);
int gI_init_mem(int mem);
void gI_free_mem(int mem);
int gI_checkout(void);
void gI_release(int mem);
void gI_incref(void);
void gI_decref(void);
casadi_int gI_n_in(void);
casadi_int gI_n_out(void);
casadi_real gI_default_in(casadi_int i);
const char* gI_name_in(casadi_int i);
const char* gI_name_out(casadi_int i);
const casadi_int* gI_sparsity_in(casadi_int i);
const casadi_int* gI_sparsity_out(casadi_int i);
int gI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gI_SZ_ARG 4
#define gI_SZ_RES 1
#define gI_SZ_IW 0
#define gI_SZ_W 2
int Ggt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int Ggt_alloc_mem(void);
int Ggt_init_mem(int mem);
void Ggt_free_mem(int mem);
int Ggt_checkout(void);
void Ggt_release(int mem);
void Ggt_incref(void);
void Ggt_decref(void);
casadi_int Ggt_n_in(void);
casadi_int Ggt_n_out(void);
casadi_real Ggt_default_in(casadi_int i);
const char* Ggt_name_in(casadi_int i);
const char* Ggt_name_out(casadi_int i);
const casadi_int* Ggt_sparsity_in(casadi_int i);
const casadi_int* Ggt_sparsity_out(casadi_int i);
int Ggt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define Ggt_SZ_ARG 4
#define Ggt_SZ_RES 1
#define Ggt_SZ_IW 0
#define Ggt_SZ_W 0
int g(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int g_alloc_mem(void);
int g_init_mem(int mem);
void g_free_mem(int mem);
int g_checkout(void);
void g_release(int mem);
void g_incref(void);
void g_decref(void);
casadi_int g_n_in(void);
casadi_int g_n_out(void);
casadi_real g_default_in(casadi_int i);
const char* g_name_in(casadi_int i);
const char* g_name_out(casadi_int i);
const casadi_int* g_sparsity_in(casadi_int i);
const casadi_int* g_sparsity_out(casadi_int i);
int g_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define g_SZ_ARG 4
#define g_SZ_RES 1
#define g_SZ_IW 0
#define g_SZ_W 0
int GgtF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int GgtF_alloc_mem(void);
int GgtF_init_mem(int mem);
void GgtF_free_mem(int mem);
int GgtF_checkout(void);
void GgtF_release(int mem);
void GgtF_incref(void);
void GgtF_decref(void);
casadi_int GgtF_n_in(void);
casadi_int GgtF_n_out(void);
casadi_real GgtF_default_in(casadi_int i);
const char* GgtF_name_in(casadi_int i);
const char* GgtF_name_out(casadi_int i);
const casadi_int* GgtF_sparsity_in(casadi_int i);
const casadi_int* GgtF_sparsity_out(casadi_int i);
int GgtF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define GgtF_SZ_ARG 4
#define GgtF_SZ_RES 1
#define GgtF_SZ_IW 0
#define GgtF_SZ_W 0
int gF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gF_alloc_mem(void);
int gF_init_mem(int mem);
void gF_free_mem(int mem);
int gF_checkout(void);
void gF_release(int mem);
void gF_incref(void);
void gF_decref(void);
casadi_int gF_n_in(void);
casadi_int gF_n_out(void);
casadi_real gF_default_in(casadi_int i);
const char* gF_name_in(casadi_int i);
const char* gF_name_out(casadi_int i);
const casadi_int* gF_sparsity_in(casadi_int i);
const casadi_int* gF_sparsity_out(casadi_int i);
int gF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gF_SZ_ARG 4
#define gF_SZ_RES 1
#define gF_SZ_IW 0
#define gF_SZ_W 0
int GgineqIt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int GgineqIt_alloc_mem(void);
int GgineqIt_init_mem(int mem);
void GgineqIt_free_mem(int mem);
int GgineqIt_checkout(void);
void GgineqIt_release(int mem);
void GgineqIt_incref(void);
void GgineqIt_decref(void);
casadi_int GgineqIt_n_in(void);
casadi_int GgineqIt_n_out(void);
casadi_real GgineqIt_default_in(casadi_int i);
const char* GgineqIt_name_in(casadi_int i);
const char* GgineqIt_name_out(casadi_int i);
const casadi_int* GgineqIt_sparsity_in(casadi_int i);
const casadi_int* GgineqIt_sparsity_out(casadi_int i);
int GgineqIt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define GgineqIt_SZ_ARG 4
#define GgineqIt_SZ_RES 1
#define GgineqIt_SZ_IW 0
#define GgineqIt_SZ_W 306
int gineqI(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gineqI_alloc_mem(void);
int gineqI_init_mem(int mem);
void gineqI_free_mem(int mem);
int gineqI_checkout(void);
void gineqI_release(int mem);
void gineqI_incref(void);
void gineqI_decref(void);
casadi_int gineqI_n_in(void);
casadi_int gineqI_n_out(void);
casadi_real gineqI_default_in(casadi_int i);
const char* gineqI_name_in(casadi_int i);
const char* gineqI_name_out(casadi_int i);
const casadi_int* gineqI_sparsity_in(casadi_int i);
const casadi_int* gineqI_sparsity_out(casadi_int i);
int gineqI_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gineqI_SZ_ARG 4
#define gineqI_SZ_RES 1
#define gineqI_SZ_IW 0
#define gineqI_SZ_W 76
int Ggineqt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int Ggineqt_alloc_mem(void);
int Ggineqt_init_mem(int mem);
void Ggineqt_free_mem(int mem);
int Ggineqt_checkout(void);
void Ggineqt_release(int mem);
void Ggineqt_incref(void);
void Ggineqt_decref(void);
casadi_int Ggineqt_n_in(void);
casadi_int Ggineqt_n_out(void);
casadi_real Ggineqt_default_in(casadi_int i);
const char* Ggineqt_name_in(casadi_int i);
const char* Ggineqt_name_out(casadi_int i);
const casadi_int* Ggineqt_sparsity_in(casadi_int i);
const casadi_int* Ggineqt_sparsity_out(casadi_int i);
int Ggineqt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define Ggineqt_SZ_ARG 4
#define Ggineqt_SZ_RES 1
#define Ggineqt_SZ_IW 0
#define Ggineqt_SZ_W 306
int gineq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gineq_alloc_mem(void);
int gineq_init_mem(int mem);
void gineq_free_mem(int mem);
int gineq_checkout(void);
void gineq_release(int mem);
void gineq_incref(void);
void gineq_decref(void);
casadi_int gineq_n_in(void);
casadi_int gineq_n_out(void);
casadi_real gineq_default_in(casadi_int i);
const char* gineq_name_in(casadi_int i);
const char* gineq_name_out(casadi_int i);
const casadi_int* gineq_sparsity_in(casadi_int i);
const casadi_int* gineq_sparsity_out(casadi_int i);
int gineq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gineq_SZ_ARG 4
#define gineq_SZ_RES 1
#define gineq_SZ_IW 0
#define gineq_SZ_W 76
int GgineqFt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int GgineqFt_alloc_mem(void);
int GgineqFt_init_mem(int mem);
void GgineqFt_free_mem(int mem);
int GgineqFt_checkout(void);
void GgineqFt_release(int mem);
void GgineqFt_incref(void);
void GgineqFt_decref(void);
casadi_int GgineqFt_n_in(void);
casadi_int GgineqFt_n_out(void);
casadi_real GgineqFt_default_in(casadi_int i);
const char* GgineqFt_name_in(casadi_int i);
const char* GgineqFt_name_out(casadi_int i);
const casadi_int* GgineqFt_sparsity_in(casadi_int i);
const casadi_int* GgineqFt_sparsity_out(casadi_int i);
int GgineqFt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define GgineqFt_SZ_ARG 3
#define GgineqFt_SZ_RES 1
#define GgineqFt_SZ_IW 0
#define GgineqFt_SZ_W 0
int gineqF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gineqF_alloc_mem(void);
int gineqF_init_mem(int mem);
void gineqF_free_mem(int mem);
int gineqF_checkout(void);
void gineqF_release(int mem);
void gineqF_incref(void);
void gineqF_decref(void);
casadi_int gineqF_n_in(void);
casadi_int gineqF_n_out(void);
casadi_real gineqF_default_in(casadi_int i);
const char* gineqF_name_in(casadi_int i);
const char* gineqF_name_out(casadi_int i);
const casadi_int* gineqF_sparsity_in(casadi_int i);
const casadi_int* gineqF_sparsity_out(casadi_int i);
int gineqF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gineqF_SZ_ARG 3
#define gineqF_SZ_RES 1
#define gineqF_SZ_IW 0
#define gineqF_SZ_W 0
int sampler_q(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_q_alloc_mem(void);
int sampler_q_init_mem(int mem);
void sampler_q_free_mem(int mem);
int sampler_q_checkout(void);
void sampler_q_release(int mem);
void sampler_q_incref(void);
void sampler_q_decref(void);
casadi_int sampler_q_n_in(void);
casadi_int sampler_q_n_out(void);
casadi_real sampler_q_default_in(casadi_int i);
const char* sampler_q_name_in(casadi_int i);
const char* sampler_q_name_out(casadi_int i);
const casadi_int* sampler_q_sparsity_in(casadi_int i);
const casadi_int* sampler_q_sparsity_out(casadi_int i);
int sampler_q_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_q_SZ_ARG 4
#define sampler_q_SZ_RES 1
#define sampler_q_SZ_IW 0
#define sampler_q_SZ_W 1
int sampler_x_MPC_window(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_x_MPC_window_alloc_mem(void);
int sampler_x_MPC_window_init_mem(int mem);
void sampler_x_MPC_window_free_mem(int mem);
int sampler_x_MPC_window_checkout(void);
void sampler_x_MPC_window_release(int mem);
void sampler_x_MPC_window_incref(void);
void sampler_x_MPC_window_decref(void);
casadi_int sampler_x_MPC_window_n_in(void);
casadi_int sampler_x_MPC_window_n_out(void);
casadi_real sampler_x_MPC_window_default_in(casadi_int i);
const char* sampler_x_MPC_window_name_in(casadi_int i);
const char* sampler_x_MPC_window_name_out(casadi_int i);
const casadi_int* sampler_x_MPC_window_sparsity_in(casadi_int i);
const casadi_int* sampler_x_MPC_window_sparsity_out(casadi_int i);
int sampler_x_MPC_window_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_x_MPC_window_SZ_ARG 4
#define sampler_x_MPC_window_SZ_RES 1
#define sampler_x_MPC_window_SZ_IW 0
#define sampler_x_MPC_window_SZ_W 1
int sampler_dq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_dq_alloc_mem(void);
int sampler_dq_init_mem(int mem);
void sampler_dq_free_mem(int mem);
int sampler_dq_checkout(void);
void sampler_dq_release(int mem);
void sampler_dq_incref(void);
void sampler_dq_decref(void);
casadi_int sampler_dq_n_in(void);
casadi_int sampler_dq_n_out(void);
casadi_real sampler_dq_default_in(casadi_int i);
const char* sampler_dq_name_in(casadi_int i);
const char* sampler_dq_name_out(casadi_int i);
const casadi_int* sampler_dq_sparsity_in(casadi_int i);
const casadi_int* sampler_dq_sparsity_out(casadi_int i);
int sampler_dq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_dq_SZ_ARG 4
#define sampler_dq_SZ_RES 1
#define sampler_dq_SZ_IW 0
#define sampler_dq_SZ_W 1
int sampler_ddq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_ddq_alloc_mem(void);
int sampler_ddq_init_mem(int mem);
void sampler_ddq_free_mem(int mem);
int sampler_ddq_checkout(void);
void sampler_ddq_release(int mem);
void sampler_ddq_incref(void);
void sampler_ddq_decref(void);
casadi_int sampler_ddq_n_in(void);
casadi_int sampler_ddq_n_out(void);
casadi_real sampler_ddq_default_in(casadi_int i);
const char* sampler_ddq_name_in(casadi_int i);
const char* sampler_ddq_name_out(casadi_int i);
const casadi_int* sampler_ddq_sparsity_in(casadi_int i);
const casadi_int* sampler_ddq_sparsity_out(casadi_int i);
int sampler_ddq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_ddq_SZ_ARG 4
#define sampler_ddq_SZ_RES 1
#define sampler_ddq_SZ_IW 0
#define sampler_ddq_SZ_W 1
int sampler_w(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_w_alloc_mem(void);
int sampler_w_init_mem(int mem);
void sampler_w_free_mem(int mem);
int sampler_w_checkout(void);
void sampler_w_release(int mem);
void sampler_w_incref(void);
void sampler_w_decref(void);
casadi_int sampler_w_n_in(void);
casadi_int sampler_w_n_out(void);
casadi_real sampler_w_default_in(casadi_int i);
const char* sampler_w_name_in(casadi_int i);
const char* sampler_w_name_out(casadi_int i);
const casadi_int* sampler_w_sparsity_in(casadi_int i);
const casadi_int* sampler_w_sparsity_out(casadi_int i);
int sampler_w_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_w_SZ_ARG 4
#define sampler_w_SZ_RES 1
#define sampler_w_SZ_IW 0
#define sampler_w_SZ_W 1
int sampler_x(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_x_alloc_mem(void);
int sampler_x_init_mem(int mem);
void sampler_x_free_mem(int mem);
int sampler_x_checkout(void);
void sampler_x_release(int mem);
void sampler_x_incref(void);
void sampler_x_decref(void);
casadi_int sampler_x_n_in(void);
casadi_int sampler_x_n_out(void);
casadi_real sampler_x_default_in(casadi_int i);
const char* sampler_x_name_in(casadi_int i);
const char* sampler_x_name_out(casadi_int i);
const casadi_int* sampler_x_sparsity_in(casadi_int i);
const casadi_int* sampler_x_sparsity_out(casadi_int i);
int sampler_x_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_x_SZ_ARG 4
#define sampler_x_SZ_RES 1
#define sampler_x_SZ_IW 0
#define sampler_x_SZ_W 1
int sampler_dx(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_dx_alloc_mem(void);
int sampler_dx_init_mem(int mem);
void sampler_dx_free_mem(int mem);
int sampler_dx_checkout(void);
void sampler_dx_release(int mem);
void sampler_dx_incref(void);
void sampler_dx_decref(void);
casadi_int sampler_dx_n_in(void);
casadi_int sampler_dx_n_out(void);
casadi_real sampler_dx_default_in(casadi_int i);
const char* sampler_dx_name_in(casadi_int i);
const char* sampler_dx_name_out(casadi_int i);
const casadi_int* sampler_dx_sparsity_in(casadi_int i);
const casadi_int* sampler_dx_sparsity_out(casadi_int i);
int sampler_dx_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_dx_SZ_ARG 4
#define sampler_dx_SZ_RES 1
#define sampler_dx_SZ_IW 0
#define sampler_dx_SZ_W 1
int sampler_ddx(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_ddx_alloc_mem(void);
int sampler_ddx_init_mem(int mem);
void sampler_ddx_free_mem(int mem);
int sampler_ddx_checkout(void);
void sampler_ddx_release(int mem);
void sampler_ddx_incref(void);
void sampler_ddx_decref(void);
casadi_int sampler_ddx_n_in(void);
casadi_int sampler_ddx_n_out(void);
casadi_real sampler_ddx_default_in(casadi_int i);
const char* sampler_ddx_name_in(casadi_int i);
const char* sampler_ddx_name_out(casadi_int i);
const casadi_int* sampler_ddx_sparsity_in(casadi_int i);
const casadi_int* sampler_ddx_sparsity_out(casadi_int i);
int sampler_ddx_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_ddx_SZ_ARG 4
#define sampler_ddx_SZ_RES 1
#define sampler_ddx_SZ_IW 0
#define sampler_ddx_SZ_W 1
int sampler_path(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_path_alloc_mem(void);
int sampler_path_init_mem(int mem);
void sampler_path_free_mem(int mem);
int sampler_path_checkout(void);
void sampler_path_release(int mem);
void sampler_path_incref(void);
void sampler_path_decref(void);
casadi_int sampler_path_n_in(void);
casadi_int sampler_path_n_out(void);
casadi_real sampler_path_default_in(casadi_int i);
const char* sampler_path_name_in(casadi_int i);
const char* sampler_path_name_out(casadi_int i);
const casadi_int* sampler_path_sparsity_in(casadi_int i);
const casadi_int* sampler_path_sparsity_out(casadi_int i);
int sampler_path_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_path_SZ_ARG 4
#define sampler_path_SZ_RES 1
#define sampler_path_SZ_IW 0
#define sampler_path_SZ_W 25
int sampler_p_w_tcp(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_p_w_tcp_alloc_mem(void);
int sampler_p_w_tcp_init_mem(int mem);
void sampler_p_w_tcp_free_mem(int mem);
int sampler_p_w_tcp_checkout(void);
void sampler_p_w_tcp_release(int mem);
void sampler_p_w_tcp_incref(void);
void sampler_p_w_tcp_decref(void);
casadi_int sampler_p_w_tcp_n_in(void);
casadi_int sampler_p_w_tcp_n_out(void);
casadi_real sampler_p_w_tcp_default_in(casadi_int i);
const char* sampler_p_w_tcp_name_in(casadi_int i);
const char* sampler_p_w_tcp_name_out(casadi_int i);
const casadi_int* sampler_p_w_tcp_sparsity_in(casadi_int i);
const casadi_int* sampler_p_w_tcp_sparsity_out(casadi_int i);
int sampler_p_w_tcp_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_p_w_tcp_SZ_ARG 4
#define sampler_p_w_tcp_SZ_RES 1
#define sampler_p_w_tcp_SZ_IW 0
#define sampler_p_w_tcp_SZ_W 13
int sampler_task_translation_error(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_task_translation_error_alloc_mem(void);
int sampler_task_translation_error_init_mem(int mem);
void sampler_task_translation_error_free_mem(int mem);
int sampler_task_translation_error_checkout(void);
void sampler_task_translation_error_release(int mem);
void sampler_task_translation_error_incref(void);
void sampler_task_translation_error_decref(void);
casadi_int sampler_task_translation_error_n_in(void);
casadi_int sampler_task_translation_error_n_out(void);
casadi_real sampler_task_translation_error_default_in(casadi_int i);
const char* sampler_task_translation_error_name_in(casadi_int i);
const char* sampler_task_translation_error_name_out(casadi_int i);
const casadi_int* sampler_task_translation_error_sparsity_in(casadi_int i);
const casadi_int* sampler_task_translation_error_sparsity_out(casadi_int i);
int sampler_task_translation_error_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_task_translation_error_SZ_ARG 4
#define sampler_task_translation_error_SZ_RES 1
#define sampler_task_translation_error_SZ_IW 0
#define sampler_task_translation_error_SZ_W 36
int sampler_task_orientation_error(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_task_orientation_error_alloc_mem(void);
int sampler_task_orientation_error_init_mem(int mem);
void sampler_task_orientation_error_free_mem(int mem);
int sampler_task_orientation_error_checkout(void);
void sampler_task_orientation_error_release(int mem);
void sampler_task_orientation_error_incref(void);
void sampler_task_orientation_error_decref(void);
casadi_int sampler_task_orientation_error_n_in(void);
casadi_int sampler_task_orientation_error_n_out(void);
casadi_real sampler_task_orientation_error_default_in(casadi_int i);
const char* sampler_task_orientation_error_name_in(casadi_int i);
const char* sampler_task_orientation_error_name_out(casadi_int i);
const casadi_int* sampler_task_orientation_error_sparsity_in(casadi_int i);
const casadi_int* sampler_task_orientation_error_sparsity_out(casadi_int i);
int sampler_task_orientation_error_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_task_orientation_error_SZ_ARG 4
#define sampler_task_orientation_error_SZ_RES 1
#define sampler_task_orientation_error_SZ_IW 0
#define sampler_task_orientation_error_SZ_W 11
int sampler_task_progress_speed_error(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int sampler_task_progress_speed_error_alloc_mem(void);
int sampler_task_progress_speed_error_init_mem(int mem);
void sampler_task_progress_speed_error_free_mem(int mem);
int sampler_task_progress_speed_error_checkout(void);
void sampler_task_progress_speed_error_release(int mem);
void sampler_task_progress_speed_error_incref(void);
void sampler_task_progress_speed_error_decref(void);
casadi_int sampler_task_progress_speed_error_n_in(void);
casadi_int sampler_task_progress_speed_error_n_out(void);
casadi_real sampler_task_progress_speed_error_default_in(casadi_int i);
const char* sampler_task_progress_speed_error_name_in(casadi_int i);
const char* sampler_task_progress_speed_error_name_out(casadi_int i);
const casadi_int* sampler_task_progress_speed_error_sparsity_in(casadi_int i);
const casadi_int* sampler_task_progress_speed_error_sparsity_out(casadi_int i);
int sampler_task_progress_speed_error_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define sampler_task_progress_speed_error_SZ_ARG 4
#define sampler_task_progress_speed_error_SZ_RES 1
#define sampler_task_progress_speed_error_SZ_IW 0
#define sampler_task_progress_speed_error_SZ_W 8
#ifdef __cplusplus
} /* extern "C" */
#endif