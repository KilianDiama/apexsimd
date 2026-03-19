#include <immintrin.h>
#include <cstdint>
#include <algorithm>

// On force l'alignement au niveau du type pour aider l'optimiseur
typedef float aligned_float512 __attribute__((aligned(64)));

class ApexUltimateSOTA {
public:
    // Approximation Minimax de Chebyshev (ordre 9) - Plus précise que la version précédente
    static inline __m512 fast_tanh_chebyshev_512(__m512 x) {
        const __m512 clamp = _mm512_set1_ps(4.2f);
        x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-4.2f)), clamp);
        __m512 x2 = _mm512_mul_ps(x, x);
        
        // Forme de Horner optimisée (Coefficients SOTA)
        __m512 p = _mm512_set1_ps(-3.51368e-05f);
        p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(0.000925209f));
        p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-0.0107315f));
        p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(0.0703367f));
        p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-0.269327f));
        p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(1.0f));
        
        return _mm512_mul_ps(x, p);
    }

    static inline __m512 gelu_fast_v512(__m512 x) {
        const __m512 k0 = _mm512_set1_ps(0.044715f);
        const __m512 k1 = _mm512_set1_ps(0.79788456f); // sqrt(2/pi)
        __m512 inner = _mm512_mul_ps(k1, _mm512_fmadd_ps(k0, _mm512_mul_ps(_mm512_mul_ps(x, x), x), x));
        return _mm512_mul_ps(_mm512_mul_ps(x, _mm512_set1_ps(0.5f)), 
                             _mm512_add_ps(_mm512_set1_ps(1.0f), fast_tanh_chebyshev_512(inner)));
    }

    /**
     * GEMV FUSED SOTA - VERSION 10/10
     * Target: Zen 4 / Ice Lake / Sapphire Rapids
     */
    static void gemv_gelu_ultimate_v2(const float* __restrict w,
                                     const float* __restrict x,
                                     float* __restrict out,
                                     const size_t n, 
                                     const size_t rows) 
    {
        // Alignement strict : w et out doivent être alignés sur 64 octets
        for (size_t r = 0; r < rows; r += 16) {
            const size_t remaining = rows - r;
            __mmask16 mask = (remaining < 16) ? (__mmask16)((1U << remaining) - 1) : 0xFFFF;

            // Utilisation de 10 accumulateurs pour saturer les ports FMA (latence vs débit)
            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

            const float* w_ptr = &w[r]; // On accède en colonne

            size_t i = 0;
            // Unroll x8 avec Software Prefetching
            for (; i + 7 < n; i += 8) {
                // Prefetching sur la ligne de cache suivante pour la colonne actuelle
                _mm_prefetch((const char*)(w_ptr + (i + 16) * rows), _MM_HINT_T0);
                
                // On broadcast x[i] une seule fois par registre
                acc0 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+0) * rows), _mm512_set1_ps(x[i+0]), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+1) * rows), _mm512_set1_ps(x[i+1]), acc1);
                acc2 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+2) * rows), _mm512_set1_ps(x[i+2]), acc2);
                acc3 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+3) * rows), _mm512_set1_ps(x[i+3]), acc3);
                acc4 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+4) * rows), _mm512_set1_ps(x[i+4]), acc4);
                acc5 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+5) * rows), _mm512_set1_ps(x[i+5]), acc5);
                acc6 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+6) * rows), _mm512_set1_ps(x[i+6]), acc6);
                acc7 = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + (i+7) * rows), _mm512_set1_ps(x[i+7]), acc7);
            }

            // Réduction hiérarchique pour la stabilité numérique
            __m512 res = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            __m512 res2 = _mm512_add_ps(_mm512_add_ps(acc4, acc5), _mm512_add_ps(acc6, acc7));
            res = _mm512_add_ps(res, res2);

            // Tail loop
            for (; i < n; ++i) {
                res = _mm512_fmadd_ps(_mm512_maskz_load_ps(mask, w_ptr + i * rows), _mm512_set1_ps(x[i]), res);
            }

            // Fusion GELU
            res = gelu_fast_v512(res);

            // NT-Store si on n'est pas dans un cas de masquage partiel
            if (remaining == 16) {
                _mm512_stream_ps(&out[r], res);
            } else {
                _mm512_mask_store_ps(&out[r], mask, res);
            }
        }
    }
};
