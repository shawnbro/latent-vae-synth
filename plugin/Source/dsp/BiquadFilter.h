#pragma once
#include <algorithm>
#include <cmath>

/**
 * Four-pole resonant lowpass filter (two cascaded biquad stages, 24 dB/oct).
 * Ported from Python BiquadFilter in app/synth.py (lines 196–254).
 *
 * Stage 1: fixed Butterworth Q for flat passband.
 * Stage 2: Q boosted by resonance for a peak near cutoff.
 *
 * Filter state is NOT reset on coefficient changes — allows smooth real-time
 * cutoff modulation without clicks.
 */
class BiquadFilter
{
public:
    BiquadFilter() { updateCoeffs(); }

    void setSampleRate(float sr) { sampleRate = sr; updateCoeffs(); }

    void setCutoff(float hz)
    {
        cutoffHz = std::clamp(hz, 30.f, sampleRate * 0.49f);
        updateCoeffs();
    }

    void setResonance(float r)
    {
        resonance = std::clamp(r, 0.f, 0.95f);
        updateCoeffs();
    }

    /** Clear filter memory. Call on fresh note starts to avoid stale-state clicks. */
    void reset()
    {
        z1_1 = z2_1 = z1_2 = z2_2 = 0.f;
    }

    /** Process block in-place (mono float array). */
    void process(float* buf, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            float s = buf[i];

            // Stage 1 — transposed direct form II (TDFII)
            float y1 = b1[0] * s + z1_1;
            float tmp1 = b1[1] * s - a1[1] * y1 + z2_1;  // new z1 (needs old z2)
            z2_1 = b1[2] * s - a1[2] * y1;
            z1_1 = tmp1;

            // Stage 2 — TDFII
            float y2 = b2[0] * y1 + z1_2;
            float tmp2 = b2[1] * y1 - a2[1] * y2 + z2_2;
            z2_2 = b2[2] * y1 - a2[2] * y2;
            z1_2 = tmp2;

            buf[i] = y2;
        }
    }

private:
    float sampleRate = 44100.f;
    float cutoffHz   = 18000.f;
    float resonance  = 0.f;

    // Biquad coefficients: b[0..2], a[0..2] (a[0] = 1 after normalisation)
    float b1[3]{}, a1[3]{};
    float b2[3]{}, a2[3]{};

    // Transposed direct form II state (2 delays per stage)
    float z1_1 = 0.f, z2_1 = 0.f;
    float z1_2 = 0.f, z2_2 = 0.f;

    /** Compute RBJ lowpass biquad coefficients for a given Q. */
    void biquadLP(float Q, float* b_out, float* a_out) const
    {
        double w0    = 2.0 * 3.141592653589793 * cutoffHz / sampleRate;
        double cosW0 = std::cos(w0);
        double alpha = std::sin(w0) / (2.0 * Q);
        double b0  = (1.0 - cosW0) * 0.5;
        double b1v =  1.0 - cosW0;
        double b2v = (1.0 - cosW0) * 0.5;
        double a0  =  1.0 + alpha;
        double a1v = -2.0 * cosW0;
        double a2v =  1.0 - alpha;
        b_out[0] = (float)(b0  / a0);
        b_out[1] = (float)(b1v / a0);
        b_out[2] = (float)(b2v / a0);
        a_out[0] = 1.f;
        a_out[1] = (float)(a1v / a0);
        a_out[2] = (float)(a2v / a0);
    }

    void updateCoeffs()
    {
        // 4th-order Butterworth pole-pair Q values (matches Python)
        float Q1 = 0.5412139f;
        float Q2 = 1.3065630f + resonance * 10.f;
        biquadLP(Q1, b1, a1);
        biquadLP(Q2, b2, a2);
        // State not touched — preserves smooth modulation
    }
};
