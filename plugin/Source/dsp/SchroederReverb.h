#pragma once
#include <array>
#include <algorithm>
#include <cstring>

/**
 * 8 parallel comb filters + 4 series all-pass filters (Freeverb topology).
 * Ported from Python SchroederReverb in app/synth.py (lines 261–360).
 *
 * C++ uses per-sample processing so the numpy sub-block workaround for small
 * all-pass buffers is unnecessary here.
 */
class SchroederReverb
{
public:
    float roomSize = 0.5f;   // 0–1 → feedback 0.70–0.98
    float damp     = 0.5f;   // 0–1 → one-pole lowpass strength
    float wet      = 0.f;    // 0 = dry, 1 = full wet

    void process(float* buf, int n)
    {
        if (wet == 0.f) return;

        const float fb    = 0.7f + roomSize * 0.28f;
        const float damp1 = damp * 0.4f;
        const float damp2 = 1.f - damp1;

        for (int i = 0; i < n; ++i)
        {
            float input = (double)buf[i];
            float wetSample = 0.f;

            // 8 parallel comb filters
            for (int c = 0; c < NUM_COMBS; ++c)
            {
                int   L   = COMB_LENGTHS[c];
                int   pos = combPos[c];
                float old = combBuf[c][pos];

                // 1-pole lowpass on comb output
                combFlt[c] = old * damp2 + combFlt[c] * damp1;
                combBuf[c][pos] = input + combFlt[c] * fb;
                combPos[c] = (pos + 1 >= L) ? 0 : pos + 1;
                wetSample += old;
            }
            wetSample /= NUM_COMBS;

            // 4 series all-pass filters
            constexpr float AP_GAIN = 0.5f;
            for (int a = 0; a < NUM_ALLPASS; ++a)
            {
                int   L   = ALLPASS_LENGTHS[a];
                int   pos = apPos[a];
                float old = apBuf[a][pos];
                apBuf[a][pos] = wetSample + AP_GAIN * old;
                apPos[a] = (pos + 1 >= L) ? 0 : pos + 1;
                wetSample = old - AP_GAIN * wetSample;
            }

            buf[i] = buf[i] * (1.f - wet) + wetSample * wet;
        }
    }

private:
    static constexpr int NUM_COMBS   = 8;
    static constexpr int NUM_ALLPASS = 4;
    static constexpr int COMB_LENGTHS[NUM_COMBS]    = {1116,1188,1277,1356,1422,1491,1557,1617};
    static constexpr int ALLPASS_LENGTHS[NUM_ALLPASS] = {225, 341, 441, 556};
    static constexpr int MAX_COMB_LEN    = 1617;
    static constexpr int MAX_ALLPASS_LEN = 556;

    float combBuf[NUM_COMBS][MAX_COMB_LEN]{};
    float apBuf[NUM_ALLPASS][MAX_ALLPASS_LEN]{};
    int   combPos[NUM_COMBS]{};
    int   apPos[NUM_ALLPASS]{};
    float combFlt[NUM_COMBS]{};
};
