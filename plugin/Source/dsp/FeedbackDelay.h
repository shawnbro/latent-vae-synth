#pragma once
#include <vector>
#include <algorithm>
#include <cstring>

/**
 * Single circular-buffer feedback delay.
 * Ported from Python FeedbackDelay in app/synth.py (lines 367–410).
 */
class FeedbackDelay
{
public:
    float timeMs   = 250.f;  // 20–1000 ms
    float feedback = 0.4f;   // 0–0.9
    float wet      = 0.f;    // 0–1

    void setSampleRate(float sr)
    {
        sampleRate = sr;
        maxSamples = static_cast<int>(sr * 2.f);  // 2-second buffer
        buf.assign(maxSamples, 0.f);
        pos = 0;
    }

    void process(float* audio, int n)
    {
        if (wet == 0.f) return;

        int ds = static_cast<int>(timeMs * sampleRate / 1000.f);
        ds = std::clamp(ds, n + 1, maxSamples - 1);

        for (int i = 0; i < n; ++i)
        {
            int rp = (pos - ds + maxSamples) % maxSamples;
            float delayed = buf[rp];
            buf[pos] = audio[i] + delayed * feedback;
            pos = (pos + 1) % maxSamples;
            audio[i] = audio[i] * (1.f - wet) + delayed * wet;
        }
    }

private:
    float sampleRate = 44100.f;
    int   maxSamples = 88200;
    int   pos = 0;
    std::vector<float> buf = std::vector<float>(88200, 0.f);
};
