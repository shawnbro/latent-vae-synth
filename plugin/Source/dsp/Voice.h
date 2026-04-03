#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include "ADSREnvelope.h"
#include "BiquadFilter.h"

/**
 * Single wavetable voice with ADSR envelope and resonant lowpass filter.
 * Ported from Python Voice in app/synth.py (lines 605–664).
 *
 * The oscillator linearly interpolates between adjacent wavetable samples.
 */
class Voice
{
public:
    static constexpr int WAVEFORM_LEN = 2048;

    int   midiNote  = -1;
    float velocity  = 0.f;
    bool  hasOwnWaveform = false;
    std::array<float, WAVEFORM_LEN> ownWaveform{};

    ADSREnvelope ampEnv;
    ADSREnvelope filtEnv;
    BiquadFilter filt;

    bool isActive() const { return ampEnv.isActive(); }

    void setSampleRate(float sr)
    {
        ampEnv.setSampleRate(sr);
        filtEnv.setSampleRate(sr);
        filt.setSampleRate(sr);
        sampleRate = sr;
    }

    void noteOn(int note, int vel, float cutoffHz, float resonance,
                const float* waveformOverride = nullptr, float detuneCents = 0.f)
    {
        midiNote  = note;
        velocity  = vel / 127.f;
        frequency = 440.f * std::pow(2.f, (note - 69) / 12.f)
                          * std::pow(2.f, detuneCents / 1200.f);

        filt.setCutoff(cutoffHz);
        filt.setResonance(resonance);

        if (!ampEnv.isActive())
            filt.reset();   // only clear on fresh start to avoid clicks

        if (waveformOverride)
        {
            std::copy(waveformOverride, waveformOverride + WAVEFORM_LEN, ownWaveform.begin());
            hasOwnWaveform = true;
        }
        else
        {
            hasOwnWaveform = false;
        }

        ampEnv.noteOn();
        filtEnv.noteOn();
    }

    void noteOff()
    {
        ampEnv.noteOff();
        filtEnv.noteOff();
        midiNote = -1;
    }

    /**
     * Render one block into buf[0..n-1], mixing on top of whatever is there.
     * globalWaveform is used when hasOwnWaveform == false.
     * Returns false if the voice is idle (caller can skip).
     */
    bool render(float* buf, int n,
                const float* globalWaveform,
                float baseCutoff, float envAmount)
    {
        if (!isActive()) return false;

        const float* wf = hasOwnWaveform ? ownWaveform.data() : globalWaveform;
        const double phaseInc = WAVEFORM_LEN * (double)frequency / sampleRate;

        // Grow heap buffers if needed (avoids stack overflow for large block sizes)
        if (n > (int)renderBuf.size())
            renderBuf.resize(n * 3); // ampBuf | filtBuf | scratch packed in one alloc

        float* const ampBufPtr  = renderBuf.data();
        float* const filtBufPtr = renderBuf.data() + n;
        float* const scratchPtr = renderBuf.data() + n * 2;

        ampEnv.process(ampBufPtr, n);
        filtEnv.process(filtBufPtr, n);

        // Filter envelope modulation: average env level for this block
        if (envAmount > 0.f)
        {
            float sum = 0.f;
            for (int i = 0; i < n; ++i) sum += filtBufPtr[i];
            float avg = sum / n;
            filt.setCutoff(baseCutoff + envAmount * (18000.f - baseCutoff) * avg);
        }

        // Wavetable oscillator with linear interpolation
        for (int i = 0; i < n; ++i)
        {
            double p   = std::fmod(phase + (double)i * phaseInc, (double)WAVEFORM_LEN);
            int    idx = (int)p;
            float  frac = (float)(p - idx);
            int    nxt = (idx + 1) % WAVEFORM_LEN;
            scratchPtr[i] = wf[idx] * (1.f - frac) + wf[nxt] * frac;
        }
        phase = std::fmod(phase + (double)n * phaseInc, (double)WAVEFORM_LEN);

        // Apply filter
        filt.process(scratchPtr, n);

        // Mix into output
        for (int i = 0; i < n; ++i)
            buf[i] += scratchPtr[i] * ampBufPtr[i] * velocity;

        return true;
    }

private:
    float  sampleRate = 44100.f;
    float  frequency  = 440.f;
    double phase      = 0.0;
    // Single heap allocation for all per-block render scratch (avoids stack overflow)
    std::vector<float> renderBuf = std::vector<float>(512 * 3, 0.f);
};
