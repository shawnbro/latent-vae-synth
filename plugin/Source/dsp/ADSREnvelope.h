#pragma once
#include <algorithm>
#include <cmath>

/**
 * Per-block ADSR amplitude envelope.
 * Ported from Python ADSREnvelope in app/synth.py (lines 116–189).
 *
 * Processes blocks of N samples, writing envelope levels into a caller-supplied
 * buffer. State transitions happen mid-block for accuracy.
 */
class ADSREnvelope
{
public:
    enum State { IDLE, ATTACK, DECAY, SUSTAIN, RELEASE };

    ADSREnvelope() { setAttack(10.f); setDecay(100.f); setRelease(300.f); }

    void setSampleRate(float sr) { sampleRate = sr; }

    void setAttack(float ms)   { attackRate  = msToRate(std::max(1.f, ms)); }
    void setDecay(float ms)    { decayRate   = msToRate(std::max(1.f, ms)); }
    void setSustain(float lvl) { sustain = std::clamp(lvl, 0.f, 1.f); }
    void setRelease(float ms)  { releaseRate = msToRate(std::max(1.f, ms)); }

    void noteOn()  { state = ATTACK; }
    void noteOff() { if (state != IDLE) { releaseStart = level; state = RELEASE; } }

    bool isActive() const { return state != IDLE; }
    float getLevel() const { return level; }

    /** Fill buf[0..n-1] with per-sample envelope levels. */
    void process(float* buf, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            switch (state)
            {
                case ATTACK:
                    level += attackRate;
                    if (level >= 1.f) { level = 1.f; state = DECAY; }
                    break;

                case DECAY:
                    level -= decayRate;
                    if (level <= sustain) { level = sustain; state = SUSTAIN; }
                    break;

                case SUSTAIN:
                    level = sustain;
                    break;

                case RELEASE:
                    level -= releaseStart * releaseRate;
                    if (level <= 0.f) { level = 0.f; state = IDLE; }
                    break;

                case IDLE:
                    level = 0.f;
                    break;
            }
            buf[i] = level;
        }
    }

private:
    float sampleRate  = 44100.f;
    State state       = IDLE;
    float level       = 0.f;
    float releaseStart = 0.f;
    float sustain     = 0.7f;
    float attackRate  = 0.f;
    float decayRate   = 0.f;
    float releaseRate = 0.f;

    float msToRate(float ms) const
    {
        return 1.f / std::max(1.f, ms * sampleRate / 1000.f);
    }
};
