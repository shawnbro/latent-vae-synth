#pragma once
#include <array>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include "Voice.h"
#include "SchroederReverb.h"
#include "FeedbackDelay.h"

/**
 * Core synthesis engine.
 * Wraps ONNX Runtime for wavetable decoding and manages the voice pool.
 * Ported from Python LatentSynth in app/synth.py.
 *
 * Usage:
 *   // Construct once with embedded model bytes (from JUCE BinaryData):
 *   LatentSynth synth(BinaryData::decoder_onnx, BinaryData::decoder_onnxSize);
 *   synth.setSampleRate(44100.f);
 *
 *   // Control:
 *   synth.setLatent(x, y);        // move latent position
 *   synth.noteOn(60, 100);
 *   synth.noteOff(60);
 *
 *   // Audio:
 *   synth.processBlock(bufL, bufR, numSamples);
 */
class LatentSynth
{
public:
    static constexpr int MAX_VOICES   = 16;
    static constexpr int WAVEFORM_LEN = 2048;

    // --- Parameters (set directly from processor) ---------------------------
    float latentX    = 0.f;
    float latentY    = 0.f;
    float masterGain = 0.7f;

    // ADSR (ms / level)
    float attackMs   = 10.f;
    float decayMs    = 100.f;
    float sustainLvl = 0.7f;
    float releaseMs  = 300.f;

    // Filter
    float cutoffHz   = 18000.f;
    float resonance  = 0.f;
    float filterEnvAmount = 0.f;

    // FX
    SchroederReverb reverb;
    FeedbackDelay   delay;

    // Unison
    int   unisonVoices = 1;
    float unisonDetune = 0.f;   // total spread in cents (voices span ±detune)

    // Velocity → Latent
    float velDepth = 0.f;       // latent units of offset at full velocity
    float velAngle = 0.f;       // direction of offset in degrees

    // -------------------------------------------------------------------------

    LatentSynth() = default;

    /**
     * Initialise with an in-memory ONNX model (e.g. from JUCE BinaryData).
     * Call once before any audio processing.
     */
    void init(const void* modelData, size_t modelSize)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        session = std::make_unique<Ort::Session>(ortEnv, modelData, modelSize, opts);
        decodeLatent(0.f, 0.f, currentWaveform.data());
    }

    void setSampleRate(float sr)
    {
        sampleRate = sr;
        for (auto& v : voices)
            v.setSampleRate(sr);
        reverb.wet = 0.f;
        delay.setSampleRate(sr);
    }

    /** Set glide time. Call from message thread. */
    void setGlide(float ms)
    {
        glideMs = ms;
        if (ms <= 0.f)
            morphRate = 0.f;
        else
            morphRate = (float)BLOCK_SIZE_HINT / std::max(1.f, ms * sampleRate / 1000.f);
    }

    /**
     * Decode a latent point and update the current waveform (with glide if set).
     * Call only from the message thread — runs ONNX inference.
     */
    void setLatent(float x, float y)
    {
        latentX = x;
        latentY = y;
        if (!session) return;

        std::array<float, WAVEFORM_LEN> newWf;
        decodeLatent(x, y, newWf.data());

        std::lock_guard<std::mutex> lk(waveformMutex);
        if (morphRate > 0.f)
        {
            // Freeze current blend position as the new "from"
            float t = morph.load();
            for (int i = 0; i < WAVEFORM_LEN; ++i)
                waveformFrom[i] = (1.f - t) * waveformFrom[i] + t * waveformTo[i];
            waveformTo = newWf;
            morph.store(0.f);
        }
        else
        {
            waveformTo   = newWf;
            waveformFrom = newWf;
            currentWaveform = newWf;
            morph.store(1.f);
        }
    }

    /** Decode a latent point without changing state — returns waveform in out2048. */
    void decodeLatent(float x, float y, float* out2048)
    {
        if (!session)
        {
            std::fill(out2048, out2048 + WAVEFORM_LEN, 0.f);
            return;
        }

        std::array<float, 2> input = {x, y};
        std::array<int64_t, 2> inputShape = {1, 2};

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), 2, inputShape.data(), 2);

        const char* inputNames[]  = {"latent"};
        const char* outputNames[] = {"waveform"};

        auto results = session->Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inputTensor, 1,
            outputNames, 1);

        const float* ptr = results[0].GetTensorData<float>();
        std::copy(ptr, ptr + WAVEFORM_LEN, out2048);
    }

    void noteOn(int midiNote, int velocity)
    {
        // Velocity → latent: decode an offset waveform at noteOn time
        const float* wfOverride = nullptr;
        std::array<float, WAVEFORM_LEN> velWf;
        if (velDepth > 0.f && session)
        {
            float vel = velocity / 127.f;
            float rad = velAngle * 3.14159265f / 180.f;
            float ox  = std::clamp(latentX + std::sin(rad) * velDepth * vel, -4.f, 4.f);
            float oy  = std::clamp(latentY + std::cos(rad) * velDepth * vel, -4.f, 4.f);
            decodeLatent(ox, oy, velWf.data());
            wfOverride = velWf.data();
        }

        // Spawn unisonVoices voices — find inactive slots first, then steal
        int spawned = 0;
        for (int i = 0; i < MAX_VOICES && spawned < unisonVoices; ++i)
            if (!voices[i].isActive())
                spawnVoice(voices[i], midiNote, velocity, spawned++, wfOverride);
        for (int i = 0; i < MAX_VOICES && spawned < unisonVoices; ++i)
            if (voices[i].midiNote != midiNote)   // don't double-steal our own note
                spawnVoice(voices[i], midiNote, velocity, spawned++, wfOverride);
    }

    void noteOff(int midiNote)
    {
        for (auto& v : voices)
            if (v.midiNote == midiNote)
                v.noteOff();
    }

private:
    /** Configure and trigger one voice within a unison chord. */
    void spawnVoice(Voice& v, int note, int vel, int vi, const float* wfOverride)
    {
        float detune = 0.f;
        if (unisonVoices > 1)
            detune = ((float)vi / (float)(unisonVoices - 1) - 0.5f) * unisonDetune * 2.f;
        v.ampEnv.setAttack(attackMs);
        v.ampEnv.setDecay(decayMs);
        v.ampEnv.setSustain(sustainLvl);
        v.ampEnv.setRelease(releaseMs);
        v.noteOn(note, vel, cutoffHz, resonance, wfOverride, detune);
    }

public:

    void allNotesOff()
    {
        for (auto& v : voices) v.noteOff();
    }

    /**
     * Fill stereo output buffers (interleaved or separate).
     * buf is a mono scratch buffer; the result is copied to both channels.
     */
    void processBlock(float* outL, float* outR, int numSamples)
    {
        // Grow scratch if block size is larger than expected
        if (numSamples > (int)scratch.size())
            scratch.resize(numSamples * 2, 0.f);
        std::fill(scratch.begin(), scratch.begin() + numSamples, 0.f);

        // Advance glide morph and update currentWaveform (audio thread)
        {
            float t = morph.load();
            if (t < 1.f)
            {
                float mr = morphRate;
                // Guard waveformFrom/To against concurrent setLatent writes
                if (waveformMutex.try_lock())
                {
                    t = std::min(1.f, t + mr * numSamples);
                    for (int i = 0; i < WAVEFORM_LEN; ++i)
                        currentWaveform[i] = (1.f - t) * waveformFrom[i] + t * waveformTo[i];
                    morph.store(t);
                    waveformMutex.unlock();
                }
            }
        }

        bool anyActive = false;
        for (auto& v : voices)
            if (v.render(scratch.data(), numSamples, currentWaveform.data(),
                         cutoffHz, filterEnvAmount))
                anyActive = true;

        if (anyActive)
        {
            reverb.process(scratch.data(), numSamples);
            delay.process(scratch.data(), numSamples);
        }

        for (int i = 0; i < numSamples; ++i)
        {
            float s = scratch[i] * masterGain;
            outL[i] += s;
            outR[i] += s;
        }
    }

    // Read-only by editor for waveform display
    std::array<float, WAVEFORM_LEN> currentWaveform{};

    // --- Glide / morph state (shared between message thread writer + audio thread reader) ---
    float glideMs  = 0.f;
    float morphRate = 0.f;                          // fraction of morph per sample
    std::atomic<float> morph{1.f};                  // 0=fully from, 1=fully to
    std::array<float, WAVEFORM_LEN> waveformFrom{}; // protected by waveformMutex
    std::array<float, WAVEFORM_LEN> waveformTo{};   // protected by waveformMutex
    std::mutex waveformMutex;

    static constexpr int BLOCK_SIZE_HINT = 512;

private:
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "LatentSynth"};
    std::unique_ptr<Ort::Session> session;

    float sampleRate = 44100.f;
    std::array<Voice, MAX_VOICES> voices;
    std::vector<float>            scratch = std::vector<float>(4096, 0.f);
};
