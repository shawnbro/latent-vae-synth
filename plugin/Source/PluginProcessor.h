#pragma once
#include <mutex>
#include <atomic>
#include <juce_audio_utils/juce_audio_utils.h>
#include "dsp/LatentSynth.h"
#include "dsp/LatentLFO.h"

/**
 * LatentSynthProcessor — JUCE AudioProcessor that wraps LatentSynth.
 *
 * Exposes these parameters to the DAW (automation-ready via APVTS):
 *   latent_x / latent_y     (-4 – +4)
 *   attack / decay / sustain / release
 *   cutoff (30–18000 Hz)    resonance (0–0.95)
 *   filter_env_amount       (0–1)
 *   reverb_size / reverb_wet
 *   delay_time (ms)  delay_feedback  delay_wet
 *   gain (0–1)
 */
class LatentSynthProcessor : public juce::AudioProcessor,
                             private juce::Timer
{
public:
    LatentSynthProcessor();
    ~LatentSynthProcessor() override = default;

    // --- AudioProcessor interface -------------------------------------------
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override { stopTimer(); }
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Latent Synth"; }
    bool   acceptsMidi()        const override { return true; }
    bool   producesMidi()       const override { return false; }
    bool   isMidiEffect()       const override { return false; }
    double getTailLengthSeconds() const override { return 2.0; }

    int  getNumPrograms()    override { return 1; }
    int  getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    // --- Parameter tree (public so editor can attach) -----------------------
    juce::AudioProcessorValueTreeState apvts;

    // Expose synth for editor waveform readback
    LatentSynth& getSynth() { return synth; }

private:
    LatentSynth synth;
    LatentLFO   lfo;
    std::once_flag synthInitFlag;

    // Pending LFO position (written by LFO background thread, read by message-thread timer)
    std::atomic<float> pendingLfoX{0.f}, pendingLfoY{0.f};
    std::atomic<bool>  lfoHasPending{false};

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void applyParameters();     // called from message-thread timer only
    void timerCallback() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentSynthProcessor)
};
