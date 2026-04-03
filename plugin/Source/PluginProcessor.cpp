#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <BinaryData.h>

// ---------------------------------------------------------------------------
// Parameter IDs (keep in sync with createParameterLayout)
// ---------------------------------------------------------------------------
namespace ParamID
{
    static const juce::String latentX       = "latent_x";
    static const juce::String latentY       = "latent_y";
    static const juce::String attack        = "attack";
    static const juce::String decay         = "decay";
    static const juce::String sustain       = "sustain";
    static const juce::String release       = "release";
    static const juce::String cutoff        = "cutoff";
    static const juce::String resonance     = "resonance";
    static const juce::String filtEnvAmt    = "filter_env_amount";
    static const juce::String reverbSize    = "reverb_size";
    static const juce::String reverbWet     = "reverb_wet";
    static const juce::String delayTime     = "delay_time";
    static const juce::String delayFeedback = "delay_feedback";
    static const juce::String delayWet      = "delay_wet";
    static const juce::String gain          = "gain";
    // LFO / Motion
    static const juce::String lfoActive     = "lfo_active";
    static const juce::String lfoShape      = "lfo_shape";
    static const juce::String lfoRate       = "lfo_rate";
    static const juce::String lfoDepth      = "lfo_depth";
    static const juce::String glide         = "glide";
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
LatentSynthProcessor::LatentSynthProcessor()
    : AudioProcessor(BusesProperties()
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    // synth.init() is deferred to prepareToPlay() to avoid running ONNX
    // during the build-time VST3 manifest extraction step.
}

// ---------------------------------------------------------------------------
// Parameter layout
// ---------------------------------------------------------------------------
juce::AudioProcessorValueTreeState::ParameterLayout
LatentSynthProcessor::createParameterLayout()
{
    using Param  = juce::AudioParameterFloat;
    using Range  = juce::NormalisableRange<float>;

    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Latent coordinates
    layout.add(std::make_unique<Param>(ParamID::latentX, "Latent X",
        Range(-4.f, 4.f, 0.001f), 0.f));
    layout.add(std::make_unique<Param>(ParamID::latentY, "Latent Y",
        Range(-4.f, 4.f, 0.001f), 0.f));

    // ADSR (milliseconds / level)
    layout.add(std::make_unique<Param>(ParamID::attack,  "Attack",
        Range(1.f, 5000.f, 1.f, 0.3f), 10.f));
    layout.add(std::make_unique<Param>(ParamID::decay,   "Decay",
        Range(1.f, 5000.f, 1.f, 0.3f), 100.f));
    layout.add(std::make_unique<Param>(ParamID::sustain, "Sustain",
        Range(0.f, 1.f, 0.001f), 0.7f));
    layout.add(std::make_unique<Param>(ParamID::release, "Release",
        Range(1.f, 8000.f, 1.f, 0.3f), 300.f));

    // Filter
    layout.add(std::make_unique<Param>(ParamID::cutoff,    "Cutoff",
        Range(30.f, 18000.f, 1.f, 0.35f), 18000.f));
    layout.add(std::make_unique<Param>(ParamID::resonance, "Resonance",
        Range(0.f, 0.95f, 0.001f), 0.f));
    layout.add(std::make_unique<Param>(ParamID::filtEnvAmt, "Filter Env",
        Range(0.f, 1.f, 0.001f), 0.f));

    // Reverb
    layout.add(std::make_unique<Param>(ParamID::reverbSize, "Reverb Size",
        Range(0.f, 1.f, 0.001f), 0.5f));
    layout.add(std::make_unique<Param>(ParamID::reverbWet,  "Reverb Wet",
        Range(0.f, 1.f, 0.001f), 0.f));

    // Delay
    layout.add(std::make_unique<Param>(ParamID::delayTime,     "Delay Time",
        Range(20.f, 1000.f, 1.f), 250.f));
    layout.add(std::make_unique<Param>(ParamID::delayFeedback, "Delay Feedback",
        Range(0.f, 0.9f, 0.001f), 0.4f));
    layout.add(std::make_unique<Param>(ParamID::delayWet,      "Delay Wet",
        Range(0.f, 1.f, 0.001f), 0.f));

    // Master gain
    layout.add(std::make_unique<Param>(ParamID::gain, "Gain",
        Range(0.f, 1.f, 0.001f), 0.7f));

    // LFO / Motion
    layout.add(std::make_unique<juce::AudioParameterBool>(
        ParamID::lfoActive, "LFO Active", false));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        ParamID::lfoShape, "LFO Shape",
        juce::StringArray{"Circ", "X Scan", "Y Scan", "Walk", "Wave"}, 0));
    layout.add(std::make_unique<Param>(ParamID::lfoRate, "LFO Rate",
        Range(0.01f, 10.f, 0.01f, 0.5f), 0.2f));
    layout.add(std::make_unique<Param>(ParamID::lfoDepth, "LFO Depth",
        Range(0.f, 4.f, 0.01f), 1.0f));
    layout.add(std::make_unique<Param>(ParamID::glide, "Glide",
        Range(0.f, 2000.f, 1.f, 0.4f), 0.f));

    // Velocity → Latent
    layout.add(std::make_unique<Param>("vel_depth", "Vel Depth",
        Range(0.f, 4.f, 0.01f), 0.f));
    layout.add(std::make_unique<Param>("vel_angle", "Vel Angle",
        Range(0.f, 360.f, 1.f), 0.f));

    // Unison / Voices
    layout.add(std::make_unique<Param>("unison_voices", "Voices",
        Range(1.f, 8.f, 1.f), 1.f));
    layout.add(std::make_unique<Param>("unison_detune", "Unison Detune",
        Range(0.f, 100.f, 0.1f, 0.5f), 0.f));

    return layout;
}

// ---------------------------------------------------------------------------
// prepareToPlay
// ---------------------------------------------------------------------------
void LatentSynthProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    std::call_once(synthInitFlag, [this] {
        synth.init(BinaryData::decoder_onnx, (size_t)BinaryData::decoder_onnxSize);
    });
    synth.setSampleRate((float)sampleRate);

    // Wire LFO: callback stores pending position; timer reads it on message thread
    lfo.onUpdate = [this](float x, float y) {
        pendingLfoX.store(x);
        pendingLfoY.store(y);
        lfoHasPending.store(true);
    };

    startTimerHz(30);
}

// ---------------------------------------------------------------------------
// processBlock
// ---------------------------------------------------------------------------
void LatentSynthProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                        juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    // Apply non-ONNX parameters directly (safe, no allocation)
    synth.attackMs        = *apvts.getRawParameterValue("attack");
    synth.decayMs         = *apvts.getRawParameterValue("decay");
    synth.sustainLvl      = *apvts.getRawParameterValue("sustain");
    synth.releaseMs       = *apvts.getRawParameterValue("release");
    synth.cutoffHz        = *apvts.getRawParameterValue("cutoff");
    synth.resonance       = *apvts.getRawParameterValue("resonance");
    synth.filterEnvAmount = *apvts.getRawParameterValue("filter_env_amount");
    synth.reverb.roomSize = *apvts.getRawParameterValue("reverb_size");
    synth.reverb.wet      = *apvts.getRawParameterValue("reverb_wet");
    synth.delay.timeMs    = *apvts.getRawParameterValue("delay_time");
    synth.delay.feedback  = *apvts.getRawParameterValue("delay_feedback");
    synth.delay.wet       = *apvts.getRawParameterValue("delay_wet");
    synth.masterGain      = *apvts.getRawParameterValue("gain");
    // Unison + velocity→latent (applied before noteOn in same block)
    synth.unisonVoices    = (int)(*apvts.getRawParameterValue("unison_voices") + 0.5f);
    synth.unisonDetune    = *apvts.getRawParameterValue("unison_detune");
    synth.velDepth        = *apvts.getRawParameterValue("vel_depth");
    synth.velAngle        = *apvts.getRawParameterValue("vel_angle");
    // NOTE: setLatent (ONNX decode) is called from timerCallback on the message thread

    // Clear output
    buffer.clear();

    // Process MIDI events
    int startSample = 0;
    for (const auto metadata : midiMessages)
    {
        const auto msg = metadata.getMessage();
        // TODO: per-event sample-accurate rendering (future improvement)
        // For now process all MIDI then render the block
        if (msg.isNoteOn())
            synth.noteOn(msg.getNoteNumber(), msg.getVelocity());
        else if (msg.isNoteOff() || (msg.isNoteOn() && msg.getVelocity() == 0))
            synth.noteOff(msg.getNoteNumber());
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            synth.allNotesOff();
        else if (msg.isController())
        {
            // CC1  (mod wheel)  → Latent X
            // CC11 (expression) → Latent Y
            // The DAW can also map any other CC to these params via automation.
            auto setParam = [&](const juce::String& id, int ccVal) {
                if (auto* p = apvts.getParameter(id))
                    p->setValueNotifyingHost(p->convertTo0to1(ccVal / 127.f * 8.f - 4.f));
            };
            if      (msg.getControllerNumber() == 1)  setParam("latent_x", msg.getControllerValue());
            else if (msg.getControllerNumber() == 11) setParam("latent_y", msg.getControllerValue());
        }
    }

    // Render audio
    const int numSamples = buffer.getNumSamples();
    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : outL;

    synth.processBlock(outL, outR, numSamples);
}

// ---------------------------------------------------------------------------
// State persistence
// ---------------------------------------------------------------------------
void LatentSynthProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void LatentSynthProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

// ---------------------------------------------------------------------------
// applyParameters — decode latent point from APVTS (no LFO override)
// ---------------------------------------------------------------------------
void LatentSynthProcessor::applyParameters()
{
    float x = *apvts.getRawParameterValue(ParamID::latentX);
    float y = *apvts.getRawParameterValue(ParamID::latentY);
    if (x != synth.latentX || y != synth.latentY)
        synth.setLatent(x, y);   // ONNX decode — safe on message thread
}

// ---------------------------------------------------------------------------
// timerCallback — runs on message thread at 30 Hz
// ---------------------------------------------------------------------------
void LatentSynthProcessor::timerCallback()
{
    // Always update glide from param
    synth.setGlide(*apvts.getRawParameterValue(ParamID::glide));

    bool lfoOn = *apvts.getRawParameterValue(ParamID::lfoActive) > 0.5f;

    if (lfoOn)
    {
        // Keep LFO params fresh
        lfo.rate .store(*apvts.getRawParameterValue(ParamID::lfoRate));
        lfo.depth.store(*apvts.getRawParameterValue(ParamID::lfoDepth));
        lfo.shape.store((int)*apvts.getRawParameterValue(ParamID::lfoShape));

        float cx = *apvts.getRawParameterValue(ParamID::latentX);
        float cy = *apvts.getRawParameterValue(ParamID::latentY);
        lfo.centerX.store(cx);
        lfo.centerY.store(cy);

        if (!lfo.active.load())
            lfo.start(cx, cy);

        // Apply any position posted by the LFO background thread
        if (lfoHasPending.exchange(false))
            synth.setLatent(pendingLfoX.load(), pendingLfoY.load());
    }
    else
    {
        if (lfo.active.load()) lfo.stop();
        applyParameters();   // normal manual-position decode
    }
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new LatentSynthProcessor();
}
