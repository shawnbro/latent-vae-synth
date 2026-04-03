#include "PluginEditor.h"
#include "PluginProcessor.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
LatentSynthEditor::LatentSynthEditor(LatentSynthProcessor& p)
    : AudioProcessorEditor(&p),
      processor(p),
      adsrGroup(p.apvts,
                { {"Attack",  "attack"},
                  {"Decay",   "decay"},
                  {"Sustain", "sustain"},
                  {"Release", "release"} }, 100),
      filterGroup(p.apvts,
                  { {"Cutoff",  "cutoff"},
                    {"Reson.",  "resonance"},
                    {"Env Amt", "filter_env_amount"} }, 100),
      reverbGroup(p.apvts,
                  { {"Rv Size", "reverb_size"},
                    {"Rv Wet",  "reverb_wet"} }, 90),
      delayGroup(p.apvts,
                 { {"Dly ms",  "delay_time"},
                   {"Dly FB",  "delay_feedback"},
                   {"Dly Wet", "delay_wet"} }, 90),
      motionGroup(p.apvts),
      voicesGroup(p.apvts,
                  { {"Voices", "unison_voices"},
                    {"Detune", "unison_detune"} }, 90),
      ioGroup(p.apvts,
              { {"Gain", "gain"} }, 90)
{
    // Latent X / Y sliders
    auto setupXY = [&](juce::Slider& s, juce::Label& lbl, const char* text,
                       const juce::String& paramId,
                       std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>& att)
    {
        s.setSliderStyle(juce::Slider::LinearHorizontal);
        s.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 14);
        addAndMakeVisible(s);
        lbl.setText(text, juce::dontSendNotification);
        lbl.setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(9.f)));
        addAndMakeVisible(lbl);
        att = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            p.apvts, paramId, s);
    };
    setupXY(latentXSlider, latentXLabel, "X  CC1:",  "latent_x", latentXAttach);
    setupXY(latentYSlider, latentYLabel, "Y CC11:", "latent_y", latentYAttach);

    setLookAndFeel(&laf);
    setSize(820, 740);

    addAndMakeVisible(titleBar);
    addAndMakeVisible(xyPad);
    addAndMakeVisible(waveDisplay);
    addAndMakeVisible(adsrSection);
    addAndMakeVisible(filterSection);
    addAndMakeVisible(reverbSection);
    addAndMakeVisible(delaySection);
    addAndMakeVisible(motionSection);
    addAndMakeVisible(voicesSection);
    addAndMakeVisible(ioSection);

    auto relayout = [this] { resized(); repaint(); };
    adsrSection.onToggle   = relayout;
    filterSection.onToggle = relayout;
    reverbSection.onToggle = relayout;
    delaySection.onToggle  = relayout;
    motionSection.onToggle = relayout;
    voicesSection.onToggle = relayout;
    ioSection.onToggle     = relayout;

    // Wire XY pad → APVTS (also updates LFO center when LFO is on)
    xyPad.onChange = [&](float x, float y) {
        if (auto* px = p.apvts.getParameter("latent_x"))
            px->setValueNotifyingHost(px->convertTo0to1(x));
        if (auto* py = p.apvts.getParameter("latent_y"))
            py->setValueNotifyingHost(py->convertTo0to1(y));
    };

    xyPad.setXY(*p.apvts.getRawParameterValue("latent_x"),
                *p.apvts.getRawParameterValue("latent_y"));

    startTimerHz(30);
}

LatentSynthEditor::~LatentSynthEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

// ---------------------------------------------------------------------------
// paint — Mac Classic chrome
// ---------------------------------------------------------------------------
void LatentSynthEditor::paint(juce::Graphics& g)
{
    g.fillAll(MacColours::BG);

    g.setColour(MacColours::BLACK);
    g.drawRect(getLocalBounds(), 1);

    const int TBH = 22;
    MacClassicLookAndFeel::drawBevel(g, 1, TBH, getWidth() - 2, getHeight() - TBH - 1, false);

    const int DIV = 320;
    g.setColour(MacColours::SHADOW);
    g.fillRect(DIV, TBH + 4, 1, getHeight() - TBH - 8);
    g.setColour(MacColours::HILIGHT);
    g.fillRect(DIV + 1, TBH + 4, 1, getHeight() - TBH - 8);

    auto sectionLabel = [&](const juce::String& t, int y) {
        g.setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(9.f).withStyle("Bold")));
        g.setColour(MacColours::BLACK);
        g.drawText(t, 8, y, 295, 12, juce::Justification::centredLeft, false);
    };
    sectionLabel("\xe2\x97\x86  LATENT SPACE", 26);
    // Oscilloscope label sits just above the waveform (XY_TOP+XY_H+4+24+24 = 22+16+290+52 = 380)
    sectionLabel("\xe2\x97\x86  OSCILLOSCOPE", 380);
}

// ---------------------------------------------------------------------------
// resized
// ---------------------------------------------------------------------------
void LatentSynthEditor::resized()
{
    const int TBH = 22;
    const int PAD = 6;
    const int DIV = 320;

    titleBar.setBounds(0, 0, getWidth(), TBH);

    // Left column
    const int XY_H   = 290;
    const int XY_TOP = TBH + 16;
    const int SL_W   = DIV - PAD * 2;

    xyPad.setBounds(PAD, XY_TOP, SL_W, XY_H);

    // X / Y sliders just below the XY pad
    int sly = XY_TOP + XY_H + 4;
    latentXLabel .setBounds(PAD,          sly,      36, 20);
    latentXSlider.setBounds(PAD + 38,     sly,      SL_W - 38, 20);
    sly += 24;
    latentYLabel .setBounds(PAD,          sly,      36, 20);
    latentYSlider.setBounds(PAD + 38,     sly,      SL_W - 38, 20);

    // Oscilloscope below sliders
    waveDisplay.setBounds(PAD, sly + 30, SL_W, 66);

    // Right column: stacked collapsible sections
    layoutSections(DIV + PAD + 2, TBH + PAD, getWidth() - DIV - PAD * 2 - 4);
}

void LatentSynthEditor::layoutSections(int x, int y, int w)
{
    auto place = [&](CollapsibleSection& sec, juce::Component& body) {
        body.setSize(w - 8, body.getHeight());
        int h = sec.getPreferredHeight();
        sec.setBounds(x, y, w, h);
        y += h + 4;
    };

    place(adsrSection,   adsrGroup);
    place(filterSection, filterGroup);
    place(reverbSection, reverbGroup);
    place(delaySection,  delayGroup);
    place(motionSection, motionGroup);
    place(voicesSection, voicesGroup);
    place(ioSection,     ioGroup);
}

// ---------------------------------------------------------------------------
// Timer — refresh waveform + sync XY pad to actual synth position
// ---------------------------------------------------------------------------
void LatentSynthEditor::timerCallback()
{
    auto& synth = processor.getSynth();
    waveDisplay.setWaveform(synth.currentWaveform.data(), LatentSynth::WAVEFORM_LEN);
    // XY cursor follows actual decode position (dances when LFO is active)
    xyPad.setXY(synth.latentX, synth.latentY);
}

// ---------------------------------------------------------------------------
// createEditor (called from PluginProcessor)
// ---------------------------------------------------------------------------
juce::AudioProcessorEditor* LatentSynthProcessor::createEditor()
{
    return new LatentSynthEditor(*this);
}
