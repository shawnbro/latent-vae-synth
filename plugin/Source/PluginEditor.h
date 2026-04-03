#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

// ---------------------------------------------------------------------------
// Mac Classic colour palette
// ---------------------------------------------------------------------------
namespace MacColours
{
    static const juce::Colour BG      { 0xffc0c0c0 };
    static const juce::Colour WHITE   { 0xffffffff };
    static const juce::Colour BLACK   { 0xff000000 };
    static const juce::Colour SHADOW  { 0xff808080 };
    static const juce::Colour HILIGHT { 0xfff0f0f0 };
    static const juce::Colour BLUE    { 0xff000099 }; // selection accent
}

// ---------------------------------------------------------------------------
// MacClassicLookAndFeel
// ---------------------------------------------------------------------------
class MacClassicLookAndFeel : public juce::LookAndFeel_V4
{
public:
    MacClassicLookAndFeel()
    {
        namespace C = MacColours;
        setColour(juce::ResizableWindow::backgroundColourId,    C::BG);
        setColour(juce::Label::textColourId,                    C::BLACK);
        setColour(juce::Label::backgroundColourId,              juce::Colours::transparentBlack);
        setColour(juce::Slider::thumbColourId,                  C::BG);
        setColour(juce::Slider::trackColourId,                  C::WHITE);
        setColour(juce::Slider::backgroundColourId,             C::WHITE);
        setColour(juce::Slider::textBoxTextColourId,            C::BLACK);
        setColour(juce::Slider::textBoxBackgroundColourId,      C::WHITE);
        setColour(juce::Slider::textBoxOutlineColourId,         C::BLACK);
        setColour(juce::Slider::textBoxHighlightColourId,       C::BLUE);
        setColour(juce::GroupComponent::textColourId,           C::BLACK);
        setColour(juce::GroupComponent::outlineColourId,        C::SHADOW);
        setColour(juce::ComboBox::textColourId,                 C::BLACK);
        setColour(juce::ComboBox::backgroundColourId,           C::BG);
        setColour(juce::ComboBox::outlineColourId,              C::SHADOW);
        setColour(juce::PopupMenu::backgroundColourId,          C::BG);
        setColour(juce::PopupMenu::textColourId,                C::BLACK);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, C::BLUE);
        setColour(juce::PopupMenu::highlightedTextColourId,     C::WHITE);
    }

    juce::Font getLabelFont(juce::Label&) override
    {
        return juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(10.f));
    }

    // Draw a Mac 3D bevel box
    static void drawBevel(juce::Graphics& g, int x, int y, int w, int h, bool raised)
    {
        auto light = raised ? MacColours::HILIGHT : MacColours::SHADOW;
        auto dark  = raised ? MacColours::SHADOW  : MacColours::HILIGHT;
        g.setColour(MacColours::BG);
        g.fillRect(x, y, w, h);
        g.setColour(light);
        g.fillRect(x,     y,     w - 1, 1);
        g.fillRect(x,     y,     1,     h - 1);
        g.setColour(dark);
        g.fillRect(x,     y + h - 1, w,     1);
        g.fillRect(x + w - 1, y,     1,     h);
    }

    // Draw a sunken inset (inverted bevel, for troughs / display areas)
    static void drawInset(juce::Graphics& g, int x, int y, int w, int h,
                          juce::Colour fill = MacColours::WHITE)
    {
        g.setColour(fill);
        g.fillRect(x + 1, y + 1, w - 2, h - 2);
        g.setColour(MacColours::SHADOW);
        g.fillRect(x,     y,     w - 1, 1);
        g.fillRect(x,     y,     1,     h - 1);
        g.setColour(MacColours::HILIGHT);
        g.fillRect(x,     y + h - 1, w,     1);
        g.fillRect(x + w - 1, y,     1,     h);
        g.setColour(MacColours::BLACK);
        g.drawRect(x, y, w, h, 1);
    }

    // Linear vertical/horizontal slider — Mac Scale style
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float /*min*/, float /*max*/,
                          juce::Slider::SliderStyle style, juce::Slider& /*s*/) override
    {
        bool isVert = (style == juce::Slider::LinearVertical);
        const int TROUGH_THICK = 5;
        const int THUMB_LONG   = 16;
        const int THUMB_THICK  = 14;

        if (isVert)
        {
            int tx = x + width / 2 - TROUGH_THICK / 2;
            drawInset(g, tx, y, TROUGH_THICK, height);
            int ty = (int)sliderPos - THUMB_LONG / 2;
            ty = std::clamp(ty, y, y + height - THUMB_LONG);
            drawBevel(g, x + width / 2 - THUMB_THICK / 2, ty, THUMB_THICK, THUMB_LONG, true);
        }
        else
        {
            int ty = y + height / 2 - TROUGH_THICK / 2;
            drawInset(g, x, ty, width, TROUGH_THICK);
            int tx = (int)sliderPos - THUMB_LONG / 2;
            tx = std::clamp(tx, x, x + width - THUMB_LONG);
            drawBevel(g, tx, y + height / 2 - THUMB_THICK / 2, THUMB_LONG, THUMB_THICK, true);
        }
    }

    void drawLinearSliderBackground(juce::Graphics&, int, int, int, int,
                                    float, float, float,
                                    juce::Slider::SliderStyle, juce::Slider&) override {}

    void drawLinearSliderThumb(juce::Graphics&, int, int, int, int,
                               float, float, float,
                               juce::Slider::SliderStyle, juce::Slider&) override {}

    void drawGroupComponentOutline(juce::Graphics& g, int w, int h,
                                   const juce::String& text,
                                   const juce::Justification&,
                                   juce::GroupComponent& comp) override
    {
        const int indent = 8;
        auto font = juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(10.f).withStyle("Bold"));
        int tw = font.getStringWidth(text) + 6;

        g.setColour(MacColours::SHADOW);
        g.drawRect(0, 8, w, h - 8);
        g.setColour(MacColours::HILIGHT);
        g.drawRect(1, 9, w, h - 8);

        g.setColour(MacColours::BG);
        g.fillRect(indent - 1, 3, tw + 2, 11);

        g.setColour(MacColours::BLACK);
        g.setFont(font);
        g.drawText(text, indent, 1, tw, 14, juce::Justification::centredLeft, false);
    }

    // Mac-style checkbox toggle button
    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& btn,
                          bool /*isMouseOver*/, bool /*isDown*/) override
    {
        const int bx = 2, boxSize = 12;
        const int by = (btn.getHeight() - boxSize) / 2;
        drawInset(g, bx, by, boxSize, boxSize);
        if (btn.getToggleState())
        {
            g.setColour(MacColours::BLACK);
            juce::Path tick;
            tick.startNewSubPath(bx + 2.f, by + 6.f);
            tick.lineTo(bx + 5.f, by + 10.f);
            tick.lineTo(bx + 10.f, by + 2.f);
            g.strokePath(tick, juce::PathStrokeType(1.5f));
        }
        g.setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(10.f)));
        g.setColour(MacColours::BLACK);
        g.drawText(btn.getButtonText(), bx + boxSize + 5, 0,
                   btn.getWidth() - boxSize - 7, btn.getHeight(),
                   juce::Justification::centredLeft, false);
    }

    // Mac-style combo box
    void drawComboBox(juce::Graphics& g, int width, int height, bool,
                      int, int, int, int, juce::ComboBox&) override
    {
        drawBevel(g, 0, 0, width, height, true);
        // Divider before arrow
        const int aw = 16;
        g.setColour(MacColours::SHADOW);
        g.fillRect(width - aw - 1, 1, 1, height - 2);
        g.setColour(MacColours::HILIGHT);
        g.fillRect(width - aw,     1, 1, height - 2);
        // Down-arrow triangle
        float cx = width - aw / 2.f;
        float cy = height / 2.f;
        juce::Path arrow;
        arrow.addTriangle(cx - 4.f, cy - 2.f, cx + 4.f, cy - 2.f, cx, cy + 3.f);
        g.setColour(MacColours::BLACK);
        g.fillPath(arrow);
    }

    juce::Font getComboBoxFont(juce::ComboBox&) override
    {
        return juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(10.f));
    }

    void positionComboBoxText(juce::ComboBox& box, juce::Label& label) override
    {
        label.setBounds(4, 1, box.getWidth() - 20, box.getHeight() - 2);
        label.setFont(getComboBoxFont(box));
    }
};

// ---------------------------------------------------------------------------
// MacTitleBar
// ---------------------------------------------------------------------------
class MacTitleBar : public juce::Component
{
public:
    MacTitleBar(const juce::String& title) : titleText(title) {}

    void paint(juce::Graphics& g) override
    {
        auto b = getLocalBounds();
        g.setColour(MacColours::BG);
        g.fillRect(b);
        g.setColour(MacColours::BLACK);
        g.fillRect(0, b.getHeight() - 1, b.getWidth(), 1);

        MacClassicLookAndFeel::drawBevel(g, 4, 4, 13, 13, true);
        g.setColour(MacColours::BLACK);
        g.drawRect(4, 4, 13, 13, 1);

        auto font = juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(11.f).withStyle("Bold"));
        g.setFont(font);
        int tw = font.getStringWidth(titleText);
        int tx = (b.getWidth() - tw) / 2;

        for (int sx = 20; sx < tx - 4; sx += 4)
        {
            g.setColour(MacColours::HILIGHT);
            g.fillRect(sx,     6, 3, 8);
            g.setColour(MacColours::SHADOW);
            g.fillRect(sx + 1, 7, 3, 8);
        }
        for (int sx = tx + tw + 4; sx < b.getWidth() - 8; sx += 4)
        {
            g.setColour(MacColours::HILIGHT);
            g.fillRect(sx,     6, 3, 8);
            g.setColour(MacColours::SHADOW);
            g.fillRect(sx + 1, 7, 3, 8);
        }

        g.setColour(MacColours::BLACK);
        g.setFont(font);
        g.drawText(titleText, tx, 0, tw + 1, b.getHeight(), juce::Justification::centredLeft, false);
    }

private:
    juce::String titleText;
};

// ---------------------------------------------------------------------------
// CollapsibleSection — Mac-style panel with toggle header
// ---------------------------------------------------------------------------
class CollapsibleSection : public juce::Component
{
public:
    CollapsibleSection(const juce::String& title, juce::Component* body, bool startOpen = true)
        : bodyComponent(body), open(startOpen)
    {
        headerBtn.setButtonText((startOpen ? juce::String("\xe2\x96\xbc") : juce::String("\xe2\x96\xb6"))
                                + "  " + title);
        headerBtn.setColour(juce::TextButton::buttonColourId,     MacColours::BG);
        headerBtn.setColour(juce::TextButton::buttonOnColourId,   MacColours::BG);
        headerBtn.setColour(juce::TextButton::textColourOffId,    MacColours::BLACK);
        headerBtn.setColour(juce::TextButton::textColourOnId,     MacColours::BLACK);
        headerBtn.onClick = [this, title] {
            open = !open;
            headerBtn.setButtonText((open ? juce::String("\xe2\x96\xbc") : juce::String("\xe2\x96\xb6"))
                                    + "  " + title);
            bodyComponent->setVisible(open);
            if (onToggle) onToggle();
        };
        addAndMakeVisible(headerBtn);
        addAndMakeVisible(*bodyComponent);
        bodyComponent->setVisible(open);
    }

    int getPreferredHeight() const
    {
        return HEADER_H + (open ? bodyComponent->getHeight() + 4 : 0);
    }

    std::function<void()> onToggle;

    void resized() override
    {
        headerBtn.setBounds(0, 0, getWidth(), HEADER_H);
        if (open)
            bodyComponent->setBounds(4, HEADER_H + 2, getWidth() - 8, bodyComponent->getHeight());
    }

    void paint(juce::Graphics& g) override
    {
        MacClassicLookAndFeel::drawBevel(g, 0, 0, getWidth(), getPreferredHeight(), false);
    }

    static constexpr int HEADER_H = 18;

private:
    juce::TextButton   headerBtn;
    juce::Component*   bodyComponent;
    bool               open;
};

// ---------------------------------------------------------------------------
// XYPad
// ---------------------------------------------------------------------------
class XYPad : public juce::Component
{
public:
    std::function<void(float x, float y)> onChange;

    void setXY(float x, float y)
    {
        normX = (x + 4.f) / 8.f;
        normY = 1.f - (y + 4.f) / 8.f;
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto b = getLocalBounds();
        MacClassicLookAndFeel::drawInset(g, 0, 0, b.getWidth(), b.getHeight(),
                                         juce::Colour(0xff1a1a2e));

        auto bf = b.toFloat().reduced(1.f);

        g.setColour(juce::Colours::white.withAlpha(0.07f));
        for (int i = 1; i < 4; ++i)
        {
            g.drawVerticalLine  ((int)(bf.getX() + bf.getWidth()  * i / 4.f), bf.getY(), bf.getBottom());
            g.drawHorizontalLine((int)(bf.getY() + bf.getHeight() * i / 4.f), bf.getX(), bf.getRight());
        }
        g.setColour(juce::Colours::white.withAlpha(0.18f));
        g.drawVerticalLine  ((int)bf.getCentreX(), bf.getY(), bf.getBottom());
        g.drawHorizontalLine((int)bf.getCentreY(), bf.getX(), bf.getRight());

        float px = bf.getX() + normX * bf.getWidth();
        float py = bf.getY() + normY * bf.getHeight();
        g.setColour(juce::Colour(0xff00d4ff));
        g.fillEllipse(px - 6.f, py - 6.f, 12.f, 12.f);
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.drawEllipse(px - 6.f, py - 6.f, 12.f, 12.f, 1.5f);
    }

    void mouseDown(const juce::MouseEvent& e) override { update(e); }
    void mouseDrag(const juce::MouseEvent& e) override { update(e); }

private:
    float normX = 0.5f, normY = 0.5f;

    void update(const juce::MouseEvent& e)
    {
        auto b = getLocalBounds().toFloat();
        normX = std::clamp((float)e.x / b.getWidth(),  0.f, 1.f);
        normY = std::clamp((float)e.y / b.getHeight(), 0.f, 1.f);
        repaint();
        if (onChange)
            onChange(normX * 8.f - 4.f, (1.f - normY) * 8.f - 4.f);
    }
};

// ---------------------------------------------------------------------------
// WaveformDisplay
// ---------------------------------------------------------------------------
class WaveformDisplay : public juce::Component
{
public:
    void setWaveform(const float* data, int len)
    {
        waveform.assign(data, data + len);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto b = getLocalBounds();
        MacClassicLookAndFeel::drawInset(g, 0, 0, b.getWidth(), b.getHeight(),
                                         juce::Colour(0xff0d0d1a));
        if (waveform.empty()) return;

        auto bf = b.toFloat().reduced(2.f);
        juce::Path path;
        int len = (int)waveform.size();
        for (int i = 0; i < len; ++i)
        {
            float xp = bf.getX() + bf.getWidth()  * (float)i / (float)(len - 1);
            float yp = bf.getCentreY() - waveform[i] * bf.getHeight() * 0.45f;
            if (i == 0) path.startNewSubPath(xp, yp); else path.lineTo(xp, yp);
        }
        g.setColour(juce::Colour(0xff00d4ff).withAlpha(0.9f));
        g.strokePath(path, juce::PathStrokeType(1.5f));
    }

private:
    std::vector<float> waveform;
};

// ---------------------------------------------------------------------------
// MacSliderGroup — a row of vertical sliders with labels
// ---------------------------------------------------------------------------
class MacSliderGroup : public juce::Component
{
public:
    struct Entry { juce::String label; juce::String paramId; };

    MacSliderGroup(juce::AudioProcessorValueTreeState& apvts,
                   std::vector<Entry> entries,
                   int sliderH = 80)
        : sliderHeight(sliderH)
    {
        for (auto& e : entries)
        {
            auto* s = sliders.add(new juce::Slider(juce::Slider::LinearVertical,
                                                    juce::Slider::TextBoxBelow));
            s->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 46, 13);
            addAndMakeVisible(s);

            auto* l = labels.add(new juce::Label({}, e.label));
            l->setJustificationType(juce::Justification::centred);
            l->setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(9.f)));
            addAndMakeVisible(l);

            attachments.push_back(
                std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                    apvts, e.paramId, *s));
        }
        setSize((int)entries.size() * SLIDER_W, sliderHeight + LABEL_H);
    }

    void resized() override
    {
        int n = sliders.size();
        int w = getWidth() / n;
        for (int i = 0; i < n; ++i)
        {
            sliders[i]->setBounds(i * w + 2, LABEL_H, w - 4, sliderHeight);
            labels[i] ->setBounds(i * w,     0,       w,     LABEL_H);
        }
    }

    static constexpr int SLIDER_W = 52;
    static constexpr int LABEL_H  = 13;

private:
    int sliderHeight;
    juce::OwnedArray<juce::Slider> sliders;
    juce::OwnedArray<juce::Label>  labels;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> attachments;
};

// ---------------------------------------------------------------------------
// MotionPanel — LFO on/off, shape selector, Rate/Depth/Glide sliders
// ---------------------------------------------------------------------------
class MotionPanel : public juce::Component
{
public:
    MotionPanel(juce::AudioProcessorValueTreeState& apvts)
    {
        // LFO Active toggle
        lfoToggle.setButtonText("LFO ON");
        addAndMakeVisible(lfoToggle);
        lfoAttach = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
            apvts, "lfo_active", lfoToggle);

        // Shape combo
        shapeCombo.addItemList({"Circ", "X Scan", "Y Scan", "Walk", "Wave"}, 1);
        addAndMakeVisible(shapeCombo);
        shapeAttach = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
            apvts, "lfo_shape", shapeCombo);

        // Shape label
        shapeLabel.setText("Shape:", juce::dontSendNotification);
        shapeLabel.setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(9.f)));
        addAndMakeVisible(shapeLabel);

        // Horizontal sliders: Rate, Depth, Glide
        auto addRow = [&](juce::Slider& s, juce::Label& lbl, const juce::String& text,
                          const juce::String& paramId)
        {
            s.setSliderStyle(juce::Slider::LinearHorizontal);
            s.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 14);
            addAndMakeVisible(s);

            lbl.setText(text, juce::dontSendNotification);
            lbl.setFont(juce::Font(juce::FontOptions{}.withName("Monaco").withHeight(9.f)));
            addAndMakeVisible(lbl);

            sliderAttachments.push_back(
                std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                    apvts, paramId, s));
        };

        addRow(rateSlider,     rateLabel,     "Rate",      "lfo_rate");
        addRow(depthSlider,    depthLabel,    "Depth",     "lfo_depth");
        addRow(glideSlider,    glideLabel,    "Glide",     "glide");
        addRow(velDepthSlider, velDepthLabel, "Vel Depth", "vel_depth");
        addRow(velAngleSlider, velAngleLabel, "Vel Angle", "vel_angle");

        setSize(100, ROW * 6 + 6);  // will be resized by parent
    }

    void resized() override
    {
        int w = getWidth();
        // Row 0: toggle + "Shape:" label + combo
        lfoToggle .setBounds(0,       2, 62, ROW - 2);
        shapeLabel.setBounds(64,      4, 38, ROW - 4);
        shapeCombo.setBounds(64 + 38, 2, w - 64 - 38, ROW - 2);

        // Rows 1-5: label (42px) + slider (rest)
        auto sliderRow = [&](juce::Label& lbl, juce::Slider& s, int row) {
            int y = row * ROW + 4;
            lbl.setBounds(0,  y, 42, ROW);
            s  .setBounds(44, y, w - 44, ROW);
        };
        sliderRow(rateLabel,     rateSlider,     1);
        sliderRow(depthLabel,    depthSlider,    2);
        sliderRow(glideLabel,    glideSlider,    3);
        sliderRow(velDepthLabel, velDepthSlider, 4);
        sliderRow(velAngleLabel, velAngleSlider, 5);
    }

    static constexpr int ROW = 24;

private:
    juce::ToggleButton lfoToggle;
    juce::Label        shapeLabel;
    juce::ComboBox     shapeCombo;
    juce::Slider       rateSlider, depthSlider, glideSlider, velDepthSlider, velAngleSlider;
    juce::Label        rateLabel,  depthLabel,  glideLabel,  velDepthLabel,  velAngleLabel;

    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>   lfoAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> shapeAttach;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
};

// ---------------------------------------------------------------------------
// PluginEditor
// ---------------------------------------------------------------------------
class LatentSynthEditor : public juce::AudioProcessorEditor,
                          private juce::Timer
{
public:
    explicit LatentSynthEditor(LatentSynthProcessor&);
    ~LatentSynthEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    LatentSynthProcessor& processor;
    MacClassicLookAndFeel laf;

    MacTitleBar       titleBar  { "Latent Synth" };
    XYPad             xyPad;
    WaveformDisplay   waveDisplay;

    // Explicit X / Y sliders (also receive MIDI CC1 / CC11)
    juce::Slider latentXSlider, latentYSlider;
    juce::Label  latentXLabel,  latentYLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> latentXAttach, latentYAttach;

    // Slider groups (bodies for collapsible sections)
    MacSliderGroup    adsrGroup;
    MacSliderGroup    filterGroup;
    MacSliderGroup    reverbGroup;
    MacSliderGroup    delayGroup;
    MotionPanel       motionGroup;
    MacSliderGroup    voicesGroup;
    MacSliderGroup    ioGroup;

    // Collapsible sections
    CollapsibleSection adsrSection  { "ENVELOPE", &adsrGroup,   true  };
    CollapsibleSection filterSection{ "FILTER",   &filterGroup, true  };
    CollapsibleSection reverbSection{ "REVERB",   &reverbGroup, false };
    CollapsibleSection delaySection { "DELAY",    &delayGroup,  false };
    CollapsibleSection motionSection{ "MOTION",   &motionGroup, true  };
    CollapsibleSection voicesSection{ "VOICES",   &voicesGroup, false };
    CollapsibleSection ioSection    { "I/O",      &ioGroup,     true  };

    void timerCallback() override;
    void layoutSections(int x, int y, int w);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentSynthEditor)
};
