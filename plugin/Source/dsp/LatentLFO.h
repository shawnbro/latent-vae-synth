#pragma once
#include <atomic>
#include <thread>
#include <functional>
#include <array>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

/**
 * LatentLFO — background thread that moves the latent cursor at ~50 Hz.
 *
 * Shapes: CIRCLE, X_SCAN, Y_SCAN, WALK (Ornstein-Uhlenbeck), WAVE (waveform LFO).
 *
 * Thread safety: all parameters are atomics. onUpdate is posted via
 * juce::MessageManager::callAsync by the caller so the actual setLatent call
 * (which runs ONNX) always happens on the JUCE message thread.
 */
class LatentLFO
{
public:
    enum Shape { CIRCLE = 0, X_SCAN, Y_SCAN, WALK, WAVE };

    std::atomic<float> rate   {0.2f};    // Hz
    std::atomic<float> depth  {1.0f};    // latent units
    std::atomic<int>   shape  {CIRCLE};
    std::atomic<float> centerX{0.f};
    std::atomic<float> centerY{0.f};
    std::atomic<bool>  active {false};

    // Wave LFO shape (written from message thread before start)
    std::array<float, 2048> waveformLFO{};
    std::atomic<bool> hasWaveform{false};

    /** Called from the LFO thread; caller must post to message thread. */
    std::function<void(float, float)> onUpdate;

    void start(float cx, float cy)
    {
        if (active.load()) return;
        centerX.store(cx);  centerY.store(cy);
        walkX = cx;         walkY = cy;
        phase = 0.f;
        active.store(true);
        std::thread([this] { run(); }).detach();
    }

    void stop() { active.store(false); }

    ~LatentLFO() { stop(); }

private:
    float phase = 0.f;
    float walkX = 0.f, walkY = 0.f;

    void run()
    {
        constexpr float DT = 0.02f;   // 50 Hz
        constexpr float PI2 = 6.283185307f;
        std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<float> gauss(0.f, 1.f);

        while (active.load())
        {
            float r  = rate.load();
            float d  = depth.load();
            float cx = centerX.load();
            float cy = centerY.load();

            phase += PI2 * r * DT;

            float x, y;
            switch (shape.load())
            {
                case CIRCLE:
                    x = cx + d * std::cos(phase);
                    y = cy + d * std::sin(phase);
                    break;
                case X_SCAN:
                    x = cx + d * std::sin(phase);
                    y = cy;
                    break;
                case Y_SCAN:
                    x = cx;
                    y = cy + d * std::sin(phase);
                    break;
                case WAVE:
                    if (hasWaveform.load())
                    {
                        float norm = std::fmod(phase, PI2) / PI2;
                        int pos = (int)(norm * 2048) % 2048;
                        x = cx + d * waveformLFO[pos];
                        y = cy;
                        break;
                    }
                    [[fallthrough]];
                case WALK:
                default:
                {
                    constexpr float theta = 0.08f;
                    float sigma = d * 0.25f;
                    walkX += theta * (cx - walkX) + sigma * gauss(rng);
                    walkY += theta * (cy - walkY) + sigma * gauss(rng);
                    x = walkX;  y = walkY;
                    break;
                }
            }

            x = std::clamp(x, -4.f, 4.f);
            y = std::clamp(y, -4.f, 4.f);

            if (onUpdate) onUpdate(x, y);

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
};
