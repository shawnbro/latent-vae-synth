cmake_minimum_required(VERSION 3.22)
project(LatentSynth VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# JUCE — fetched automatically at configure time
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.4)
FetchContent_MakeAvailable(juce)

# ---------------------------------------------------------------------------
# ONNX Runtime — prebuilt binaries
#
# Download from: https://github.com/microsoft/onnxruntime/releases
# e.g. onnxruntime-osx-arm64-1.20.1.tgz  → extract to vendor/
#
# Usage:
#   cmake -B build \
#         -DONNXRUNTIME_DIR="${PWD}/vendor/onnxruntime-osx-arm64-1.20.1" \
#         -DCMAKE_BUILD_TYPE=Release
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime-osx-arm64-1.20.1"
    CACHE PATH "Path to the extracted ONNX Runtime package")

find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib"
    REQUIRED)

message(STATUS "ONNX Runtime lib: ${ORT_LIB}")

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
juce_add_plugin(LatentSynth
    PLUGIN_MANUFACTURER_CODE Lsyn
    PLUGIN_CODE             LtSn
    FORMATS                 AU VST3 Standalone
    PRODUCT_NAME            "Latent Synth"
    COMPANY_NAME            "Latent Synth"
    BUNDLE_ID               com.latentsynth.plugin
    PLUGIN_AU_MAIN_TYPE     kAudioUnitType_MusicDevice
    IS_SYNTH                TRUE
    NEEDS_MIDI_INPUT        TRUE
    NEEDS_MIDI_OUTPUT       FALSE
    IS_MIDI_EFFECT          FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    VST3_CATEGORIES         "Instrument" "Synth"
)

# ---------------------------------------------------------------------------
# Embed assets (decoder.onnx + latent_index.npz) directly in the binary
# ---------------------------------------------------------------------------
juce_add_binary_data(LatentSynthAssets
    SOURCES
        Assets/decoder.onnx
        Assets/latent_index.npz)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
target_sources(LatentSynth PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/dsp/LatentSynth.cpp)

target_include_directories(LatentSynth PRIVATE
    Source
    "${ONNXRUNTIME_DIR}/include")

# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------
target_link_libraries(LatentSynth PRIVATE
    LatentSynthAssets
    juce::juce_audio_utils
    juce::juce_dsp
    "${ORT_LIB}")

target_compile_definitions(LatentSynth PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

# ---------------------------------------------------------------------------
# Bundle libonnxruntime inside each plugin artefact so it is self-contained.
# Uses @loader_path so the dylib is found relative to the plugin binary.
# ---------------------------------------------------------------------------
set(ORT_DYLIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.1.20.1.dylib")

foreach(_fmt VST3 AU Standalone)
    string(TOLOWER "${_fmt}" _fmt_lc)
    if(_fmt STREQUAL "VST3")
        set(_binary_dir "$<TARGET_BUNDLE_CONTENT_DIR:LatentSynth_VST3>/MacOS")
        set(_target LatentSynth_VST3)
    elseif(_fmt STREQUAL "AU")
        set(_binary_dir "$<TARGET_BUNDLE_CONTENT_DIR:LatentSynth_AU>/MacOS")
        set(_target LatentSynth_AU)
    else()
        set(_binary_dir "$<TARGET_BUNDLE_CONTENT_DIR:LatentSynth_Standalone>/MacOS")
        set(_target LatentSynth_Standalone)
    endif()

    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DYLIB}" "${_binary_dir}/"
        COMMAND install_name_tool -change
            "@rpath/libonnxruntime.1.20.1.dylib"
            "@loader_path/libonnxruntime.1.20.1.dylib"
            "${_binary_dir}/Latent Synth"
        COMMAND codesign --force --sign - --deep "$<TARGET_BUNDLE_DIR:${_target}>"
        COMMENT "Bundling libonnxruntime and re-signing ${_fmt}"
        VERBATIM)
endforeach()

# ---------------------------------------------------------------------------
# Install helpers (macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    set(AU_INSTALL_DIR   "$ENV{HOME}/Library/Audio/Plug-Ins/Components")
    set(VST3_INSTALL_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")

    install(DIRECTORY
        "$<TARGET_FILE_DIR:LatentSynth_AU>/../LatentSynth.component"
        DESTINATION "${AU_INSTALL_DIR}")

    install(DIRECTORY
        "$<TARGET_FILE_DIR:LatentSynth_VST3>/../LatentSynth.vst3"
        DESTINATION "${VST3_INSTALL_DIR}")
endif()
