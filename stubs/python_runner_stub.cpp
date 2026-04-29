/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/runner.hpp"

#include <expected>
#include <utility>

namespace lfs::python {

    namespace {
        std::function<void(const std::string&, bool)> output_callback;
    }

    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts) {
        if (!scripts.empty()) {
            return std::unexpected("Python scripts are disabled in the standalone headless training build");
        }
        return {};
    }

    void set_output_callback(std::function<void(const std::string&, bool)> callback) {
        output_callback = std::move(callback);
    }

    void write_output(const std::string& text, bool is_error) {
        if (output_callback) {
            output_callback(text, is_error);
        }
    }

    void ensure_initialized() {}
    void ensure_builtin_ui_registered() {}
    void ensure_plugins_loaded() {}
    void preload_user_plugins_async() {}
    void join_plugin_preload() {}
    void start_embedded_repl(int, int) {}
    void stop_embedded_repl() {}
    bool start_debugpy(int) { return false; }
    void install_output_redirect() {}
    void finalize() {}
    bool was_python_used() { return false; }

    FormatResult format_python_code(const std::string& code) {
        return {.code = code, .success = true};
    }

    FormatResult clean_python_code(const std::string& code) {
        return {.code = code, .success = true};
    }

    void set_frame_callback(std::function<void(float)>) {}
    void clear_frame_callback() {}
    void tick_frame_callback(float) {}
    bool has_frame_callback() { return false; }

    std::filesystem::path get_user_packages_dir() {
        return {};
    }

    void update_python_path() {}

    CapabilityResult invoke_capability(const std::string&, const std::string&) {
        return {.success = false, .error = "Python capabilities are disabled in the standalone headless training build"};
    }

    bool has_capability(const std::string&) {
        return false;
    }

    std::vector<CapabilityInfo> list_capabilities() {
        return {};
    }

} // namespace lfs::python
