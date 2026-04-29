/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

#include <filesystem>
#include <memory>
#include <print>
#include <utility>
#include <variant>

namespace {

    bool check_cuda_driver_version() {
        const auto info = lfs::core::check_cuda_version();
        if (info.query_failed) {
            LOG_WARN("Failed to query CUDA driver version");
            return true;
        }

        LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
        if (!info.supported) {
            LOG_WARN("CUDA {}.{} unsupported. Requires 12.8+ (driver 570+)", info.major, info.minor);
            return false;
        }
        return true;
    }

    void configure_image_loader(const lfs::core::param::TrainingParameters& params) {
        lfs::io::CacheLoader::getInstance(
            params.dataset.loading_params.use_cpu_memory,
            params.dataset.loading_params.use_fs_cache);

        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8});
        });
    }

    int train_from_params(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        if (!params) {
            LOG_ERROR("No training parameters provided");
            return 1;
        }

        params->optimization.headless = true;
        configure_image_loader(*params);
        check_cuda_driver_version();

        lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

        {
            lfs::core::Scene scene;

            if (params->resume_checkpoint) {
                LOG_INFO("Resuming from checkpoint: {}", lfs::core::path_to_utf8(*params->resume_checkpoint));

                auto checkpoint_params_result = lfs::core::load_checkpoint_params(*params->resume_checkpoint);
                if (!checkpoint_params_result) {
                    LOG_ERROR("Failed to load checkpoint params: {}", checkpoint_params_result.error());
                    return 1;
                }

                auto checkpoint_params = std::move(*checkpoint_params_result);
                if (!params->dataset.data_path.empty()) {
                    checkpoint_params.dataset.data_path = params->dataset.data_path;
                }
                if (!params->dataset.output_path.empty()) {
                    checkpoint_params.dataset.output_path = params->dataset.output_path;
                }
                if (!params->dataset.output_name.empty()) {
                    checkpoint_params.dataset.output_name = params->dataset.output_name;
                }
                checkpoint_params.optimization.headless = true;

                if (checkpoint_params.dataset.data_path.empty()) {
                    LOG_ERROR("Checkpoint has no dataset path and none was provided with --data-path");
                    return 1;
                }
                if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                    LOG_ERROR("Dataset path does not exist: {}",
                              lfs::core::path_to_utf8(checkpoint_params.dataset.data_path));
                    return 1;
                }

                if (const auto result = lfs::training::validateDatasetPath(checkpoint_params); !result) {
                    LOG_ERROR("Dataset validation failed: {}", result.error());
                    return 1;
                }
                if (const auto result = lfs::training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                    LOG_ERROR("Failed to load training data: {}", result.error());
                    return 1;
                }

                for (const auto* node : scene.getNodes()) {
                    if (node->type == lfs::core::NodeType::POINTCLOUD) {
                        scene.removeNode(node->name, false);
                        break;
                    }
                }

                auto splat_result = lfs::core::load_checkpoint_splat_data(*params->resume_checkpoint);
                if (!splat_result) {
                    LOG_ERROR("Failed to load checkpoint splat data: {}", splat_result.error());
                    return 1;
                }

                auto splat_data = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                scene.addSplat("Model", std::move(splat_data), lfs::core::NULL_NODE);
                scene.setTrainingModelNode("Model");
                checkpoint_params.resume_checkpoint = *params->resume_checkpoint;

                lfs::training::Trainer trainer(scene);
                if (const auto result = trainer.initialize(checkpoint_params); !result) {
                    LOG_ERROR("Failed to initialize trainer: {}", result.error());
                    return 1;
                }

                const auto restored_iteration = trainer.load_checkpoint(*params->resume_checkpoint);
                if (!restored_iteration) {
                    LOG_ERROR("Failed to restore checkpoint state: {}", restored_iteration.error());
                    return 1;
                }
                LOG_INFO("Resumed from iteration {}", *restored_iteration);

                lfs::core::Tensor::trim_memory_pool();
                if (const auto result = trainer.train(); !result) {
                    LOG_ERROR("Training error: {}", result.error());
                    return 1;
                }
            } else {
                if (params->dataset.data_path.empty()) {
                    LOG_ERROR("Training requires --data-path");
                    return 1;
                }
                if (params->dataset.output_path.empty()) {
                    LOG_ERROR("Training requires --output-path");
                    return 1;
                }

                LOG_INFO("Starting training-only headless run");

                if (const auto result = lfs::training::loadTrainingDataIntoScene(*params, scene); !result) {
                    LOG_ERROR("Failed to load training data: {}", result.error());
                    return 1;
                }
                if (const auto result = lfs::training::initializeTrainingModel(*params, scene); !result) {
                    LOG_ERROR("Failed to initialize model: {}", result.error());
                    return 1;
                }

                lfs::training::Trainer trainer(scene);
                if (const auto result = trainer.initialize(*params); !result) {
                    LOG_ERROR("Failed to initialize trainer: {}", result.error());
                    return 1;
                }

                lfs::core::Tensor::trim_memory_pool();
                if (const auto result = trainer.train(); !result) {
                    LOG_ERROR("Training error: {}", result.error());
                    return 1;
                }
            }

            LOG_INFO("Headless training completed");
        }

        lfs::core::Tensor::shutdown_memory_pool();
        lfs::core::PinnedMemoryAllocator::instance().shutdown();
        return 0;
    }

} // namespace

int main(int argc, char* argv[]) {
    auto parsed = lfs::core::args::parse_args(argc, argv);
    if (!parsed) {
        std::println(stderr, "Error: {}", parsed.error());
        return 1;
    }

    return std::visit([](auto&& mode) -> int {
        using T = std::decay_t<decltype(mode)>;
        if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
            return train_from_params(std::move(mode.params));
        } else if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
            return 0;
        } else {
            std::println(stderr, "train_colmap_headless only supports training arguments");
            return 1;
        }
    },
                      std::move(*parsed));
}
