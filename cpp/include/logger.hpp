#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace rfdetr {

/**
 * @brief Minimal header-only logger for the RF-DETR library.
 *
 * Provides INFO, WARN, and ERR macros that write to stdout/stderr with a
 * consistent "[RFDETR][LEVEL] message" prefix. No external dependencies.
 *
 * Usage:
 *   LOG_INFO("Session created on " << provider);
 *   LOG_WARN("Provider " << name << " unavailable, falling back");
 *   LOG_ERR("Failed to load model: " << e.what());
 */

namespace detail {

inline std::string timestamp() {
  using clock = std::chrono::system_clock;
  auto now = clock::now();
  std::time_t t = clock::to_time_t(now);
  std::tm tm_buf{};
  localtime_r(&t, &tm_buf);
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%H:%M:%S");
  return oss.str();
}

} // namespace detail

} // namespace rfdetr

// clang-format off
#define LOG_INFO(msg)                                                          \
  do {                                                                         \
    std::cout << "[RFDETR][" << rfdetr::detail::timestamp() << "][INFO]  "    \
              << msg << "\n";                                                  \
  } while (0)

#define LOG_WARN(msg)                                                          \
  do {                                                                         \
    std::cerr << "[RFDETR][" << rfdetr::detail::timestamp() << "][WARN]  "    \
              << msg << "\n";                                                  \
  } while (0)

#define LOG_ERR(msg)                                                           \
  do {                                                                         \
    std::cerr << "[RFDETR][" << rfdetr::detail::timestamp() << "][ERROR] "    \
              << msg << "\n";                                                  \
  } while (0)
// clang-format on
