/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mode_switcher_keyboard.hpp"

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include "holoscan/logger/logger.hpp"

namespace atracsys::ops {

void ModeSwitcherKeyboard::start(bool enabled) {
  stop();
  if (!enabled) { return; }

  keyboard_fd_ = STDIN_FILENO;
  if (!isatty(keyboard_fd_)) {
    keyboard_fd_ = ::open("/dev/tty", O_RDONLY | O_NONBLOCK);
    if (keyboard_fd_ < 0) {
      HOLOSCAN_LOG_WARN("Mode switcher keyboard enabled but no TTY available; skipping");
      return;
    }
  }

  orig_stdin_flags_ = fcntl(keyboard_fd_, F_GETFL, 0);
  if (orig_stdin_flags_ != -1) {
    fcntl(keyboard_fd_, F_SETFL, orig_stdin_flags_ | O_NONBLOCK);
  }

  if (tcgetattr(keyboard_fd_, &orig_termios_) == 0) {
    termios raw = orig_termios_;
    raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(keyboard_fd_, TCSANOW, &raw) == 0) { termios_configured_ = true; }
  }

  if (!termios_configured_) {
    HOLOSCAN_LOG_WARN(
        "Mode switcher keyboard enabled but terminal raw mode failed; using best-effort reads");
  }
}

void ModeSwitcherKeyboard::stop() {
  if (keyboard_fd_ >= 0 && orig_stdin_flags_ != -1) {
    fcntl(keyboard_fd_, F_SETFL, orig_stdin_flags_);
  }
  if (termios_configured_ && keyboard_fd_ >= 0) { tcsetattr(keyboard_fd_, TCSANOW, &orig_termios_); }
  if (keyboard_fd_ >= 0 && keyboard_fd_ != STDIN_FILENO) { ::close(keyboard_fd_); }

  keyboard_fd_ = -1;
  orig_stdin_flags_ = -1;
  termios_configured_ = false;
}

std::optional<char> ModeSwitcherKeyboard::poll_key() {
  if (keyboard_fd_ < 0) { return std::nullopt; }

  unsigned char last_char = 0;
  bool have_char = false;

  pollfd pfd{};
  pfd.fd = keyboard_fd_;
  pfd.events = POLLIN;
  const int ret = poll(&pfd, 1, 0);
  if (ret <= 0 || !(pfd.revents & POLLIN)) { return std::nullopt; }

  unsigned char buf = 0;
  while (::read(keyboard_fd_, &buf, 1) == 1) {
    last_char = buf;
    have_char = true;
  }

  if (!have_char) { return std::nullopt; }
  return static_cast<char>(last_char);
}

}  // namespace atracsys::ops
