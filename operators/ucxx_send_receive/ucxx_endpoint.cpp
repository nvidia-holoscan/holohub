/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ucxx_endpoint.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <stdexcept>
#include <vector>

namespace holoscan::ops {

void UcxxEndpoint::add_close_callback(std::function<void(ucs_status_t)> callback) {
  std::scoped_lock lock(close_callbacks_mutex_);
  close_callbacks_.push_back(std::move(callback));
}

UcxxEndpoint::~UcxxEndpoint() {
  stop_listen_.store(true, std::memory_order_release);
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (listen_thread_.joinable()) {
    listen_thread_.join();
  }
  std::atomic_store(&endpoint_, std::shared_ptr<::ucxx::Endpoint>{});
  if (worker_) {
    worker_->stopProgressThread();
  }
}

void UcxxEndpoint::setup(holoscan::ComponentSpec& spec) {
  spec.param(
      hostname_, "hostname", "Hostname", "Hostname of the endpoint", std::string("127.0.0.1"));
  spec.param(port_, "port", "Port", "Port of the endpoint", 50008);
  spec.param(listen_,
             "listen",
             "Listen",
             "Whether to listen for connections (server), or initiate a connection (client)",
             false);

  is_alive_condition_ = fragment()->make_condition<holoscan::AsynchronousCondition>(
      fmt::format("{}_is_alive", name()));
}

namespace {

int create_listen_socket(const std::string& hostname, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));

  if (hostname.empty() || hostname == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else {
    if (::inet_pton(AF_INET, hostname.c_str(), &addr.sin_addr) != 1) {
      ::close(fd);
      return -1;
    }
  }

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  if (::listen(fd, 8) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int connect_socket(const std::string& hostname, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, hostname.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return -1;
  }

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

bool send_all(int fd, const void* data, size_t size) {
  const auto* buffer = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < size) {
    const ssize_t bytes_sent = ::send(fd, buffer + sent, size - sent, 0);
    if (bytes_sent < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    if (bytes_sent == 0) { return false; }
    sent += static_cast<size_t>(bytes_sent);
  }
  return true;
}

bool recv_all(int fd, void* data, size_t size) {
  auto* buffer = static_cast<uint8_t*>(data);
  size_t received = 0;
  while (received < size) {
    const ssize_t bytes_received = ::recv(fd, buffer + received, size - received, 0);
    if (bytes_received < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    if (bytes_received == 0) { return false; }
    received += static_cast<size_t>(bytes_received);
  }
  return true;
}

// Set SO_RCVTIMEO and SO_SNDTIMEO on a socket fd to bound blocking I/O during handshake.
void set_handshake_timeout(int socket_fd, int timeout_seconds) {
  struct timeval timeout{};
  timeout.tv_sec = timeout_seconds;
  timeout.tv_usec = 0;
  ::setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  ::setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool send_worker_address(int fd, const std::string& address) {
  const uint64_t len = static_cast<uint64_t>(address.size());
  return send_all(fd, &len, sizeof(len)) && send_all(fd, address.data(), address.size());
}

std::string recv_worker_address(int fd) {
  constexpr uint64_t kMaxAddressLen = 64 * 1024;
  uint64_t len = 0;
  if (!recv_all(fd, &len, sizeof(len))) {
    return {};
  }
  if (len == 0 || len > kMaxAddressLen) {
    return {};
  }
  std::string address(len, '\0');
  if (!recv_all(fd, address.data(), address.size())) {
    return {};
  }
  return address;
}

}  // namespace

void UcxxEndpoint::activate_endpoint(std::shared_ptr<::ucxx::Endpoint> ep) {
  auto id = endpoint_id_.fetch_add(1, std::memory_order_relaxed) + 1;
  ep->setCloseCallback(
      [this, id](ucs_status_t status, std::shared_ptr<void>) {
        on_endpoint_closed(status, id);
      },
      nullptr);

  {
    std::scoped_lock lock(endpoint_mutex_);
    std::atomic_store(&endpoint_, ep);
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
  }
  HOLOSCAN_LOG_INFO("Endpoint connected ({}:{}, id={})", hostname_.get(), port_.get(), id);
}

void UcxxEndpoint::initialize() {
  if (is_initialized_) {
    return;
  }
  holoscan::Resource::initialize();

  context_ = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
  worker_ = context_->createWorker();
  worker_->startProgressThread(/*pollingMode=*/false);

  is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);

  if (listen_) {
    listen_fd_ = create_listen_socket(hostname_.get(), port_.get());
    const int saved_errno = errno;
    if (listen_fd_ < 0) {
      HOLOSCAN_LOG_ERROR("Failed to open control socket on {}:{} (errno: {})",
                         hostname_.get(),
                         port_.get(),
                         saved_errno);
      throw std::runtime_error(fmt::format("Failed to open control socket on {}:{} (errno: {})",
                                           hostname_.get(),
                                           port_.get(),
                                           saved_errno));
    }

    HOLOSCAN_LOG_INFO("Listening on {}:{}", hostname_.get(), port_.get());
    listen_thread_ = std::thread([this]() {
      const std::string local_address_str = worker_->getAddress()->getString();

      while (!stop_listen_.load(std::memory_order_acquire)) {
        int client_fd = ::accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
          if (stop_listen_.load(std::memory_order_acquire)) {
            break;
          }
          continue;
        }

        set_handshake_timeout(client_fd, 5);
        std::string remote_address_str = recv_worker_address(client_fd);
        if (remote_address_str.empty() || !send_worker_address(client_fd, local_address_str)) {
          ::close(client_fd);
          continue;
        }
        ::close(client_fd);

        auto remote_address = ::ucxx::createAddressFromString(remote_address_str);
        activate_endpoint(worker_->createEndpointFromWorkerAddress(remote_address, true));
      }
    });
  } else {
    int fd = connect_socket(hostname_.get(), port_.get());
    const int saved_errno = errno;
    if (fd < 0) {
      HOLOSCAN_LOG_ERROR("Failed to connect control socket to {}:{} (errno: {})",
                         hostname_.get(),
                         port_.get(),
                         saved_errno);
      throw std::runtime_error(fmt::format("Failed to connect control socket to {}:{} (errno: {})",
                                           hostname_.get(),
                                           port_.get(),
                                           saved_errno));
    }

    set_handshake_timeout(fd, 5);
    const std::string local_address_str = worker_->getAddress()->getString();
    if (!send_worker_address(fd, local_address_str)) {
      ::close(fd);
      HOLOSCAN_LOG_ERROR("Failed to send worker address to {}:{}", hostname_.get(), port_.get());
      throw std::runtime_error(
          fmt::format("Failed to send worker address to {}:{}", hostname_.get(), port_.get()));
    }

    std::string remote_address_str = recv_worker_address(fd);
    ::close(fd);
    if (remote_address_str.empty()) {
      HOLOSCAN_LOG_ERROR(
          "Failed to receive worker address from {}:{}", hostname_.get(), port_.get());
      throw std::runtime_error(
          fmt::format("Failed to receive worker address from {}:{}", hostname_.get(), port_.get()));
    }

    auto remote_address = ::ucxx::createAddressFromString(remote_address_str);
    activate_endpoint(worker_->createEndpointFromWorkerAddress(remote_address, true));
  }
}

void UcxxEndpoint::on_endpoint_closed(ucs_status_t status, uint64_t id) {
  {
    std::scoped_lock lock(endpoint_mutex_);
    if (endpoint_id_.load(std::memory_order_relaxed) != id) { return; }

    // Clear the endpoint so operators can quickly detect disconnection.
    std::atomic_store(&endpoint_, std::shared_ptr<::ucxx::Endpoint>{});

    // Prevent operators from executing until a new connection is established (server mode)
    // or indefinitely (client mode).
    if (listen_) {
      is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    } else {
      is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }
  }

  HOLOSCAN_LOG_INFO("Endpoint closed ({}:{}, id={})", hostname_.get(), port_.get(), id);
  if (status != UCS_OK) {
    // These are expected when subscriber disconnects/restarts.
    if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_NOT_CONNECTED ||
        status == UCS_ERR_UNREACHABLE || status == UCS_ERR_CANCELED) {
      HOLOSCAN_LOG_WARN("Endpoint closed ({}:{}, id={}) with status: {}",
                        hostname_.get(),
                        port_.get(),
                        id,
                        ucs_status_string(status));
    } else {
      HOLOSCAN_LOG_ERROR("Endpoint closed ({}:{}, id={}) with status: {}",
                         hostname_.get(),
                         port_.get(),
                         id,
                         ucs_status_string(status));
    }
  }

  // Notify any registered callbacks. (May be invoked from UCXX progress thread.)
  {
    std::vector<std::function<void(ucs_status_t)>> callbacks_copy;
    {
      std::scoped_lock lock(close_callbacks_mutex_);
      callbacks_copy = close_callbacks_;
    }
    for (auto& cb : callbacks_copy) {
      if (cb) {
        cb(status);
      }
    }
  }
}

}  // namespace holoscan::ops
