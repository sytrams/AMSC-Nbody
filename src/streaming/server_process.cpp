#include "streaming/server_process.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(__unix__) && !defined(__APPLE__)
#error "Automatic N-body stream server startup requires a POSIX platform"
#endif

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace nbody::streaming {

ServerProcess::ServerProcess(
    const std::filesystem::path &executable, const ServerConfig &config,
    const std::optional<RelaySourceConfig> &relayConfig, std::ostream &log) {
  if (executable.empty())
    throw std::invalid_argument("Streaming server executable path is empty");

  const std::string executableText = executable.string();
  const std::string spoolText = config.spoolDirectory.string();
  const std::string portText = std::to_string(config.port);
  const std::string pollText = std::to_string(config.pollInterval.count());

  std::vector<std::string> arguments{executableText, "--spool-dir", spoolText,
                                     "--poll-ms", pollText};
  if (relayConfig) {
    arguments.insert(arguments.end(),
                     {"--relay-host", relayConfig->host, "--relay-port",
                      std::to_string(relayConfig->port), "--relay-token-file",
                      relayConfig->tokenFile.string(), "--reconnect-ms",
                      std::to_string(relayConfig->reconnectInterval.count())});
  } else {
    arguments.insert(arguments.end(),
                     {"--bind", config.bindAddress, "--port", portText});
  }

  const pid_t child = ::fork();
  if (child < 0)
    throw std::runtime_error("Could not fork streaming server: " +
                             std::string(std::strerror(errno)));

  if (child == 0) {
    std::vector<char *> argumentPointers;
    argumentPointers.reserve(arguments.size() + 1);
    for (auto &argument : arguments)
      argumentPointers.push_back(argument.data());
    argumentPointers.push_back(nullptr);
    ::execv(executableText.c_str(), argumentPointers.data());
    const std::string message = "Could not start N-body stream server '" +
                                executableText + "': " + std::strerror(errno) +
                                "\n";
    (void)::write(STDERR_FILENO, message.data(), message.size());
    ::_exit(127);
  }

  processId_ = static_cast<int>(child);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  int status = 0;
  const pid_t result = ::waitpid(child, &status, WNOHANG);
  if (result == child) {
    processId_ = -1;
    throw std::runtime_error(
        "N-body stream server exited during startup (status " +
        std::to_string(status) + ")");
  }
  if (result < 0) {
    processId_ = -1;
    throw std::runtime_error("Could not verify streaming server startup: " +
                             std::string(std::strerror(errno)));
  }

  log << "Started N-body stream server (pid " << processId_ << ")\n";
}

ServerProcess::~ServerProcess() noexcept {
  if (processId_ < 0)
    return;

  const pid_t child = static_cast<pid_t>(processId_);
  int status = 0;
  const pid_t state = ::waitpid(child, &status, WNOHANG);
  if (state == child)
    return;
  if (state < 0 && errno == ECHILD)
    return;

  (void)::kill(child, SIGTERM);
  for (int attempt = 0; attempt < 20; ++attempt) {
    const pid_t result = ::waitpid(child, &status, WNOHANG);
    if (result == child || (result < 0 && errno == ECHILD))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  (void)::kill(child, SIGKILL);
  (void)::waitpid(child, &status, 0);
}

std::filesystem::path
defaultServerExecutable(const std::filesystem::path &simulationExecutable) {
  std::filesystem::path runningExecutable = simulationExecutable;
#if defined(__linux__)
  std::array<char, 4096> executablePath{};
  const ssize_t pathBytes = ::readlink("/proc/self/exe", executablePath.data(),
                                       executablePath.size());
  if (pathBytes > 0)
    runningExecutable =
        std::string(executablePath.data(), static_cast<std::size_t>(pathBytes));
#elif defined(__APPLE__)
  std::uint32_t pathBytes = 0;
  (void)::_NSGetExecutablePath(nullptr, &pathBytes);
  std::vector<char> executablePath(pathBytes);
  if (::_NSGetExecutablePath(executablePath.data(), &pathBytes) == 0)
    runningExecutable = executablePath.data();
#endif

  std::error_code error;
  std::filesystem::path absolute =
      std::filesystem::absolute(runningExecutable, error);
  if (error)
    absolute = runningExecutable;
  return absolute.parent_path() / "nbody_stream_server";
}

} // namespace nbody::streaming
