#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "streaming/frame.hpp"
#include "streaming/network.hpp"

namespace {

using namespace std::chrono_literals;

std::filesystem::path makeNetworkTestDirectory() {
  std::random_device randomDevice;
  const auto token = (static_cast<std::uint64_t>(randomDevice()) << 32U) |
                     static_cast<std::uint64_t>(randomDevice());
  const auto directory = std::filesystem::temp_directory_path() /
                         ("nbody-network-test-" + std::to_string(token));
  std::filesystem::create_directories(directory);
  return directory;
}

std::uint16_t reserveEphemeralPort() {
  const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
  if (descriptor < 0)
    throw std::runtime_error("Could not create test socket");

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(descriptor, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0) {
    ::close(descriptor);
    throw std::runtime_error("Could not bind test socket");
  }

  socklen_t addressBytes = sizeof(address);
  if (::getsockname(descriptor, reinterpret_cast<sockaddr *>(&address),
                    &addressBytes) != 0) {
    ::close(descriptor);
    throw std::runtime_error("Could not inspect test socket");
  }
  const std::uint16_t port = ntohs(address.sin_port);
  ::close(descriptor);
  return port;
}

void sendTestBytes(int descriptor, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const char *>(data);
  while (bytes > 0) {
#if defined(MSG_NOSIGNAL)
    const ssize_t sent = ::send(descriptor, cursor, bytes, MSG_NOSIGNAL);
#else
    const ssize_t sent = ::send(descriptor, cursor, bytes, 0);
#endif
    if (sent <= 0)
      throw std::runtime_error("Test client could not send protocol bytes");
    cursor += sent;
    bytes -= static_cast<std::size_t>(sent);
  }
}

void receiveTestBytes(int descriptor, void *data, std::size_t bytes) {
  auto *cursor = static_cast<char *>(data);
  while (bytes > 0) {
    const ssize_t received = ::recv(descriptor, cursor, bytes, 0);
    if (received <= 0)
      throw std::runtime_error("Test client peer disconnected");
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

void sendTestU32(int descriptor, std::uint32_t value) {
  value = htonl(value);
  sendTestBytes(descriptor, &value, sizeof(value));
}

std::uint32_t receiveTestU32(int descriptor) {
  std::uint32_t value = 0;
  receiveTestBytes(descriptor, &value, sizeof(value));
  return ntohl(value);
}

void sendTestU64(int descriptor, std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<unsigned char>(value >> ((7U - index) * 8U));
  sendTestBytes(descriptor, bytes.data(), bytes.size());
}

std::uint64_t receiveTestU64(int descriptor) {
  std::array<unsigned char, 8> bytes{};
  receiveTestBytes(descriptor, bytes.data(), bytes.size());
  std::uint64_t value = 0;
  for (const unsigned char byte : bytes)
    value = (value << 8U) | byte;
  return value;
}

int connectTestClient(std::uint16_t port) {
  for (int attempt = 0; attempt < 100; ++attempt) {
    const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
    if (descriptor < 0)
      throw std::runtime_error("Could not create test client socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (::connect(descriptor, reinterpret_cast<sockaddr *>(&address),
                  sizeof(address)) == 0)
      return descriptor;
    ::close(descriptor);
    std::this_thread::sleep_for(10ms);
  }
  throw std::runtime_error("Could not connect delayed-ACK test client");
}

class StreamingNetworkTest : public ::testing::Test {
protected:
  void SetUp() override { directory = makeNetworkTestDirectory(); }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }

  std::filesystem::path directory;
};

TEST_F(StreamingNetworkTest,
       AuthenticatedRelayTransfersAndAcknowledgesDurableFrame) {
  const auto spoolDirectory = directory / "spool";
  const auto outputDirectory = directory / "received";
  const auto tokenFile = directory / "relay.token";
  {
    std::ofstream token(tokenFile);
    token
        << "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
  }
  std::filesystem::permissions(tokenFile,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::replace);

  nbody::streaming::FrameSpool spool(spoolDirectory, "relay-test");
  const auto queuedFrame = spool.writePositions(
      0, 17, 0.25, 2, [](std::size_t, std::size_t count, float *positions) {
        for (std::size_t index = 0; index < count; ++index) {
          positions[index * 3] = static_cast<float>(index) + 1.0F;
          positions[index * 3 + 1] = static_cast<float>(index) + 2.0F;
          positions[index * 3 + 2] = static_cast<float>(index) + 3.0F;
        }
      });

  const std::uint16_t sourcePort = reserveEphemeralPort();
  std::uint16_t clientPort = reserveEphemeralPort();
  while (clientPort == sourcePort)
    clientPort = reserveEphemeralPort();

  nbody::streaming::RelayConfig relayConfig;
  relayConfig.sourceBindAddress = "127.0.0.1";
  relayConfig.sourcePort = sourcePort;
  relayConfig.clientBindAddress = "127.0.0.1";
  relayConfig.clientPort = clientPort;
  relayConfig.tokenFile = tokenFile;

  nbody::streaming::ServerConfig serverConfig;
  serverConfig.spoolDirectory = spoolDirectory;
  serverConfig.pollInterval = 10ms;

  nbody::streaming::RelaySourceConfig sourceConfig;
  sourceConfig.host = "127.0.0.1";
  sourceConfig.port = sourcePort;
  sourceConfig.tokenFile = tokenFile;
  sourceConfig.reconnectInterval = 10ms;

  std::atomic<bool> stop{false};
  std::ostringstream relayLog;
  std::ostringstream sourceLog;
  std::exception_ptr relayFailure;
  std::exception_ptr sourceFailure;

  std::thread relay([&] {
    try {
      nbody::streaming::runRelay(
          relayConfig, [&] { return stop.load(); }, relayLog);
    } catch (...) {
      relayFailure = std::current_exception();
    }
  });
  std::this_thread::sleep_for(25ms);
  std::thread source([&] {
    try {
      nbody::streaming::runRelaySource(
          serverConfig, sourceConfig, [&] { return stop.load(); }, sourceLog);
    } catch (...) {
      sourceFailure = std::current_exception();
    }
  });

  std::size_t received = 0;
  std::exception_ptr clientFailure;
  for (int attempt = 0; attempt < 100 && received == 0; ++attempt) {
    try {
      nbody::streaming::ClientConfig clientConfig;
      clientConfig.host = "127.0.0.1";
      clientConfig.port = clientPort;
      clientConfig.outputDirectory = outputDirectory;
      clientConfig.follow = false;
      std::ostringstream clientLog;
      received = nbody::streaming::downloadFrames(
          clientConfig, [&] { return stop.load(); }, clientLog);
      clientFailure = nullptr;
    } catch (...) {
      clientFailure = std::current_exception();
      std::this_thread::sleep_for(10ms);
    }
  }

  stop = true;
  source.join();
  relay.join();

  if (clientFailure) {
    try {
      std::rethrow_exception(clientFailure);
    } catch (const std::exception &error) {
      FAIL() << "Client failed: " << error.what() << "\nRelay log:\n"
             << relayLog.str() << "\nSource log:\n"
             << sourceLog.str();
    }
  }
  if (sourceFailure) {
    try {
      std::rethrow_exception(sourceFailure);
    } catch (const std::exception &error) {
      FAIL() << "Source failed: " << error.what() << "\nRelay log:\n"
             << relayLog.str() << "\nSource log:\n"
             << sourceLog.str();
    }
  }
  if (relayFailure) {
    try {
      std::rethrow_exception(relayFailure);
    } catch (const std::exception &error) {
      FAIL() << "Relay failed: " << error.what() << "\nRelay log:\n"
             << relayLog.str() << "\nSource log:\n"
             << sourceLog.str();
    }
  }

  EXPECT_EQ(received, 1U);
  EXPECT_FALSE(std::filesystem::exists(queuedFrame));
  const auto receivedFrame = outputDirectory / queuedFrame.filename();
  ASSERT_TRUE(std::filesystem::exists(receivedFrame));
  const auto header = nbody::streaming::readFrameHeader(receivedFrame);
  EXPECT_EQ(header.sequence, 0U);
  EXPECT_EQ(header.simulationStep, 17U);
  EXPECT_DOUBLE_EQ(header.simulationTime, 0.25);
  std::ifstream latestFrame(outputDirectory / ".nbody-latest");
  std::string latestName;
  ASSERT_TRUE(static_cast<bool>(std::getline(latestFrame, latestName)));
  EXPECT_EQ(latestName, queuedFrame.filename().string());
  EXPECT_NE(relayLog.str().find("Authenticated relay source"),
            std::string::npos);
  EXPECT_NE(sourceLog.str().find("Relay paired a client"), std::string::npos);
}

TEST_F(StreamingNetworkTest, WaitsForDelayedFrameAcknowledgement) {
  const auto spoolDirectory = directory / "spool";
  nbody::streaming::FrameSpool spool(spoolDirectory, "delayed-ack");
  const auto queuedFrame = spool.writePositions(
      0, 3, 1.5, 1, [](std::size_t, std::size_t, float *positions) {
        positions[0] = 1.0F;
        positions[1] = 2.0F;
        positions[2] = 3.0F;
      });

  nbody::streaming::ServerConfig config;
  config.spoolDirectory = spoolDirectory;
  config.bindAddress = "127.0.0.1";
  config.port = reserveEphemeralPort();
  config.pollInterval = 10ms;

  std::atomic<bool> stop{false};
  std::ostringstream serverLog;
  std::exception_ptr serverFailure;
  std::thread server([&] {
    try {
      nbody::streaming::runServer(config, [&] { return stop.load(); },
                                  serverLog);
    } catch (...) {
      serverFailure = std::current_exception();
    }
  });

  std::exception_ptr clientFailure;
  try {
    const int descriptor = connectTestClient(config.port);
    const std::array<char, 8> magic{'N', 'B', 'S', 'T',
                                    'R', 'M', '0', '1'};
    sendTestBytes(descriptor, magic.data(), magic.size());
    const std::uint8_t backlogMode = 0;
    sendTestBytes(descriptor, &backlogMode, sizeof(backlogMode));

    const std::uint32_t type = receiveTestU32(descriptor);
    const std::uint32_t nameBytes = receiveTestU32(descriptor);
    const std::uint64_t contentBytes = receiveTestU64(descriptor);
    std::string name(nameBytes, '\0');
    receiveTestBytes(descriptor, name.data(), name.size());
    std::vector<char> payload(static_cast<std::size_t>(contentBytes));
    receiveTestBytes(descriptor, payload.data(), payload.size());
    if (type != 1 || name != queuedFrame.filename().string())
      throw std::runtime_error("Test server sent an unexpected frame");

    // The former implementation called recv immediately and failed after its
    // two-second SO_RCVTIMEO before a large frame could be published locally.
    std::this_thread::sleep_for(2500ms);
    sendTestU32(descriptor, 1);
    sendTestU32(descriptor, nameBytes);
    sendTestU64(descriptor, 0);
    sendTestBytes(descriptor, name.data(), name.size());

    const std::uint32_t endType = receiveTestU32(descriptor);
    const std::uint32_t endNameBytes = receiveTestU32(descriptor);
    const std::uint64_t endContentBytes = receiveTestU64(descriptor);
    if (endType != 2 || endNameBytes != 0 || endContentBytes != 0)
      throw std::runtime_error("Test server sent an invalid end message");
    ::close(descriptor);
  } catch (...) {
    clientFailure = std::current_exception();
  }

  stop = true;
  server.join();

  if (clientFailure) {
    try {
      std::rethrow_exception(clientFailure);
    } catch (const std::exception &error) {
      FAIL() << "Delayed-ACK client failed: " << error.what()
             << "\nServer log:\n"
             << serverLog.str();
    }
  }
  if (serverFailure) {
    try {
      std::rethrow_exception(serverFailure);
    } catch (const std::exception &error) {
      FAIL() << "Delayed-ACK server failed: " << error.what()
             << "\nServer log:\n"
             << serverLog.str();
    }
  }

  EXPECT_FALSE(std::filesystem::exists(queuedFrame));
  EXPECT_NE(serverLog.str().find("Acknowledged"), std::string::npos);
}

TEST_F(StreamingNetworkTest, RejectsWeakRelayTokenBeforeListening) {
  const auto tokenFile = directory / "weak.token";
  {
    std::ofstream token(tokenFile);
    token << "too-short\n";
  }
  std::filesystem::permissions(tokenFile,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::replace);

  nbody::streaming::RelayConfig config;
  config.sourceBindAddress = "127.0.0.1";
  config.sourcePort = reserveEphemeralPort();
  config.clientBindAddress = "127.0.0.1";
  config.clientPort = reserveEphemeralPort();
  config.tokenFile = tokenFile;
  std::ostringstream log;

  EXPECT_THROW(nbody::streaming::runRelay(
                   config, [] { return true; }, log),
               std::runtime_error);
}

} // namespace
