cmake_minimum_required(VERSION 3.14)
project(MovieTicketBooking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

# Set common FetchContent options
set(FETCHCONTENT_QUIET FALSE)

# Fetch GTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch Crow
FetchContent_Declare(
  Crow
  URL https://github.com/CrowCpp/Crow/releases/download/v1.1.0/Crow-1.1.0-source.zip
)
set(CROW_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(CROW_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
FetchContent_MakeAvailable(Crow)

# Try to find PostgreSQL first (needed by libpqxx)
# Note: On Windows this expects PostgreSQL to be installed and in CMake's search path
find_package(PostgreSQL REQUIRED)
find_package(Threads REQUIRED)

# Fetch libpqxx
FetchContent_Declare(
  libpqxx
  GIT_REPOSITORY https://github.com/jtv/libpqxx.git
  GIT_TAG 7.8.1
)
set(SKIP_BUILD_TEST CACHE BOOL "Skip building libpqxx tests" FORCE)
set(BUILD_DOC CACHE BOOL "Skip building libpqxx docs" FORCE)
FetchContent_MakeAvailable(libpqxx)

# Set up include directories
include_directories(
  src
  ${PostgreSQL_INCLUDE_DIRS}
)

# Core library (Database, Repositories, Services, and Controllers)
add_library(core_lib
  src/core/Database.cpp
  src/repositories/UserRepository.cpp
  src/repositories/CityRepository.cpp
  src/repositories/TheatreRepository.cpp
  src/repositories/ScreenRepository.cpp
  src/repositories/MovieRepository.cpp
  src/repositories/ShowRepository.cpp
  src/repositories/SeatRepository.cpp
  src/repositories/BookingRepository.cpp
  src/repositories/ShowSeatRepository.cpp
  src/repositories/PaymentRepository.cpp
  src/services/state/BookingState.cpp
  src/services/state/CreatedState.cpp
  src/services/state/SeatsLockedState.cpp
  src/services/state/PaymentPendingState.cpp
  src/services/state/ConfirmedState.cpp
  src/services/state/CancelledState.cpp
  src/services/payment/MockPaymentGateway.cpp
  src/services/payment/PaymentGatewayFactory.cpp
  src/services/BookingService.cpp
  src/services/PaymentService.cpp
  src/services/ShowService.cpp
  src/services/LockCleanupService.cpp
  src/controllers/ShowController.cpp
  src/controllers/BookingController.cpp
  src/controllers/PaymentController.cpp
  src/controllers/ResetController.cpp
)
target_include_directories(core_lib PUBLIC src)
target_link_libraries(core_lib PUBLIC Crow::Crow pqxx::pqxx ${PostgreSQL_LIBRARIES} Threads::Threads)
if(WIN32)
  target_link_libraries(core_lib PUBLIC ws2_32 mswsock)
endif()

# Full Server Executable
add_executable(server src/main.cpp)
target_link_libraries(server PRIVATE core_lib)

# Tests
enable_testing()

add_executable(DatabaseTest tests/core/DatabaseTest.cpp)
target_link_libraries(DatabaseTest PRIVATE core_lib gtest_main gmock)

add_executable(CoreRepositoriesTest tests/integration/CoreRepositoriesTest.cpp)
target_link_libraries(CoreRepositoriesTest PRIVATE core_lib gtest_main gmock)

add_executable(DomainRepositoriesTest tests/integration/DomainRepositoriesTest.cpp)
target_link_libraries(DomainRepositoriesTest PRIVATE core_lib gtest_main gmock)

add_executable(BookingStateTest tests/unit/BookingStateTest.cpp)
target_link_libraries(BookingStateTest PRIVATE core_lib gtest_main gmock)

add_executable(PaymentGatewayTest tests/unit/PaymentGatewayTest.cpp)
target_link_libraries(PaymentGatewayTest PRIVATE core_lib gtest_main gmock)

add_executable(BookingServiceTest tests/unit/BookingServiceTest.cpp)
target_link_libraries(BookingServiceTest PRIVATE core_lib gtest_main gmock)

add_executable(PaymentServiceTest tests/unit/PaymentServiceTest.cpp)
target_link_libraries(PaymentServiceTest PRIVATE core_lib gtest_main gmock)

add_executable(ShowServiceTest tests/unit/ShowServiceTest.cpp)
target_link_libraries(ShowServiceTest PRIVATE core_lib gtest_main gmock)

add_executable(LockCleanupServiceTest tests/unit/LockCleanupServiceTest.cpp)
target_link_libraries(LockCleanupServiceTest PRIVATE core_lib gtest_main gmock)

add_executable(ControllerTest tests/unit/ControllerTest.cpp)
target_link_libraries(ControllerTest PRIVATE core_lib gtest_main gmock)

include(GoogleTest)
gtest_discover_tests(DatabaseTest)
gtest_discover_tests(CoreRepositoriesTest)
gtest_discover_tests(DomainRepositoriesTest)
gtest_discover_tests(BookingStateTest)
gtest_discover_tests(PaymentGatewayTest)
gtest_discover_tests(BookingServiceTest)
gtest_discover_tests(PaymentServiceTest)
gtest_discover_tests(ShowServiceTest)
gtest_discover_tests(LockCleanupServiceTest)
gtest_discover_tests(ControllerTest)
