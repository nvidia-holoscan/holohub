/*
 * Place the license header here
 */

#include <holoscan/holoscan.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    // Add your operators here
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();
  return 0;
}
