#include "component/LidarSensor.hh"
#include "component/MapGenerator.hh"
#include "component/RobotEntry.hh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "geometry/GeometryManager.cuh"

#include <SFML/Graphics.hpp>
#include <stdexcept>

using namespace cuda_simulator;
using namespace cuda_simulator::core;

// constexpr int MAP_WIDTH = 80;
// constexpr int MAP_HEIGHT = 60;
// constexpr int GRID_SIZE = 1;

constexpr float SCALING = 200;
constexpr float OFFSET = 20;

template <typename T> sf::Vector2<T> convertPoint(sf::RenderWindow &window_, sf::Vector2<T> v) {
  auto size = window_.getView().getSize();
  return sf::Vector2<T>(v.x * SCALING + OFFSET, size.y - (v.y * SCALING + OFFSET));
}

template <typename T> sf::Vector2<T> invConvertPoint(sf::RenderWindow &window_, sf::Vector2<T> v) {
  auto size = window_.getSize();
  return sf::Vector2<T>((v.x - OFFSET) / (size.x / 1280.f) / SCALING,
                        ((size.y - v.y) - OFFSET) / (size.y / 1024.f) / SCALING);
}

class TestLidar {
  sf::RenderWindow window_;
  sf::Text mouse_pos_text_;
  float2 lidar_pos_ = make_float2(2, 2);
  std::vector<float4> ray_lines;
  const float4 *static_lines = nullptr;
  // std::vector<float> static_lines;
  uint32_t num_static_lines = 0;

  map_gen::MapGenerator *map_generator;
  robot_entry::RobotEntry *robot_entry;
  lidar_sensor::LidarSensor *lidar_sensor;

public:
  TestLidar() : window_(sf::VideoMode(1280, 1024), "SFML Draw Lines") {
    window_.setVerticalSyncEnabled(true); // call it once, after creating the window_

    // sf::Font font;
    // if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")) { // 确保有合适的字体文件
    //   throw std::runtime_error("Failed to load font!");
    // }

    // 文本对象
    // mouse_pos_text_.setFont(font);
    // mouse_pos_text_.setCharacterSize(20);           // 设置字体大小
    // mouse_pos_text_.setFillColor(sf::Color::Green); // 设置文字颜色
    // mouse_pos_text_.setPosition(10, 10);            // 设置文字位置

    getContext()->initialize();
    map_generator = getContext()->createComponent<map_gen::MapGenerator>(3, 3, 0.05);
    robot_entry = getContext()->createComponent<robot_entry::RobotEntry>(1);
    lidar_sensor = getContext()->createComponent<lidar_sensor::LidarSensor>();

    // todo. 统一地图尺寸
    getContext()->setup();

    static_lines = getGeometryManager()->getStaticLines()->at(0).typed_data<float4>();
    num_static_lines = getGeometryManager()->getNumStaticLines()->hostAt(0);
    ray_lines.resize(lidar_sensor->getLidarRayNum());
  }

  void drawStaticLines() {
    for (size_t i = 0; i < num_static_lines; i++) {
      // 创建一个 sf::VertexArray 用于绘制线段
      sf::VertexArray lineShape(sf::Lines, 2);

      // 设置第一个点
      lineShape[0].position = convertPoint(window_, sf::Vector2f(static_lines[i].x, static_lines[i].y));
      lineShape[0].color = sf::Color::Red;

      // 设置第二个点
      lineShape[1].position = convertPoint(window_, sf::Vector2f(static_lines[i].z, static_lines[i].w));
      lineShape[1].color = sf::Color::White;

      // 绘制线段
      window_.draw(lineShape);
    }
  }

  void drawRays() {
    for (const auto &line : ray_lines) {
      // 创建一个 sf::VertexArray 用于绘制线段
      sf::VertexArray lineShape(sf::Lines, 2);

      // 设置第一个点
      lineShape[0].position = convertPoint(window_, sf::Vector2f(line.x, line.y));
      lineShape[0].color = sf::Color::Blue;
      // 设置第二个点
      lineShape[1].position = convertPoint(window_, sf::Vector2f(line.z, line.w));
      lineShape[1].color = sf::Color::White;

      // 绘制线段
      window_.draw(lineShape);
    }
  }

  void updateLidar(float2 lidar_pos) {
      // std::vector<float> rays = rasterGPU(lbegins.size(), make_float3(lidar_pos_.x, lidar_pos_.y, 0), lbegins, lends);
      robot_entry->setRobotPose(make_float4(lidar_pos.x, lidar_pos.y, 0, 0));

      getContext()->trigger("default");

      MessageQueue* queue = getMessageBus()->getMessageQueue("lidar_sensor", "lidar");
      auto &lidar = queue->getHistoryTensorHandle(0);

      std::vector<uint32_t> raw_data;
      lidar[{0,0,0}].toHostVector(raw_data);

      for (size_t i = 0; i < raw_data.size(); i++) {
        float angle = i * lidar_sensor->getLidarResolution();
        float r = (raw_data[i] >> 16) / 1024.f;
        float4 cartesian_ray = make_float4(lidar_pos.x, lidar_pos.y,r * cosf(angle) + lidar_pos_.x, r * sinf(angle) + lidar_pos_.y);
        ray_lines[i] = cartesian_ray;
      }
  }

  void repaint(const sf::Vector2f &mouse_world_pos) {
    // 获取鼠标位置并更新文本内容
    mouse_pos_text_.setString("Mouse: (" + std::to_string(mouse_world_pos.x) + ", " + std::to_string(mouse_world_pos.y) +
                              ")");

    // 清空窗口
    window_.clear(sf::Color::Black);

    drawStaticLines();
    drawRays();

    // 绘制鼠标位置文本
    window_.draw(mouse_pos_text_);
    // 显示窗口内容
    window_.display();
  }

  void show() {
    // 主循环
    while (window_.isOpen()) {
      sf::Event event;

      sf::Vector2i mousePos = sf::Mouse::getPosition(window_);
      sf::Vector2f mouse_world_pos =
          invConvertPoint<float>(window_, {static_cast<float>(mousePos.x), static_cast<float>(mousePos.y)});

      while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
          window_.close();
        }

        // 鼠标点击事件
        if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
          // 获取鼠标点击位置

          // 转换为逻辑坐标
          lidar_pos_.x = mouse_world_pos.x;
          lidar_pos_.y = mouse_world_pos.y;

          // 打印鼠标点击位置（调试用）
          std::cout << "Mouse clicked at: (" << lidar_pos_.x << ", " << lidar_pos_.y << ")\n";

          updateLidar(lidar_pos_);
        }
      }

      repaint(mouse_world_pos);
    }
  }
};

int main() {
  TestLidar test_lidar;
  test_lidar.show();
  return 0;
}
