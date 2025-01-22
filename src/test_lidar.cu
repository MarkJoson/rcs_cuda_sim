#include "component/LidarSensor.hh"
#include "component/MapGenerator.hh"
#include "component/RobotEntry.hh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "geometry/GeometryManager.cuh"

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/CircleShape.hpp>
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

  // 环境id，环境组id
  int env_id_ = 0;
  int active_group_id_ = 0;

  // 当前lidar位置
  float2 mouse_pos_ = make_float2(2, 2);
  float4 lidar_pose_;

  // 动态线段
  std::vector<float4> ray_lines;

  // 静态线段
  const float4 *static_lines = nullptr;
  uint32_t num_static_lines = 0;

  map_gen::MapGenerator *map_generator;
  robot_entry::RobotEntry *robot_entry;
  lidar_sensor::LidarSensor *lidar_sensor;

public:
  TestLidar() : window_(sf::VideoMode(1280, 1024), "SFML Draw Lines") {
    window_.setVerticalSyncEnabled(true); // call it once, after creating the window_

    // sf::Font font;
    // if (!font.loadFromFile("../../Ubuntu-R.ttf")) { // 确保有合适的字体文件
    //   throw std::runtime_error("Failed to load font!");
    // }

    // // 文本对象
    // mouse_pos_text_.setFont(font);
    // mouse_pos_text_.setCharacterSize(20);           // 设置字体大小
    // mouse_pos_text_.setFillColor(sf::Color::Green); // 设置文字颜色
    // mouse_pos_text_.setPosition(10, 10);            // 设置文字位置

    getContext()->initialize(16, 16, 16);
    map_generator = getContext()->createComponent<map_gen::MapGenerator>(3, 3, 0.05);
    robot_entry = getContext()->createComponent<robot_entry::RobotEntry>(1);
    lidar_sensor = getContext()->createComponent<lidar_sensor::LidarSensor>();

    // todo. 统一地图尺寸
    getContext()->setup();

    ray_lines.resize(lidar_sensor->getLidarRayNum());

    // 设置默认环境为0,0
    changeEnvironGroup(2);
    changeEnviron(0);
  }

  void changeEnvironGroup(int active_group_id) {
    if (active_group_id >= getEnvGroupMgr()->getNumActiveGroup()) {
      throw std::runtime_error("Invalid group id");
    }

    active_group_id_ = active_group_id;

    // 主机访问Config时是通过访问其在Handle内的影子数据完成的
    static_lines = getGeometryManager()->getStaticLines()->activeGroupAt(active_group_id_).typed_data<float4>();
    num_static_lines = getGeometryManager()->getNumStaticLines()->activeGroupAt(active_group_id_);
  }

  void changeEnviron(int env_id) {
    if (env_id >= getEnvGroupMgr()->getNumEnvPerGroup()) {
      throw std::runtime_error("Invalid env id");
    }

    env_id_ = env_id;
  }

  void grabLidarPose() {
    MessageQueue *pose = getMessageBus()->getMessageQueue("robot_entry", "pose");
    auto &pose_tensor = pose->getHistoryGTensor(0);
    std::vector<float> pose_data;
    pose_tensor[{active_group_id_, env_id_, 0}].toHostVector(pose_data);
    lidar_pose_ = make_float4(pose_data[0], pose_data[1], pose_data[2], pose_data[3]);
  }

  void grabLidarResult() {
    MessageQueue *queue = getMessageBus()->getMessageQueue("lidar_sensor", "lidar");
    auto &lidar = queue->getHistoryGTensor(0);

    std::vector<uint32_t> raw_lidar;
    lidar[{active_group_id_, env_id_, 0}].toHostVector(raw_lidar);

    for (size_t i = 0; i < raw_lidar.size(); i++) {
      float angle = i * lidar_sensor->getLidarResolution();
      float r = (raw_lidar[i] >> 16) / 1024.f;
      float4 cartesian_ray =
          make_float4(lidar_pose_.x, lidar_pose_.y, r * cosf(angle) + lidar_pose_.x, r * sinf(angle) + lidar_pose_.y);
      ray_lines[i] = cartesian_ray;
    }
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
    sf::CircleShape circle(30);
    circle.setFillColor(sf::Color::Green);
    circle.setOrigin(circle.getRadius(), circle.getRadius());
    circle.setPosition(convertPoint(window_, sf::Vector2f(lidar_pose_.x, lidar_pose_.y)));
    window_.draw(circle);

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

  void updateLidar(float2 click_pos) {
    // std::vector<float> rays = rasterGPU(lbegins.size(), make_float3(lidar_pos_.x, lidar_pos_.y, 0), lbegins, lends);
    robot_entry->setRobotPose(make_float4(click_pos.x, click_pos.y, 0, 0));
    getContext()->trigger("default");

    grabLidarPose();
    grabLidarResult();
  }

  void repaint(const sf::Vector2f &mouse_world_pos) {
    // 获取鼠标位置并更新文本内容
    mouse_pos_text_.setString("Mouse: (" + std::to_string(mouse_world_pos.x) + ", " +
                              std::to_string(mouse_world_pos.y) + ")");

    // 清空窗口
    window_.clear(sf::Color::Black);

    drawStaticLines();
    drawRays();

    // 绘制鼠标位置文本
    // window_.draw(mouse_pos_text_);
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

        if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
          // 鼠标点击事件
          mouse_pos_.x = mouse_world_pos.x;
          mouse_pos_.y = mouse_world_pos.y;
          std::cout << "Mouse clicked at: (" << mouse_pos_.x << ", " << mouse_pos_.y << ")\n";
          updateLidar(mouse_pos_);
        } else if (event.type == sf::Event::KeyPressed && event.key.code >= sf::Keyboard::Num0 &&
                   event.key.code <= sf::Keyboard::Num9) {
          // 0-9键按下, 获取按下的数字
          int num = event.key.code - sf::Keyboard::Num0;
          std::cout << "Switching to environ group" << num << "!\n";
          changeEnvironGroup(num);

          grabLidarPose();
          grabLidarResult();
        } else if (event.type == sf::Event::KeyPressed && event.key.code >= sf::Keyboard::Numpad0 &&
                   event.key.code <= sf::Keyboard::Numpad9) {
          // 0-9键按下, 获取按下的数字
          int num = event.key.code - sf::Keyboard::Numpad0;
          std::cout << "Switching to environ" << num << "!\n";
          changeEnviron(num);

          grabLidarPose();
          grabLidarResult();
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
