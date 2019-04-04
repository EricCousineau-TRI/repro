// Adapted from: http://wiki.ros.org/rviz/Tutorials/Markers%3A%20Basic%20Shapes

#include <iostream>

#include <rclcpp/executors.hpp>
#include <rclcpp/node.hpp>
#include <visualization_msgs/msg/marker.hpp>

using visualization_msgs::msg::Marker;
using std::chrono_literals::operator""s;

int main( int argc, char** argv )
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("basic_shapes");
  auto marker_pub = node->create_publisher<Marker>("visualization_marker");

  // Set our initial shape type to be a cube
  uint32_t shape = Marker::CUBE;

  auto callback = [&]() {
    Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/my_frame";
    marker.header.stamp = rclcpp::Clock().now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "basic_shapes";
    marker.id = 0;

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = rclcpp::Duration(0.0);

    // Publish the marker
    std::cout << "Publish" << std::endl;
    marker_pub->publish(marker);

    // Cycle between different shapes
    switch (shape)
    {
    case Marker::CUBE:
      shape = Marker::SPHERE;
      break;
    case Marker::SPHERE:
      shape = Marker::ARROW;
      break;
    case Marker::ARROW:
      shape = Marker::CYLINDER;
      break;
    case Marker::CYLINDER:
      shape = Marker::CUBE;
      break;
    }
  };
  rclcpp::TimerBase::SharedPtr timer = node->create_wall_timer(1s, callback);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
