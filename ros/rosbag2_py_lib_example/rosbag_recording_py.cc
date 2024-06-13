#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/yaml.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <rosbag2_transport/record_options.hpp>
#include <rosbag2_transport/recorder.hpp>

#include "drake/common/drake_assert.h"

namespace py = pybind11;

namespace example {

// from https://github.com/ros2/rosbag2/blob/8c94497/rosbag2_py/src/rosbag2_py/_transport.cpp#L174-L262
class Recorder {
 public:
  Recorder() : is_recording_(false), record_count_(0) {
    DRAKE_DEMAND(rclcpp::ok());
  }

  ~Recorder() {
    cancel();
  }

  void spin_node() {
    {
        std::lock_guard<std::mutex> lock(wait_for_start_th_mutex_);
        is_recording_ = true;
        wait_for_start_th_cv_.notify_one();
    }
    executor_.spin();
  }

  void record(
    const rosbag2_storage::StorageOptions & storage_options,
    const rosbag2_transport::RecordOptions & record_options) {
    if (is_recording_) {
      throw std::runtime_error("already recording!!");
    }
    record_count_++;

    DRAKE_DEMAND(!record_options.rmw_serialization_format.empty());

    auto writer =
        rosbag2_transport::ReaderWriterFactory::make_writer(record_options);
    auto node_name = "rosbag2_recorder_" + std::to_string(record_count_);
    recorder_ = std::make_shared<rosbag2_transport::Recorder>(
      std::move(writer), storage_options, record_options, node_name);
    recorder_->record();
    executor_.add_node(recorder_);

    recording_thread_ = std::thread(&Recorder::spin_node, this);
    // ensure spin start
    std::unique_lock<std::mutex> lock(wait_for_start_th_mutex_);
    wait_for_start_th_cv_.wait(lock, [&]{ return is_recording_; });
  }

  void cancel() {
    if (!is_recording_) {
      return;
    }
    is_recording_ = false;
    recorder_->stop();
    executor_.cancel();
    if (recording_thread_.joinable()) {
      recording_thread_.join();
    }
    executor_.remove_node(recorder_);
  }

  bool is_recording() const {
    return is_recording_;
  }

 private:
  bool is_recording_;
  int record_count_;
  std::thread recording_thread_;
  std::condition_variable wait_for_start_th_cv_;
  std::mutex wait_for_start_th_mutex_;
  std::shared_ptr<rosbag2_transport::Recorder> recorder_;
  rclcpp::executors::SingleThreadedExecutor executor_;
};


PYBIND11_MODULE(rosbag_recording, m) {
  py::class_<Recorder>(m, "Recorder")
  .def(py::init<>())
  .def("record", &Recorder::record,
    py::arg("storage_options"), py::arg("record_options"))
  .def("cancel", &Recorder::cancel)
  .def("is_recording", &Recorder::is_recording);

  {
    // we need to bind this because Recorder.record() couldn't accept
    // ros2bag_py.RecordOptions() and we don't not bind
    // rosbag2_storage::StorageOptions() because import register issue
    // occur when import rosbag_recording(this) and ros2bag_py both.
    using RecordOptions = rosbag2_transport::RecordOptions;
    py::class_<RecordOptions>(m, "RecordOptions")
    .def(py::init<>())
    .def_readwrite("all", &RecordOptions::all)
    .def_readwrite("exclude", &RecordOptions::exclude)
    .def_readwrite(
      "rmw_serialization_format", &RecordOptions::rmw_serialization_format);
  }
}

}  // namespace example
