import multiprocessing as mp
import time

import rclpy
from ros2_rclpy_rclcpp_perf.msg import ExampleStatus, ExampleCommand

RATE_HZ = 1000.0


def make_status():
    msg = ExampleStatus()
    return msg


def make_command():
    msg = ExampleCommand()
    return msg


def wrap_rclpy(main):
    rclpy.init(signal_handler_options=rclpy.SignalHandlerOptions.NO)
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        print("del")


def pub_main():
    node = rclpy.create_node("pub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    pub_status = node.create_publisher(
        ExampleStatus, "/status", 1
    )
    pub_command = node.create_publisher(
        ExampleCommand, "/command", 1
    )

    rate = node.create_rate(RATE_HZ)
    print("Pub running")
    try:
        while rclpy.ok():
            pub_status.publish(make_status())
            pub_command.publish(make_command())
            executor.spin_once(timeout_sec=50e-6)
            rate.sleep()
    finally:
        print("Pub done")


def sub_main():
    node = rclpy.create_node("sub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def status_callback(msg):
        pass

    def command_callback(msg):
        pass

    sub_status = node.create_subscription(
        ExampleStatus, "/status", status_callback, 1
    )
    sub_command = node.create_subscription(
        ExampleCommand, "/command", command_callback, 1
    )

    rate = node.create_rate(RATE_HZ)
    print("Sub running")
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=50e-6)
            rate.sleep()
    finally:
        print("Sub done")


def main():
    procs = [
        mp.Process(target=wrap_rclpy, args=[pub_main]),
        mp.Process(target=wrap_rclpy, args=[sub_main]),
    ]
    t_end = time.time() + 5.0

    for proc in procs:
        proc.start()

    try:
        while time.time() < t_end:
            for proc in procs:
                assert proc.is_alive()
            time.sleep(1 / RATE_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            if proc.is_alive():
                print("kill")
                proc.kill()
                proc.join()


if __name__ == "__main__":
    main()
