'''
Python script to create ROS bags. 
It reads the cameras input to store info on ROS bag.
Once executed, it takes 5 seconds from message "I'm running" to start recording
Change path name to your choose
'''

path = 'realsense_data_29.bag' #Change it to visualize specific ROS bags
d435_SN = '0123' #change serial number to the one's of your D435 camera
t265_SN = '1234' #change serial number to the one's of your T265 camera


import cv2
#import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np
import keyboard
from transforms3d.quaternions import quat2mat
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import time
def check_key():
    return keyboard.is_pressed('space')  # Change 'space' to the desired key

def convert_to_ros_image(frame):
    """
    Convert a RealSense frame to a ROS Image message.

    Args:
        frame: A RealSense frame object.

    Returns:
        msg: A ROS Image message.
    """
    bridge = CvBridge()

    # Convert RealSense frame to numpy array
    image_data = np.asanyarray(frame.get_data())

    # Convert numpy array to OpenCV image
    image_cv = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to ROS Image message
    msg = bridge.cv2_to_imgmsg(image_cv, encoding="rgb8")

    return msg


def convert_to_ros_depth(frame):
    """
    Convert a RealSense frame to a ROS Image message.

    Args:
        frame: A RealSense frame object.

    Returns:
        msg: A ROS Image message.
    """
    bridge = CvBridge()

    # Convert RealSense frame to numpy array
    image_data = np.asanyarray(frame.get_data())

    # Convert numpy array to OpenCV image
    image_cv = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to ROS Image message
    msg = bridge.cv2_to_imgmsg(image_cv, encoding="16UC3")

    return msg




def main():
    # Initialize ROS node
    rospy.init_node('realsense_capture', anonymous=True)
    print("I'm running")

    pipeline_d435 = rs.pipeline()
    config_d435 = rs.config()
    config_d435.enable_device(d435_SN)
    config_d435.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_d435.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline_d435.start(config_d435)

    pipeline_t265 = rs.pipeline()
    config_t265 = rs.config()
    config_t265.enable_device(t265_SN)
    config_t265.enable_stream(rs.stream.pose)
    pipeline_t265.start(config_t265)

    depth_sensor = pipeline_d435.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("depth_scale is: ", depth_scale)

    # Initialize ROSBAG
    bag = rosbag.Bag(path, 'w')
    time.sleep(5)
    print("Cameras configured and ROSBAG initialized. Starting capturing...")
    i=1
    try:
        while not rospy.is_shutdown():
            # Capture frames from both cameras
            frames1 = pipeline_d435.wait_for_frames()
            frames2 = pipeline_t265.wait_for_frames()

            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames1)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()

            pose_frame = frames2.get_pose_frame()
            pose_data = pose_frame.get_pose_data()
            # Create ROS Image messages
            msg_color1 = Image()
            msg_depth1 = Image()


            # Populate ROS Image messages with data
            # (Assuming you have functions to convert RealSense frames to ROS Image messages)
            msg_color1 = convert_to_ros_image(aligned_color_frame)
            msg_depth1 = convert_to_ros_depth(aligned_depth_frame)
            # Create ROS PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera pose frame"
            # Set the position of the message
            pose_msg.pose.position.x = pose_data.translation.x
            pose_msg.pose.position.y = pose_data.translation.y
            pose_msg.pose.position.z = pose_data.translation.z

            # Set the orientation of the message
            pose_msg.pose.orientation.x = pose_data.rotation.x
            pose_msg.pose.orientation.y = pose_data.rotation.y
            pose_msg.pose.orientation.z = pose_data.rotation.z
            pose_msg.pose.orientation.w = pose_data.rotation.w
            #pose_msg = create_pose_msg(pose_frame)
            # Publish ROS Image messages
            # (You need to define publisher for each camera's topic)
            # pub_color1.publish(msg_color1)
            # pub_depth1.publish(msg_depth1)

            print("x: ",pose_msg.pose.position.x, " y: ",pose_msg.pose.position.y, " z: ", pose_msg.pose.position.z )


            i= i+1
            # Write data to ROSBAG
            bag.write('/camera1/color/image_raw', msg_color1)
            bag.write('/camera1/depth/image_raw', msg_depth1)
            bag.write('/camera2/pose', pose_msg)

            if check_key():
                print("Key pressed. Exiting loop.")
                break


    finally:
        # Stop RealSense pipeline
        pipeline_d435.stop()
        pipeline_t265.stop()

        # Close ROSBAG
        bag.close()

        print("Stopping pipelines and closing bag")

    print("close")

if __name__ == '__main__':
    main()
