'''
Python script to read ROS bags. 
It shows on different screens both the RGB and depth image.
On terminal, pose gets printed.
Finally, both, RGB and depth video compositions are stored on output_video_color.mp4 and output_video_depth.mp4 , respectively
'''

path = 'realsense_data_29.bag' #Change it to visualize specific ROS bags

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import cv2
import numpy as np



def print_pose(pose_msg):
    # Extract position and orientation from PoseStamped message
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation

    # Print position
    print("Position:")
    print("x: {:.5f}, y: {:.2f}, z: {:.2f}".format(position.x, position.y, position.z))

    # Print orientation
    print("Orientation:")
    print(
        "x: {:.2f}, y: {:.2f}, z: {:.2f}, w: {:.2f}".format(orientation.x, orientation.y, orientation.z, orientation.w))


def main():
    bag = rosbag.Bag(path)

    output_video_color = 'output_video_color.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is for mp4 files
    fps = 30  # Frames per second
    video_writer = cv2.VideoWriter(output_video_color, fourcc, fps, (640,480))

    output_video_depth = 'output_video_depth.mp4'
    video_writer2 = cv2.VideoWriter(output_video_depth, fourcc, fps, (640, 480))

    try:
        

        # Iterate over each message in the bag file
        for msg1, msg2, msg3, in zip(bag.read_messages(topics=['/camera1/color/image_raw']),
                                     bag.read_messages(topics=['/camera1/depth/image_raw']),
                                     bag.read_messages(topics=['/camera2/pose'])):
            # Print the topic, message, and timestamp
            #print("Topic: {}, Message: {}, Timestamp: {}".format(topic, msg, t))
            bridge = CvBridge()
            cv2imgcolor = bridge.imgmsg_to_cv2(msg1.message, desired_encoding="bgr8")
            cv2img = bridge.imgmsg_to_cv2(msg2.message, desired_encoding="16UC3")
            # Grayish depth img to colors
            depth_8bit = cv2.normalize(cv2img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            cv2.imshow('color', cv2imgcolor)
            cv2.imshow('depth', depth_colormap)
            cv2.waitKey(1)
            video_writer.write(cv2imgcolor)
            video_writer2.write(depth_colormap)
            print_pose(msg3.message)




        

    finally:
        # Close the bag file
        bag.close()
        video_writer.release()
        video_writer2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()