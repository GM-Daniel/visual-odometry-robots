# visual-odometry-robots
## Introduction
This repository contains the code and resources developed for the Undergraduate's dissertation titled "Visual Odometry Service for Teams of Heterogeneous Robots", presented at University of Zaragoza. The main goal of this project is to explore and address a solution to obtain 6-DOF transformation of robots along the scene.

Throughout the project, Python has been used as the primary programming language. For tracking purposes, an Intel D435 camera has been utilized to capture changes in the robots' position and orientation within the scene, while an Intel T265 camera has been employed to monitor the movement of both cameras from their initial recording position. The project leverages the pyrealsense2 library to interface with the cameras, along with computer vision techniques (using PyTorch) to recognize and identify the robots. This combination of tools allows for tracking and analysis of the robots' rotation.

The main contents of this repository include:

    Python Scripts for Scene Capture and Rosbag Handling
    Main Code: The primary script that analyzes the trajectories of robots as described in the report. It uses the recorded scenes, using data from the captured ROS bags (instead of a real-time implementation).
    Trained Model: A pre-trained model used for recognizing and tracking the robots within the recorded scenes.
    Recorded Scenes ROS bags.

For more details on the project objectives and findings, please refer to the accompanying thesis document [link if applicable].
Table of Contents

    Installation
    Usage
    Features
    Contributing
    License

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

    Install Intel RealSense SDK v2.50: download and install the Intel RealSense SDK version 2.50 from the official Intel website. This SDK is required to interface with the Intel T265 and D435 cameras.

    Create a new Conda environment with Python version 3.8.3
    
    Install pyrealsense2 v2.43.0.3018

    Install Robot Operating System
    
    Additional Dependencies, such as OpenCV

