#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

import math
import numpy as np

import tf2_ros
import tf2_geometry_msgs

import tf
from tf import TransformListener
from tf.transformations import quaternion_from_euler

# TODO: import ROS msg types and libraries

class PurePursuit(object):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        # TODO: create ROS subscribers and publishers.
        path_topic = rospy.get_param('path_topic')
        gt_pose_topic = rospy.get_param('gt_pose_topic')
        drive_topic = rospy.get_param('drive_topic')
        self.L = rospy.get_param('pp_lookahead')
        self.velocity = 0.5
        self.Kp = 0.2

        self.curr_path = Path()
        self.stopped = False
        self.curr_pose = PoseStamped()

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.path_sub = rospy.Subscriber(path_topic, Path, self.path_callback, queue_size=1)
        self.gt_pose_sub = rospy.Subscriber(gt_pose_topic, PoseStamped, self.gt_pose_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1) #TODO: Publish to drive
    
    def gt_pose_callback(self, pose_msg):
        self.curr_pose = pose_msg

        path_msg = self.curr_path

        waypoint_distances = np.zeros((len(path_msg.poses,)))
        
        quaternion = (
                    self.curr_pose.pose.orientation.x,
                    self.curr_pose.pose.orientation.y,
                    self.curr_pose.pose.orientation.z,
                    self.curr_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        for i in range(len(path_msg.poses)):
            curr_point = path_msg.poses[i]
            dx = curr_point.pose.position.x - self.curr_pose.pose.position.x
            dy = curr_point.pose.position.y - self.curr_pose.pose.position.y
            
            dx_local = dy * np.sin(yaw) + dx * np.cos(yaw)

            if (dx_local < 0):
                waypoint_distances[i] = -1.0 * np.sqrt(dx**2 + dy**2)
            else:
                waypoint_distances[i] = np.sqrt(dx**2 + dy**2)


        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "base_link"


        if (all(l < self.L for l in waypoint_distances)):
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            if (not self.stopped):
                print("Stopping")
                self.stopped = True
        else:
            self.stopped = False
            i = 0
            while (waypoint_distances[i] <= self.L):
                i = i + 1
            
            if i == 0:
                goal_x = self.curr_pose.pose.position.x + self.L / (waypoint_distances[0]) * (path_msg.poses[0].pose.position.x - self.curr_pose.pose.position.x)
                goal_y = self.curr_pose.pose.position.y + self.L / (waypoint_distances[0]) * (path_msg.poses[0].pose.position.y - self.curr_pose.pose.position.y)
            
            else:
                L1 = waypoint_distances[i-1]
                L2 = waypoint_distances[i]
                p1x = path_msg.poses[i-1].pose.position.x
                p2x = path_msg.poses[i].pose.position.x
                p1y = path_msg.poses[i-1].pose.position.y
                p2y = path_msg.poses[i].pose.position.y

                goal_x = p1x + (self.L - L1) / (L2 - L1) * (p2x - p1x)
                goal_y = p1y + (self.L - L1) / (L2 - L1) * (p2y - p1y)
                
                goal_x = p2x
                goal_y = p2y

            goal_point = PointStamped()
            goal_point.header.frame_id = 'map'
            goal_point.header.stamp = rospy.Time(0)
            goal_point.point = Point()
            goal_point.point.x = goal_x
            goal_point.point.y = goal_y
            goal_point.point.z = 0.0

            # TODO: transform goal point to vehicle frame of reference
            transform = self.tf_buffer.lookup_transform('base_link',
                                    # source frame:
                                    goal_point.header.frame_id,
                                    # get the tf at the time the pose was valid
                                    rospy.Time(),
                                    # wait for at most 1 second for transform, otherwise throw
                                    rospy.Duration(3.0))

            goal_point_local = tf2_geometry_msgs.do_transform_point(goal_point, transform)

            
            # TODO: calculate curvature/steering angle
            y = goal_point_local.point.y
            
            gamma = 2.0 * y /self.L**2

            angle = self.Kp * gamma
            if angle < -0.4189:
                angle = -0.4189
            elif angle > 0.4189:
                angle = 0.4189

            drive_msg.drive.steering_angle = angle
            drive_msg.drive.speed = self.velocity

            print(goal_point_local.point.x, goal_point_local.point.y)
            print("Velocity command: ", drive_msg.drive.speed)
            print("Steering command: ", drive_msg.drive.steering_angle)
            # print("new steering command")
        
        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians
        self.drive_pub.publish(drive_msg)

    def path_callback(self, path_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture

        self.curr_path = path_msg

        # print("new path")

def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rospy.spin()
if __name__ == '__main__':
    main()