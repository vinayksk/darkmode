#!/usr/bin/python3

"""
ESE 680
RRT assignment
Author: Hongrui Zheng

This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math
import time

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path

import tf2_ros
import tf2_geometry_msgs

import tf
from tf import TransformListener
from tf.transformations import quaternion_from_euler

import rosparam
import yaml
import matplotlib.pyplot as plt

# data = rosparam.load_file("/home/kvedula/f1tenth_ws/src/f1tenth_labs/lab7/code/rrt_params.yaml")

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(object):
    def __init__(self):
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        gt_pose_topic = rospy.get_param('gt_pose_topic')
        scan_topic = rospy.get_param('scan_topic')
        og_topic = rospy.get_param('og_topic')
        path_topic = rospy.get_param('path_topic')

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.dx = rospy.get_param('dx')
        self.dy = rospy.get_param('dy')
        self.Lx = rospy.get_param('Lx')
        self.Ly = rospy.get_param('Ly')

        self.global_goal = PointStamped()
        self.global_goal.header.stamp = rospy.Time.now()
        self.global_goal.header.frame_id = 'map'
        self.global_goal.point.x = rospy.get_param('goal_x')
        self.global_goal.point.y = rospy.get_param('goal_y')

        self.local_goal = PointStamped()

        self.occ = 10.0

        self.goal_tol = rospy.get_param('goal_tol')
        self.goal_local_tol_f = 1.0

        self.tree = []
        self.r_nbhd = rospy.get_param('r_nbhd')
        self.r_steer = rospy.get_param('r_steer')

        self.curr_pose = PoseStamped()
        
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.gt_pose_sub = rospy.Subscriber(gt_pose_topic, PoseStamped, self.gt_pose_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.og_sub = rospy.Subscriber(og_topic, OccupancyGrid, self.og_callback, queue_size=1)

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.og_pub = rospy.Publisher(og_topic, OccupancyGrid, queue_size=1)
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=1)

        # drive_topic = rospy.get_param('drive_topic')
        # self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1) #TODO: Publish to drive
        # pf_topic = rospy.get_param('pose_topic')
        # self.pf_sub = rospy.Subscriber(pf_topic, PoseStamped, self.pf_callback, queue_size=1)


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """

        og = np.zeros((np.int(np.round(self.Lx/self.dx)), np.int(np.round(self.Ly/self.dy))))

        r = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(r))
        angles_shift = angles + np.pi/2.0

        for i in range(len(r)):
            if r[i] < scan_msg.range_min:
                r[i] = scan_msg.range_min
            if r[i] > scan_msg.range_max:
                r[i] = scan_msg.range_max
        
        for i in range(len(r)):
            theta = angles_shift[i]

            if theta < np.pi and theta > 0.0:
                x_occ = r[i] * np.sin(theta)
                y_occ = -r[i] * np.cos(theta)

                if (x_occ < self.Lx and y_occ < 0.5*self.Ly and y_occ > -0.5*self.Ly):
                    x_ind, y_ind = self.coord_to_ind(x_occ, y_occ)
                    og[x_ind, y_ind] = self.occ

        og_msg = OccupancyGrid()
        og_msg.header.frame_id = 'base_link'
        og_msg.header.stamp = rospy.Time.now()
        og_msg.info.resolution = np.sqrt(self.dx * self.dy)
        og_msg.info.width = np.int(np.round(self.Ly/self.dy))
        og_msg.info.height = np.int(np.round(self.Lx/self.dx))
        og_msg.info.origin.position.x = 0.0 + self.curr_pose.pose.position.x
        og_msg.info.origin.position.y = self.Ly/2.0 + self.curr_pose.pose.position.x
        og_msg.info.origin.orientation.x = self.curr_pose.pose.orientation.x
        og_msg.info.origin.orientation.y = self.curr_pose.pose.orientation.y
        og_msg.info.origin.orientation.z = self.curr_pose.pose.orientation.z
        og_msg.info.origin.orientation.w = self.curr_pose.pose.orientation.w

        og_flat = og.flatten()
        og_msg.data = og_flat.astype(int)

        self.og_pub.publish(og_msg)

    def og_callback(self, og_msg):

        tree = []
        root_node = Node()
        root_node.x = 0.0
        root_node.y = 0.0
        root_node.is_root = True
        root_node.cost = 0.0

        width = og_msg.info.width
        height = og_msg.info.height
        og = np.array(og_msg.data).reshape(height, width)

        tree.insert(0, root_node)

        self.local_goal = self.find_local_goal(self.curr_pose, og_msg, self.global_goal)
        
        print(self.local_goal.point.x, self.local_goal.point.y)

        while True:
            sampled_point = self.sample(og)

            nearest_node = self.nearest(tree, sampled_point)

            new_node = self.steer(nearest_node, sampled_point)

            best_node = self.best_neighbor(tree, new_node)

            collision = self.check_collision(og, best_node, new_node)
            
            if not collision:
                tree.append(new_node)
                goal = self.is_goal(new_node, self.local_goal)
                
                if goal:
                    path = self.find_path(tree, new_node)
                    break
        
        print('Path generated')
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.curr_pose.header.stamp
        path_msg.poses = []

        for i in range(len(path)):

            curr_node = path[i]
            
            new_pose = PoseStamped()
            new_pose.header.frame_id = 'base_link'
            new_pose.header.stamp = rospy.Time(0)
            new_pose.pose = Pose()
            new_pose.pose.position.x = curr_node.x
            new_pose.pose.position.y = curr_node.y
            new_pose.pose.position.z = 0.0
            new_pose.pose.orientation.x = 0.0
            new_pose.pose.orientation.y = 0.0
            new_pose.pose.orientation.z = 0.0
            new_pose.pose.orientation.w = 1.0
            

            transform = self.tf_buffer.lookup_transform('map',
                                            # source frame:
                                            new_pose.header.frame_id,
                                            # get the tf at the time the pose was valid
                                            rospy.Time(),
                                            # wait for at most 1 second for transform, otherwise throw
                                            rospy.Duration(1.0))

            new_pose_map = tf2_geometry_msgs.do_transform_pose(new_pose, transform)
            path_msg.poses.append(new_pose_map)
        
        self.path_pub.publish(path_msg)
        print('Path published')
        self.visualize_path(og_msg, path, self.local_goal, True)
        
        

    def gt_pose_callback(self, pose_msg):

        self.curr_pose = pose_msg

        return None

    def find_local_goal(self, curr_pose, og_msg, global_goal):

        transform = self.tf_buffer.lookup_transform('base_link',
                                        # source frame:
                                        global_goal.header.frame_id,
                                        # get the tf at the time the pose was valid
                                        rospy.Time(),
                                        # wait for at most 1 second for transform, otherwise throw
                                        rospy.Duration(1.0))

        local_goal = tf2_geometry_msgs.do_transform_point(global_goal, transform)
        
        return local_goal
        
    def coord_to_ind(self, x, y):

        x_ind = np.int(-1.0 * np.round((x + 0.5000001 * self.dx) / self.dx))
        y_ind = np.int(-1.0 * np.round((y + 0.5 * self.dy) / self.dy) + np.round(self.Ly / self.dy / 2.0))

        return x_ind, y_ind

    def sample(self, og):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """

        while True:
            x = self.Lx * np.random.random()
            y = self.Ly * np.random.random() - self.Ly/2.0

            x_ind, y_ind = self.coord_to_ind(x, y)

            if(og[x_ind, y_ind] == 0):
                break

        sampled_point = Point()
        sampled_point.x = x
        sampled_point.y = y
        sampled_point.z = 0

        return sampled_point

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """

        min_dist = 10000.0
        min_ind = 0

        for i in range(len(tree)):
            node = tree[i]
            l = (node.x - sampled_point.x)**2 + (node.y - sampled_point.y)**2
            if l < min_dist:
                min_ind = i
                min_dist = l
        nearest_node = tree[min_ind]
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """

        l = np.sqrt((sampled_point.x - nearest_node.x)**2 + (sampled_point.y - nearest_node.y)**2)

        x_new = nearest_node.x + min(self.r_steer, l) * (sampled_point.x - nearest_node.x) / l
        y_new = nearest_node.y + min(self.r_steer, l) * (sampled_point.y - nearest_node.y) / l

        new_node = Node()
        new_node.x = x_new
        new_node.y = y_new
        return new_node

    def check_collision(self, og, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid

        """
        num_samples = 10 * np.sqrt((new_node.x - nearest_node.x)**2 + (new_node.y - nearest_node.y)**2) / np.sqrt((self.dx**2+self.dy**2))
        for i in range(np.int(num_samples)):
            x = nearest_node.x + i * (new_node.x - nearest_node.x) / num_samples
            y = nearest_node.y +  i * (new_node.y - nearest_node.y) / num_samples

            x_ind, y_ind = self.coord_to_ind(x, y)

            if(og[x_ind, y_ind] !=0):
                return True
 

        return False

    def is_goal(self, latest_added_node, local_goal):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """

        l = np.sqrt((local_goal.point.x - latest_added_node.x)**2 +(local_goal.point.y - latest_added_node.y)**2)

        if l < self.goal_tol:
            return True

        return False

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """

        path = [latest_added_node]

        currNode = latest_added_node

        while currNode.is_root == False:
            currNode = currNode.parent
            path.insert(0, currNode)

        return path

    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        if node.is_root == True:
            cost = 0.0

        else:
            parent = node.parent
            cost = parent.cost + self.line_cost(node, parent)

        return cost

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        cost = np.sqrt((n1.x - n2.x)**2 +(n1.y - n2.y)**2)

        return cost

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []

        for i in range(len(tree)):
            l = self.line_cost(node, tree[i])
            if l < self.r_nbhd:
                neighborhood.append(tree[i])

        return neighborhood

    def best_neighbor(self, tree, node):

        neighborhood = self.near(tree, node)

        point = Point()
        point.x = node.x
        point.x = node.y
        point.z = 0
        
        best_node = self.nearest(tree, point)
        node.parent = best_node
        min_cost = self.cost(tree, node)

        for i in range(len(neighborhood)):
            node.parent = neighborhood[i]
            curr_cost = self.cost(tree, node)
            if curr_cost < min_cost:
                min_cost = curr_cost
                best_node = neighborhood[i]

        node.parent = best_node
        node.cost = min_cost

        return best_node

    def visualize_path(self, og_msg, path, local_goal, show):

        width = og_msg.info.width
        height = og_msg.info.height
        grid = np.array(og_msg.data).reshape(height, width)

        x_ind, y_ind = self.coord_to_ind(local_goal.point.x, local_goal.point.y)
        grid[x_ind, y_ind] = self.occ * 2.0

        for i in range(len(path)):
            curr_node = path[i]
            x_ind, y_ind = self.coord_to_ind(curr_node.x, curr_node.y)
            grid[x_ind, y_ind] = self.occ * (i+1) / len(path)
        
        plt.matshow(grid)
        if show:
            plt.show()
            time.sleep(5)
            plt.close('all')

    # def pf_callback(self, pose_msg):
    #     """
    #     The pose callback when subscribed to particle filter's inferred pose
    #     Here is where the main RRT loop happens

    #     Args: 
    #         pose_msg (PoseStamped): incoming message from subscribed topic
    #     Returns:
        
    #     """
    #     return None

                        # try:
                    #     og[x_ind-1, y_ind] = self.occ/2.0
                    #     og[x_ind+1, y_ind] = self.occ/2.0
                    #     og[x_ind, y_ind-1] = self.occ/2.0
                    #     og[x_ind, y_ind+1] = self.occ/2.0


    # self.tf_listener = TransformListener()   
    # print("Map: ", self.tf_listener.frameExists("/map"))
    # print("Base_link: ", self.tf_listener.frameExists("/base_link"))
    # t = self.tf_listener.getLatestCommonTime("/map", "/base_link")
    # new_pose_map = self.tf_listener.transformPose("/map", new_pose)

    # if(listener.canTransform("map", new_pose.header.frame_id, new_pose.header.stamp)):
    # new_pose_map = listener.transformPose("/map", new_pose)

    # tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
    # tf_listener = tf2_ros.TransformListener(tf_buffer)

    # goal_pose = PoseStamped()
    # goal_pose.header.frame_id = 'map'
    # goal_pose.header.stamp = self.curr_pose.header.stamp
    # goal_pose.pose = Pose()
    # goal_pose.pose.position.x = global_goal.point.x
    # goal_pose.pose.position.y = global_goal.point.y


def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rospy.spin()

if __name__ == '__main__':
    main()