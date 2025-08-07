import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray


class conesdetect(Node):
    def __init__(self):
        super().__init__('conesdetect')
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.cluster, 10)
        self.publisher = self.create_publisher(MarkerArray, '/cones', 10)
        self.marker_id = 0

    def cluster(self, msg):
        if not msg.points:
            return

        # Convert point cloud to numpy array
        pc_data = np.array([[point.x, point.y, point.z] for point in msg.points])

        # Add intensity channel
        intensities = np.array(msg.channels[0].values)
        if np.max(intensities) == 0:
            return
        intensities /= np.max(intensities)
        pc_data = np.column_stack((pc_data, intensities))

        # Step 1: Ground Removal using RANSAC
        X = pc_data[:, :2]  # x, y
        Z = pc_data[:, 2]   # z (height)

        try:
            ransac = RANSACRegressor(residual_threshold=0.05)
            ransac.fit(X, Z)
            inlier_mask = ransac.inlier_mask_
            pc_data = pc_data[~inlier_mask]  # Remove ground points
        except Exception as e:
            self.get_logger().warn(f"RANSAC failed: {e}")
            return

        if pc_data.shape[0] == 0:
            return

        # Step 2: Clustering using DBSCAN
        int_data = pc_data[:, :2]  # Use x, y for clustering
        db = DBSCAN(eps=2, min_samples=2).fit(int_data)
        labels = db.labels_
        unique_labels = set(labels)

        # Step 3: Prepare MarkerArray
        markerarr = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markerarr.markers.append(delete_marker)

        self.marker_id = 0

        for i in unique_labels:
            if i == -1:
                continue  # Skip noise

            cluster = pc_data[labels == i]
            x, y = np.mean(cluster[:, :2], axis=0)

            marker = Marker()
            marker.header.frame_id = "Lidar_F"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "yellowcones"
            marker.id = self.marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.155  # Fixed height
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.31
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=0.5).to_msg()

            markerarr.markers.append(marker)
            self.marker_id += 1

        self.publisher.publish(markerarr)


def main(args=None):
    rclpy.init(args=args)
    node = conesdetect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
