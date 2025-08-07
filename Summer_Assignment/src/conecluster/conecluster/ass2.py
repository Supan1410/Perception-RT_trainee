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

    def remove_ground_ransac(self, pc_data):
        """
        Remove ground using RANSAC fit to z = f(x, y)
        """
        X = pc_data[:, :2]
        y = pc_data[:, 2]
        if len(X) < 10:
            return pc_data

        ransac = RANSACRegressor(residual_threshold=0.02)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        return pc_data[~inlier_mask]

    def classify_color(self, cluster):
        z = cluster[:, 2]
        intensity = cluster[:, 3]

        top_band = (z >= 0.08)
        mid_band = (z >= -0.02) & (z < 0.01)
        bot_band = (z < -0.09)

        # Use mean even if only 1 point is in the band
        top_int = np.mean(intensity[top_band]) if np.any(top_band) else 0.0
        mid_int = np.mean(intensity[mid_band]) if np.any(mid_band) else 0.0
        bot_int = np.mean(intensity[bot_band]) if np.any(bot_band) else 0.0

        # Classification logic
        if (top_int == 0.0 or bot_int==0.0) or (mid_int == 0.0) :
            return 'white'
        elif top_int - mid_int > 0:
            return 'yellow'
        elif top_int - mid_int < 0:
            return 'blue'
        elif bot_int - mid_int > 0:
            return 'yellow'
        elif bot_int-mid_int < 0:
            return 'blue'
        else:
            return 'white'
        

    def cluster(self, msg):
        if not msg.points:
            return

        pc_data = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensities = np.array(msg.channels[0].values)
        intensities /= np.max(intensities) if np.max(intensities) != 0 else 1.0
        pc_data = np.column_stack((pc_data, intensities))

        pc_data = self.remove_ground_ransac(pc_data)

        if pc_data.shape[0] == 0:
            return

        clustering = DBSCAN(eps=1, min_samples=2).fit(pc_data[:, :2])
        labels = clustering.labels_
        unique_labels = set(labels)

        markerarr = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markerarr.markers.append(delete_marker)
        self.marker_id = 0

        for label in unique_labels:
            if label == -1:
                continue
            cluster = pc_data[labels == label]
            if cluster.shape[0] < 3:
                continue

            x, y = np.mean(cluster[:, :2], axis=0)

            color_label = self.classify_color(cluster)
            if color_label == 'yellow':
                color = (1.0, 1.0, 0.0)
            elif color_label == 'blue':
                color = (0.0, 0.0, 1.0)
            elif color_label == 'white':
                continue
                color = (1.0, 1.0, 1.0)

            marker = Marker()
            marker.header.frame_id = "Lidar_F"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cones"
            marker.id = self.marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.155
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.31
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=1).to_msg()

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
