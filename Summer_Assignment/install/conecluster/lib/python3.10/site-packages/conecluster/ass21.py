import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sklearn.cluster import DBSCAN
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
        


        pc_data = np.array([[point.x, point.y, point.z] for point in msg.points])
        
        pc_data = np.column_stack((pc_data, np.array(msg.channels[0].values)))
        #pc_data[:,0]+=2.921
        pc_data[:,3]/=np.max(pc_data[:,3])



        pc_data = pc_data[pc_data[:, 2] > -0.15]
        int_data= pc_data[:, :3]  


        if pc_data.shape[0] == 0:
            return

        db = DBSCAN(eps=0.75, min_samples=4).fit(int_data)
        labels = db.labels_
        unique_labels = set(labels)
        markerarr = MarkerArray()

        
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markerarr.markers.append(delete_marker)

        self.marker_id= 0

        for i in unique_labels:
            if i == -1:
                continue
            cluster = pc_data[labels == i]
            x, y = np.mean(cluster[:, :2], axis=0)

            # top_intensity = cluster[(cluster[:, 2] > 0.1) & (cluster[:, 2] < 0.14)]
            # mid_intensity = cluster[(cluster[:, 2] > -0.01) & (cluster[:, 2] < 0)]

            # if top_intensity.size == 0 or mid_intensity.size == 0:
            #     continue


            # top_int=np.max(top_intensity[:,3], axis=0)
            # mid_int=np.mean(mid_intensity[:,3], axis=0)


            # if(top_int-mid_int > 1e-10):
            marker = Marker()
            marker.header.frame_id = "Lidar_F"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "yellowcones"
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
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=0.05).to_msg()
            # marker.lifetime.sec= 1
            markerarr.markers.append(marker)
            self.marker_id += 1
            # else:
            #     marker = Marker()
            #     marker.header.frame_id = "Lidar_F"
            #     marker.header.stamp = self.get_clock().now().to_msg()
            #     marker.ns = "bluecones"
            #     marker.id = self.marker_id
            #     marker.type = Marker.CYLINDER
            #     marker.action = Marker.ADD
            #     marker.pose.position.x = float(x)
            #     marker.pose.position.y = float(y)
            #     marker.pose.position.z = 0.155
            #     marker.pose.orientation.w = 1.0
            #     marker.scale.x = 0.2
            #     marker.scale.y = 0.2
            #     marker.scale.z = 0.31
            #     marker.color.r = 0.0
            #     marker.color.g = 0.0
            #     marker.color.b = 1.0
            #     marker.color.a = 1.0
            #     # marker.lifetime = Duration(seconds=0.1).to_msg()
            #     # marker.lifetime.sec= 1
            #     markerarr.markers.append(marker)
            #     self.marker_id += 1

        self.publisher.publish(markerarr)

def main(args=None):
    rclpy.init(args=args)
    node = conesdetect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()