import open3d as o3d
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import cv2
from PIL import Image
from sort import Sort
import pdb
from tf.transformations import quaternion_matrix
import pyrealsense2 as rs
import json


path = 'realsense_data_02.bag'
pth_model = 'model_complete_dict.pth'


def draw_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        color = (0, 255, 0)  # Green (RGB)
        box = box.cpu().numpy().astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, str(label), (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def initialize_model():
    global model
    num_classes = 2  # Replace with the number of classes in your dataset
    model = get_model(num_classes)
    model_path = pth_model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

def run_model(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    threshold = 0.95
    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]
    return filtered_boxes, filtered_labels

def undistort_depth_image(depth_image, intrinsics, distortion_coeffs):
    undistorted_depth_image = cv2.undistort(depth_image, intrinsics, distortion_coeffs)
    return undistorted_depth_image

def depth_to_point_cloud(depth_image, intrinsics):
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points = []
    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u]  #Add if you want in KMS/ 1000.0  # millimeters
            if Z > 0:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])
    return np.array(points)



def print_pose(pose_msg):
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation

    # Print
    print("Position:")
    print("x: {:.2f}, y: {:.2f}, z: {:.2f}".format(position.x, position.y, position.z))
    print("Orientation:")
    print(
        "x: {:.2f}, y: {:.2f}, z: {:.2f}, w: {:.2f}".format(orientation.x, orientation.y, orientation.z, orientation.w))


def pose_to_matrix(pose_msg):
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation

    # Convert quaternion to rotation matrix
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = quaternion_matrix(quaternion)

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
    transformation_matrix[:3, 3] = [position.x, position.y, position.z]

    return transformation_matrix

def invert_transformation_matrix(matrix):
    # Invert the rotation part
    rotation_inv = matrix[:3, :3].T

    # Invert the translation part
    translation_inv = -rotation_inv @ matrix[:3, 3]

    # Combine the inverted rotation and translation into a 4x4 matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = rotation_inv
    inverse_matrix[:3, 3] = translation_inv

    return inverse_matrix
def expand_bounding_box(bbox, expansion_pixels, image_width, image_height):
    x1, y1, x2, y2 = bbox
    new_x1 = x1 - expansion_pixels
    new_y1 = y1 - expansion_pixels
    new_x2 = x2 + expansion_pixels
    new_y2 = y2 + expansion_pixels

    # Make sure bb stay inside the margin of the img
    new_x1 = max(1, new_x1)
    new_y1 = max(1, new_y1)
    new_x2 = min(image_width-1, new_x2)
    new_y2 = min(image_height-1, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

def is_vertical(plane_model, threshold=0.1):
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    return abs(abs(normal[2]) - 1) < threshold

def is_horizontal(plane_model, threshold =0.1):
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    return abs(normal[2]) < threshold


# Function to set a fixed camera view
def set_camera_view():
    view_control = vis.get_view_control()
    # Set custom camera parameters
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Modify the camera extrinsic parameters for a custom view
    camera_params.extrinsic = np.array([
        [0.5, 0.0, 0.866, 0.0],  # Rotate 30 degrees around Y-axis (cos(30°) ≈ 0.866)
        [0.0, 1.0, 0.0, -2.0],  # Move camera along Y-axis
        [-0.866, 0.0, 0.5, 2.0],  # Rotate 30 degrees around Y-axis and move along Z-axis
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Apply the custom camera parameters
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)


# Function to create a video for a set of point clouds


def create_video_for_scene(scene_name, pcd_array,height, width):
    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1400, height=1050, visible=True)
    # Prepare to capture frames
    frames = []
    #vis.add_geometry(pcd_paths[0])
    # Access the view control object

    view_control =vis.get_view_control()
    view_control.camera_local_translate(forward= 1000.0, right= 1000.0, up= 0.0)
    '''
    view_control = vis.get_view_control()
    # Configurar los parámetros de la cámara
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Establecer la cámara para que observe desde el origen
    camera_position = np.array([0.0, 0.0, 5.0])  # Cámara ubicada a 5 unidades de distancia en el eje Z
    lookat = np.array([0.0, 0.0, 0.0])  # Punto de mira en el origen
    up_direction = np.array([0.0, 1.0, 0.0])  # Dirección "arriba" del eje Y
    # Calcular la dirección de la cámara (desde el origen hacia el punto de mira)
    forward = (lookat - camera_position)
    forward /= np.linalg.norm(forward)

    # Calcular el vector derecha usando el producto cruzado
    right = np.cross(up_direction, forward)
    right /= np.linalg.norm(right)

    # Calcular el nuevo vector "arriba" (up)
    up = np.cross(forward, right)

    # Crear la matriz de rotación (3x3)
    rotation_matrix = np.vstack([right, up, forward]).T

    # Crear la matriz de extrínsecos (4x4) combinando rotación y traslación
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = -rotation_matrix @ camera_position

    # Asignar la matriz de extrínsecos a los parámetros de la cámara
    camera_params.extrinsic = extrinsic_matrix

    # Aplicar los parámetros de la cámara
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    '''
    # Load and render each point cloud
    for index, pcd in enumerate(pcd_array):





        vis.add_geometry(pcd)
        #set_camera_view(vis)

        vis.poll_events()
        vis.update_renderer()


    vis.clear_geometries()
    for index, pcd in enumerate(pcd_array):
        vis.add_geometry(pcd)
        # set_camera_view(vis)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'p{index}.jpg')
        # Capture the frame
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=False))
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
        vis.clear_geometries()

    vis.destroy_window()

    # Define video writer
    height, width, _ = frames[0].shape
    video_path = f'{scene_name}_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

    # Write frames to video
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

    print(f"Video saved to {video_path}")
    vis.destroy_window()

def draw_registration_result(source, target, transformation):
    source_temp = source.transform(transformation)
    source_temp.paint_uniform_color([1, 0, 0])  # Source in red
    target_temp = target.paint_uniform_color([0, 1, 0])  # Target in green
    o3d.visualization.draw_geometries([source_temp, target_temp])


def combine_pcds_with_color(pcd_files):
    """
    Combine multiple PCD files into a single PCD, each with a unique color.

    Parameters:
        pcd_files (list of str): List of file paths to PCD files.

    Returns:
        o3d.geometry.PointCloud: Combined point cloud with unique colors for each PCD.
    """
    combined_pcd = o3d.geometry.PointCloud()
    colors = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [1, 0.5, 0],  # Orange
        [0.5, 0, 0.5],  # Purple
        [0.5, 0.5, 0.5]  # Gray
    ])

    # Loop through each PCD file
    for i, pcd in enumerate(pcd_files):


        # Assign color
        color = colors[i % len(colors)]
        pcd.paint_uniform_color(color)

        # Combine with the main point cloud
        combined_pcd += pcd

    return combined_pcd


def main():
    bag = rosbag.Bag(path)
    initialize_model()
    tracker = Sort()
    #480p depth intrinsics (if not aligned)
    '''
    intrinsics_matrix = np.array([
        [381.46832275390625, 0, 316.8101501464844],
        [0, 381.46832275390625, 235.5249786376953],
        [0, 0, 1]
    ])

    fx=381.46832275390625
    fy=381.46832275390625
    cx=316.8101501464844
    cy=235.5249786376953
    '''

    #480p COLOR intrinsics
    intrinsics_matrix = np.array([
        [610.047119140625, 0, 328.30047607421875],
        [0, 609.0606689453125, 247.893798828125],
        [0, 0, 1]
    ])

    fx = 610.047119140625
    fy = 609.0606689453125
    cx = 328.30047607421875
    cy = 247.893798828125
    '''
    #720p depth intrinsics
    intrinsics = np.array([
        [635.780517578125, 0, 634.68359375],
        [0, 635.780517578125, 352.5416259765625],
        [0, 0, 1]
    ])

    fx = 635.780517578125
    fy = 635.780517578125
    cx = 634.68359375
    cy = 352.5416259765625
    '''

    '''
    #720p color intrinsics
    intrinsics = np.array([
        [635.780517578125, 0, 634.68359375],
        [0, 635.780517578125, 352.5416259765625],
        [0, 0, 1]
    ])

    fx = 915.0706787109375
    fy = 913.5910034179688
    cx = 652.45068359375
    cy = 371.8406982421875
    '''


    distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


    old_point_clouds = {}
    all_point_clouds = {}  # Dictionary to store lists of all point clouds for each track_id
    transformations = {}  # Dictionary to store transformations for each track_id
    centroid_positions = {}
    counter=0

    try:

        last_msg1, last_msg2, last_msg3 = None, None, None

        # Preprocesa todos los mensajes y guarda el último
        all_messages = list(zip(
            bag.read_messages(topics=['/camera1/color/image_raw']),
            bag.read_messages(topics=['/camera1/depth/image_raw']),
            bag.read_messages(topics=['/camera2/pose'])
        ))


        last_msg1, last_msg2, last_msg3 = all_messages[-1]

        for msg1,msg2,msg3, in zip(bag.read_messages(topics=['/camera1/color/image_raw']),bag.read_messages(topics=['/camera1/depth/image_raw']),bag.read_messages(topics=['/camera2/pose'])):

            if counter % 15 != 0:
                counter += 1
                if((msg1, msg2, msg3) != (last_msg1, last_msg2, last_msg3)):
                    continue  # Salta la iteración actual
            counter += 1
            point_clouds = {}  # Dictionary to store point clouds for each track_id
            bridge = CvBridge()
            cv2imgcolor = bridge.imgmsg_to_cv2(msg1.message, desired_encoding="bgr8")
            #cv2.imshow('rgb', cv2imgcolor)
            cv2.waitKey(1)

            cv2imgdepth = bridge.imgmsg_to_cv2(msg2.message, desired_encoding="16UC3")

            # Grayish depth img to colors
            depth_8bit = cv2.normalize(cv2imgdepth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            #cv2.imshow('depth', depth_colormap)
            cv2.waitKey(1)

            image_rgb = cv2.cvtColor(cv2imgcolor, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            #cv2undistorteddepthimg = undistort_depth_image(cv2imgdepth, intrinsics, distortion_coeffs)

            t265matrix = pose_to_matrix(msg3.message)
            inverted_t265matrix = invert_transformation_matrix(t265matrix)
            width = cv2imgcolor.shape[1]
            height = cv2imgcolor.shape[0]
            rgb_image_o3d = o3d.geometry.Image(cv2.cvtColor(cv2imgcolor, cv2.COLOR_BGR2RGB))
            depth_image_o3d = o3d.geometry.Image(cv2imgdepth)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d, depth_scale=1560)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d,  depth_scale=1000, depth_trunc=10e9, convert_rgb_to_intensity=False)
            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)


            cloud.transform([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

            #cloud.transform(inverted_t265matrix)
            print("t265matrix ES: ")
            print(t265matrix)
            print("inverted_t265matrix ES:")
            print(inverted_t265matrix)
            o3d.visualization.draw_geometries([cloud], window_name="Escena original")

            '''
            '''
            vertical_planes = []
            while True:
                print("Filtrando")
                plane_model, inliers = cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)

                if len(inliers) < 150000:
                    break

                if is_vertical(plane_model):
                    vertical_plane = cloud.select_by_index(inliers)
                    vertical_planes.append(vertical_plane)

                # Actualiza la nube de puntos eliminando los puntos del plano detectado
                cloud = cloud.select_by_index(inliers, invert=True)

            # Visualiza la nube restante
            #o3d.visualization.draw_geometries([cloud], window_name="Filtrando planos verticales")
            '''
            '''
            points = np.asarray(cloud.points)
            distance_threshold = 1.75
            distances = np.linalg.norm(points, axis=1)
            mask = distances < distance_threshold
            filtered_points = points[mask]
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            #o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtrando fondo")

            floor = 0
            limit=10
            while floor == 0 :

                plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
                cloud_plane = filtered_pcd.select_by_index(inliers)

                if is_horizontal(plane_model):
                    cloud_without_plane = filtered_pcd.select_by_index(inliers, invert=True)
                    floor=1
                    a, b, c, d = plane_model

                    # Create a numpy array of point cloud points
                    points = np.asarray(cloud_without_plane.points)

                    # Calculate the distance of each point from the plane
                    distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

                    # Define a side to remove: for example, removing points where distances > 0
                    side_to_remove = distances < 0

                    # Create a mask of points to keep
                    keep_mask = ~side_to_remove

                    # Filter the point cloud
                    filtered_points = points[keep_mask]
                    cloud_without_planev2 = o3d.geometry.PointCloud()
                    cloud_without_planev2.points = o3d.utility.Vector3dVector(filtered_points)
                if limit > 10:
                    print("PLano suelo no encontrado")
                    cloud_without_planev2 = o3d.geometry.PointCloud()
                    cloud_without_planev2.points = o3d.utility.Vector3dVector(filtered_points)
                    break

                limit = limit + 1

            #o3d.visualization.draw_geometries([cloud_without_planev2], window_name="Escena filtrando plano suelo")

            filtered_boxes, filtered_labels = run_model(image_pil)
            draw_boxes(cv2imgcolor, filtered_boxes, filtered_labels)
            cv2.imshow('Object Detection', cv2imgcolor)
            #cv2.waitKey(1)

            detections = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, 1.0])  # 1.0 is confidence score
            print(detections)
            if len(detections) == 0:
                trackers = tracker.update(np.empty((0, 5)))
            else:
                trackers = tracker.update(np.array(detections))
            print(trackers)
            for track in trackers:
                x_1, y_1, x_2, y_2, track_id = track.astype(int)
                #height, width = cv2undistorteddepthimg.shape[:2]
                #print ("height is:", height)
                #print ("width is :", width)
                #x_1, y_1, x_2, y_2 = expand_bounding_box([x_1, y_1, x_2, y_2], 50, width, height)
                x_1 = int(x_1)
                y_1 = int(y_1)
                x_2 = int(x_2)
                y_2 = int(y_2)
                print("El track ID es: ", track_id)

                mask = np.zeros(cv2imgdepth.shape, dtype=np.uint8)
                mask[y_1:y_2, x_1:x_2] = 1
                # mask[y_1 - 50:y_2 + 50, x_1 - 50:x_2 + 50] = 1
                masked_depth_image = np.where(mask, cv2imgdepth, 0)
                masked_depth_image_o3d = o3d.geometry.Image(masked_depth_image)

                #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, masked_depth_image_o3d,depth_scale=1560)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, masked_depth_image_o3d,
                                                                          convert_rgb_to_intensity=False)
                cloudBB = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

                cloudBB.transform([[1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]])

                #cloudBB.transform(inverted_t265matrix)
                #o3d.visualization.draw_geometries([cloudBB], window_name="Escena bounding box")

                try:
                    points1 = np.asarray(cloud_without_planev2.points)
                    #colors1 = np.asarray(cloud_without_planev2.colors)
                    points2 = np.asarray(cloudBB.points)
                    colors2 = np.asarray(cloudBB.colors)
                    tolerance = 1e-6
                    set1 = set(map(tuple, points1))
                    set2 = set(map(tuple, points2))
                    intersected_points = np.array(list(set1 & set2))



                    # Filtrar los colores correspondientes a los puntos coincidentes
                    intersected_colors = []

                    # Crear un diccionario de puntos a colores para `cloudBB`
                    point_to_color_dict = {tuple(point): color for point, color in zip(points2, colors2)}

                    for point in intersected_points:
                        point_tuple = tuple(point)
                        if point_tuple in point_to_color_dict:
                            intersected_colors.append(point_to_color_dict[point_tuple])


                    intersected_colors = np.array(intersected_colors)


                    coincident_pcd = o3d.geometry.PointCloud()
                    coincident_pcd.points = o3d.utility.Vector3dVector(intersected_points)
                    coincident_pcd.colors = o3d.utility.Vector3dVector(intersected_colors)


                    #o3d.visualization.draw_geometries([coincident_pcd])
                    coincident_pcd.transform(t265matrix)
                    #o3d.visualization.draw_geometries([coincident_pcd], window_name="Puntos Coincidentes")

                    point_clouds[track_id] = coincident_pcd


                    if track_id not in all_point_clouds:
                        all_point_clouds[track_id] = []
                        centroid_positions[track_id] = []
                    all_point_clouds[track_id].append(coincident_pcd)

                    coincident_points = np.asarray(coincident_pcd.points)
                    centroid = np.mean(coincident_points, axis=0)
                    t_x, t_y, t_z = centroid
                    transformation_matrix = np.array([
                        [1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0, 1]])
                    centroid_positions[track_id].append(transformation_matrix)

                    if track_id not in transformations:
                        transformations[track_id] = []

                        coincident_points = np.asarray(coincident_pcd.points)
                        centroid = np.mean(coincident_points, axis=0)
                        t_x, t_y, t_z = centroid
                        transformation_matrix = np.array([
                            [1, 0, 0, t_x],
                            [0, 1, 0, t_y],
                            [0, 0, 1, t_z],
                            [0, 0, 0, 1] ])
                        print("transformation_matrix del centroide ES:", transformation_matrix)
                        #Edited to show only composed transformation from beginning for case examples in chapter 6
                        #transformations[track_id].append(transformation_matrix)
                        transformations[track_id].append(np.eye(4))


                        #CENTROID
                        # Create a small sphere at the centroid position
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                        sphere.translate(centroid)
                        sphere.paint_uniform_color([0, 1, 0])  # Red color

                        # Visualize the point cloud and the centroid
                        vis = o3d.visualization.Visualizer()
                        vis.create_window()

                        # Add the original point cloud
                        vis.add_geometry(coincident_pcd)

                        # Add the sphere representing the centroid
                        vis.add_geometry(sphere)

                        # Run the visualizer
                        vis.run()


                    if track_id in old_point_clouds:
                        source = old_point_clouds[track_id]
                        target = point_clouds[track_id]

                        source_points = np.asarray(source.points)
                        source_centroid = np.mean(source_points, axis=0)
                        t_x, t_y, t_z = source_centroid
                        t_m_source_centroid = np.array([
                            [1, 0, 0, t_x],
                            [0, 1, 0, t_y],
                            [0, 0, 1, t_z-0.04],
                            [0, 0, 0, 1]])
                        inv_t_m_source_centroid = invert_transformation_matrix(t_m_source_centroid)



                        target_points = np.asarray(target.points)
                        target_centroid = np.mean(target_points, axis=0)
                        t_x, t_y, t_z = target_centroid
                        t_m_target_centroid = np.array([
                            [1, 0, 0, t_x],
                            [0, 1, 0, t_y],
                            [0, 0, 1, t_z],
                            [0, 0, 0, 1]])
                        inv_t_m_target_centroid = invert_transformation_matrix(t_m_target_centroid)

                        scale = 15

                        source_traslated = o3d.geometry.PointCloud()
                        source_traslated.points = o3d.utility.Vector3dVector(np.asarray(source.points))
                        source_traslated.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
                        source_traslated.transform(inv_t_m_source_centroid)
                        source_traslated.points = o3d.utility.Vector3dVector(np.asarray(source_traslated.points)*scale)
                        source_traslated.colors = o3d.utility.Vector3dVector(np.asarray(source_traslated.colors)*scale)
                        source_traslated.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                        alpha = 0.07  # Ajustar el parámetro alpha para obtener una buena forma
                        mesh_source = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(source_traslated, alpha)
                        mesh_pcd_source = mesh_source.sample_points_uniformly(number_of_points=80000)
                        mesh_pcd_source.paint_uniform_color([1, 0, 0])


                        target_traslated = o3d.geometry.PointCloud()
                        target_traslated.points = o3d.utility.Vector3dVector(np.asarray(target.points))
                        target_traslated.colors = o3d.utility.Vector3dVector(np.asarray(target.colors))
                        target_traslated.transform(inv_t_m_source_centroid)
                        target_traslated.points = o3d.utility.Vector3dVector(np.asarray(target_traslated.points)*scale)
                        target_traslated.colors = o3d.utility.Vector3dVector(np.asarray(target_traslated.colors)*scale)
                        target_traslated.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                        mesh_target = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(target_traslated,alpha)
                        mesh_pcd_target = mesh_target.sample_points_uniformly(number_of_points=80000)
                        mesh_pcd_target.paint_uniform_color([0, 1, 0])

                        #o3d.visualization.draw_geometries([mesh_pcd_source, mesh_pcd_target], window_name="MESH")
                        #o3d.visualization.draw_geometries([source, target], window_name="PREVIOUS ICP")
                        composed_transformation =np.eye(4)
                        for trans in transformations[track_id]:
                            composed_transformation = composed_transformation @ trans
                        giro = np.eye(4)
                        giro[:3,:3] = np.linalg.inv(composed_transformation[:3,:3])

                        mesh_pcd_source.transform(giro)
                        mesh_pcd_target.transform(giro)




                        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120000)

                        threshold = 0.045
                        trans_init = np.identity(4)




                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            mesh_pcd_source,
                            mesh_pcd_target,
                            threshold,
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                            criteria
                        )
                        #o3d.visualization.draw_geometries([target_traslated, source_traslated], window_name="PREVIOUS COLORED ICP")




                        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)





                        scaled_reg_p2p = np.copy(reg_p2p.transformation)
                        scaled_reg_p2p[0, 3] /= scale  # t_x / 3
                        scaled_reg_p2p[1, 3] /= scale  # t_y / 3
                        scaled_reg_p2p[2, 3] /= scale  # t_z / 3

                        print("Transformation NEW matrix:")
                        print(scaled_reg_p2p)

                        #transformations[track_id].append(inv_t_m_source_centroid @ reg_p2p.transformation @ t_m_target_centroid)
                        transformations[track_id].append(scaled_reg_p2p)
                        mesh_pcd_source.transform(reg_p2p.transformation)



                        #Draw ICP graph between frames

                        trans_init = np.identity(4)
                        #draw_registration_result(mesh_pcd_source, mesh_pcd_target, trans_init)



                except Exception as e:
                    print("Exception is: ", e)
            old_point_clouds = point_clouds.copy()
            print(old_point_clouds)


        for track_id, trans_list in transformations.items():
            composed_transformation = np.eye(4)
            for trans in trans_list:

                composed_transformation = composed_transformation @ trans
            print(f"Composed transformation for track_id {track_id}:")
            print(composed_transformation)


        # Create a video for each scene
        for track_id, pcd in all_point_clouds.items():
            create_video_for_scene(track_id, pcd,height, width)

        for track_id, pcd_array in all_point_clouds.items():
            combined_pcd = combine_pcds_with_color(pcd_array)
            #o3d.visualization.draw_geometries([combined_pcd])

        for track_id, pcd_array in all_point_clouds.items():
            source = pcd_array[0]
            target = pcd_array[-1]

            coincident_points = np.asarray(source.points)
            centroid = np.mean(coincident_points, axis=0)
            t_x, t_y, t_z = centroid
            transformation_matrix = np.array([
                [1, 0, 0, t_x],
                [0, 1, 0, t_y],
                [0, 0, 1, t_z],
                [0, 0, 0, 1]])
            print("transformation_matrix del centroide SOURCE ES:", transformation_matrix)
            result_of_centroid_source = np.copy(transformation_matrix)
            coincident_points = np.asarray(target.points)
            centroid = np.mean(coincident_points, axis=0)
            t_x, t_y, t_z = centroid
            transformation_matrix = np.array([
                [1, 0, 0, t_x],
                [0, 1, 0, t_y],
                [0, 0, 1, t_z],
                [0, 0, 0, 1]])
            print("transformation_matrix del centroide TARGET ES:", transformation_matrix)

            o3d.visualization.draw_geometries([source], window_name="source")
            o3d.visualization.draw_geometries([target], window_name="target")



            # Estimate normals for the source point cloud
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Estimate normals for the target point cloud
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            threshold = 0.02
            trans_init = np.identity(4)
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source,
                target,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria
            )

            print(f"FINAL transformation for {track_id} is:")
            print(reg_p2p.transformation)

        for track_id, trans_list in transformations.items():
            puntos_trayectoria = []
            ejes = []
            ejes_2 = []
            print(trans_list)
            # Initialize composition with  identity matrix
            counter = 0
            for msg in bag.read_messages(topics=['/camera2/pose']):
                if counter % 20 != 0:
                    counter += 1
                    continue
                counter += 1
                t265matrix = pose_to_matrix(msg.message)
                eje_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                eje_2.transform(t265matrix)
                ejes_2.append(eje_2)


            acumulada= np.copy(result_of_centroid_source)
            for i in trans_list:


                acumulada = acumulada @ i
                print("transformacion_acumulada ES:")
                print(acumulada)

                # Extrae la traslación de la transformación acumulada
                punto = acumulada[:3, 3]
                puntos_trayectoria.append(punto)

                # Crea un conjunto de ejes en la posición actual
                eje = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                eje.transform(acumulada)
                ejes.append(eje)
            puntos = o3d.utility.Vector3dVector(puntos_trayectoria)
            lineas = [[i, i + 1] for i in range(len(puntos_trayectoria) - 1)]
            colores = [[1, 0, 0] for _ in range(len(lineas))]  # Color rojo para las líneas

            line_set = o3d.geometry.LineSet(points=puntos, lines=o3d.utility.Vector2iVector(lineas))
            line_set.colors = o3d.utility.Vector3dVector(colores)

            # Visualizar todo
            o3d.visualization.draw_geometries([line_set] + ejes+ ejes_2)
    finally:
        bag.close()
        # Specify the file path where you want to save the JSON file
        file_path = 'info.json'

        # Save the dictionary to a JSON file
        for track_id, trans_list in transformations.items():
            np.savez('arrays_file.npz', *trans_list)

if __name__ == '__main__':
    main()