import open3d as o3d
import numpy as np
import json
import cv2

data_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reform_SG/burger"


# Currently this doesn't consider different fx and fy
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        if not z_flip:
            shoot_points.append(
                np.dot(transformation, np.array([0, 0, -length, 1]))[0:3]
            )
        else:
            shoot_points.append(
                np.dot(transformation, np.array([0, 0, length, 1]))[0:3]
            )
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


if __name__ == "__main__":
    # Load the images and the camera parameters
    # Load the camera parameters
    with open(f"{data_path}/camera_info.json", "r") as f:
        camera_info = json.load(f)

    intrinsic = np.array(camera_info["intrinsic"])
    c2ws = camera_info["c2ws"]
    cameras = list(c2ws.keys())

    visuals = []
    for index in range(20):
        object_masks = []
        for camera in cameras:
            object_mask = cv2.imread(f"{data_path}/masks/{camera}/{index}.png", cv2.IMREAD_GRAYSCALE)
            object_masks.append(object_mask)
        object_masks = np.stack(object_masks, axis=0)

        data = np.load(f"results/points_colors_{index}.npz")
        pts3d = data["pts3d"]
        imgs = data["imgs"]
        confidence_masks = data["confidence_masks"]
        final_masks = np.logical_and(confidence_masks, object_masks)
        
        points = pts3d[final_masks]
        colors = imgs[final_masks]
        focals = data["focals"]
        c2ws = data["poses"]
        pp = data["pp"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        visuals.append(pcd)

        if index == 0:
            camera_visuals = []
            for i in range(len(c2ws)):
                camera_visual = getCamera(
                    c2ws[i],
                    focals[i][0],
                    focals[i][0],
                    pp[i][0],
                    pp[i][1],
                    scale=0.5,
                    z_flip=True,
                )
                camera_visuals += camera_visual

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            visuals += [coordinate] + camera_visuals

    o3d.visualization.draw_geometries(visuals)
