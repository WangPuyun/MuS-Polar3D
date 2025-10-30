import numpy as np
import imageio
import open3d as o3d


def ensure_mesh_from_ply(ply_path: str) -> o3d.geometry.TriangleMesh:
    """
    Reads a PLY file: returns the mesh directly if available. 
    If it's a point cloud, estimates normals, performs Poisson surface reconstruction, and clips the result to the point cloud's bounding box.
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():  # The input might be a point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise ValueError(f"Failed to read mesh or point cloud from file: {ply_path}")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=30))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def render_turntable_win(
    mesh: o3d.geometry.TriangleMesh,
    out_path="rotate_result.mp4",
    width=1080,
    height=1080,
    seconds=6,
    fps=30,
    axis="y",
    cam_dist_scale=1.8,
    bg_color=(255, 255, 255),
):
    """
    Renders a turntable animation on Windows using the classic Open3D Visualizer (windowed) 
    and exports it as an MP4 video.
    """
    mesh.paint_uniform_color([0.6, 0.4, 0.3])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Turntable", width=width, height=height, visible=True)

    # Background & Rendering Options
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color) / 255.0
    opt.mesh_show_back_face = True

    # Add geometry (create a copy for incremental rotation)
    mesh_to_draw = o3d.geometry.TriangleMesh(mesh)
    if not mesh_to_draw.has_vertex_normals():
        mesh_to_draw.compute_vertex_normals()
    if not mesh_to_draw.has_vertex_colors():
        mesh_to_draw.paint_uniform_color([0.92, 0.94, 0.98])

    # ==== Manual Pre-rotation (e.g., treat Z as Y or fix pitch) ====
    pre_rot_deg = (90, -90, 180)  # Example: Convert from Z-up to Y-up. Can be adjusted as (roll, pitch, yaw).
    R0 = o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad(pre_rot_deg))
    bbox0 = mesh_to_draw.get_axis_aligned_bounding_box()
    center0 = bbox0.get_center()
    mesh_to_draw.rotate(R0, center=center0)
    # ===============================================================

    vis.add_geometry(mesh_to_draw)

    # Viewpoint setup (using the ViewControl API)
    bbox = mesh_to_draw.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = float(np.linalg.norm(bbox.get_extent()))
    if extent == 0:
        extent = 1.0

    ctr = vis.get_view_control()
    cam_pos = center + np.array([0, 0, cam_dist_scale * extent])  # Camera faces the +Z direction
    front = (cam_pos - center) / np.linalg.norm(cam_pos - center) # Direction vector from the target to the camera
    ctr.set_lookat(center)
    ctr.set_front(front)   # Note: In Open3D, 'front' is defined as "the direction from the camera to the target". The above formula can be used.
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7)

    # Rotation parameters
    axis_map = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
    axis_vec = axis_map.get(axis.lower(), axis_map["y"])
    total_frames = int(seconds * fps)

    writer = imageio.get_writer(
        out_path,
        format="ffmpeg",  # ← ffmpeg is enforced here
        fps=fps,
        codec="libx264",
        quality=9
    )

    # Apply a small incremental rotation per frame (around the model's center)
    dtheta = 2 * np.pi / total_frames
    R_inc = mesh_to_draw.get_rotation_matrix_from_axis_angle(axis_vec * dtheta)

    for _ in range(total_frames):
        mesh_to_draw.rotate(R_inc, center=center)

        vis.update_geometry(mesh_to_draw)
        vis.poll_events()
        vis.update_renderer()

        # Screenshot (do_render=True is more reliable)
        img = vis.capture_screen_float_buffer(do_render=True)
        frame = (np.asarray(img) * 255).astype(np.uint8)
        writer.append_data(frame)

    writer.close()
    vis.destroy_window()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    # === Replace with your PLY file path ===
    ply_path = r"D:\postgraduate\bilateral_normal_integration\data\Dragon2_U2Net\mesh_k_2.ply"
    mesh = ensure_mesh_from_ply(ply_path)

    render_turntable_win(
        mesh,
        out_path="rotate_result.mp4",
        width=1080,
        height=1080,
        seconds=6,
        fps=30,
        axis="y",  # Allowed values: 'x' or 'z'
        cam_dist_scale=1.8,
        bg_color=(255, 255, 255),
    )
