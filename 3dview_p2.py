import time
import numpy as np
import imageio
import open3d as o3d


def ensure_mesh_from_ply(ply_path: str) -> o3d.geometry.TriangleMesh:
    """
    Read a PLY file.
    - If it already contains a triangle mesh, return it.
    - If it is a point cloud, estimate normals and reconstruct a mesh with Poisson.
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise ValueError(f"Failed to read mesh or point cloud from file: {ply_path}")

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def _apply_pre_rotation(mesh: o3d.geometry.TriangleMesh, pre_rot_deg=(90, -90, 180)):
    """Rotate the mesh once before rendering, to fix its initial pose."""
    center = mesh.get_axis_aligned_bounding_box().get_center()
    R0 = o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad(pre_rot_deg))
    mesh.rotate(R0, center=center)
    return mesh


def _make_pingpong_yaws(
    total_frames: int,
    total_yaw_deg: float,
    cycles: float = 1.0,
    yaw_offset_deg: float = 0.0,
    start_at_center: bool = True,
):
    """
    Create yaw angles for an orbit camera with back-and-forth motion.

    Parameters
    ----------
    total_frames : int
        Number of frames.
    total_yaw_deg : float
        Total left-right swing angle in degrees.
    cycles : float
        How many back-and-forth cycles to complete over the whole video.
        1.0 means one full cycle.
    yaw_offset_deg : float
        Global yaw offset in degrees. Use this to choose the initial side/front view.
        Example: +90 or -90 often changes front view to side view.
    start_at_center : bool
        True  -> first frame starts at the center view, then swings left/right.
        False -> first frame starts at one endpoint, then swings across.
    """
    amp = np.deg2rad(total_yaw_deg) / 2.0
    offset = np.deg2rad(yaw_offset_deg)
    t = np.linspace(0.0, 1.0, total_frames, endpoint=False)

    if start_at_center:
        # center -> right -> center -> left -> center ...
        yaws = offset + amp * np.sin(2.0 * np.pi * cycles * t)
    else:
        # left endpoint -> right endpoint -> left endpoint ...
        yaws = offset - amp * np.cos(2.0 * np.pi * cycles * t)

    return yaws


def render_orbit_video_win(
    mesh: o3d.geometry.TriangleMesh,
    out_path="orbit_result.mp4",
    width=1080,
    height=1080,
    seconds=10,
    fps=30,
    bg_color=(255, 255, 255),
    mesh_rgb=(0.78, 0.81, 0.66),
    zoom=0.72,
    total_yaw_deg=40.0,
    elev_deg=15.0,
    cycles=2.0,
    yaw_offset_deg=0.0,
    start_at_center=True,
    pre_rot_deg=(90, -90, 180),
    mesh_show_back_face=True,
    brightness=0.86,
    use_lighting=True,
):
    """
    Render a Windows-friendly orbit video using the classic Open3D Visualizer.

    Key idea:
    - mesh is fixed
    - camera moves around the mesh center
    - yaw angle uses ping-pong motion, so the result has both clockwise and counterclockwise motion
    - yaw_offset_deg changes which side is shown at the start

    Extra controls:
    - brightness < 1.0 will darken captured frames to reduce overexposure
    - cycles controls how many back-and-forth swings happen in the whole video
    - seconds controls total video duration
    """
    total_frames = max(1, int(seconds * fps))
    elev = np.deg2rad(elev_deg)

    # Make a copy so the input mesh is not modified.
    mesh_to_draw = o3d.geometry.TriangleMesh(mesh)
    if not mesh_to_draw.has_vertex_normals():
        mesh_to_draw.compute_vertex_normals()

    # Fix initial pose.
    _apply_pre_rotation(mesh_to_draw, pre_rot_deg=pre_rot_deg)

    # Force a uniform color, even if the PLY originally contains vertex colors.
    mesh_to_draw.paint_uniform_color(mesh_rgb)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D Orbit Renderer",
        width=width,
        height=height,
        visible=True,
    )

    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color, dtype=np.float64) / 255.0
    opt.mesh_show_back_face = mesh_show_back_face
    opt.light_on = bool(use_lighting)

    vis.add_geometry(mesh_to_draw)

    bbox = mesh_to_draw.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(float(zoom))

    # Warm up a little to avoid black / stale first frames on some Windows setups.
    for _ in range(8):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    yaws = _make_pingpong_yaws(
        total_frames=total_frames,
        total_yaw_deg=total_yaw_deg,
        cycles=cycles,
        yaw_offset_deg=yaw_offset_deg,
        start_at_center=start_at_center,
    )

    writer = imageio.get_writer(
        out_path,
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        quality=9,
    )

    try:
        for yaw in yaws:
            # Open3D front vector is the direction from camera toward target.
            front = np.array([
                np.sin(yaw) * np.cos(elev),
                -np.sin(elev),
                -np.cos(yaw) * np.cos(elev),
            ], dtype=np.float64)
            front /= np.linalg.norm(front)

            ctr.set_lookat(center)
            ctr.set_front(front)
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(float(zoom))

            vis.poll_events()
            vis.update_renderer()

            img = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
            img = np.clip(img * float(brightness), 0.0, 1.0)
            frame = (img * 255.0).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()
        vis.destroy_window()

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    # Replace this with your own PLY file path.
    ply_path = r"E:\postgraduate\bilateral_normal_integration\data\Dragon2_U2Net\mesh_k_2.ply"

    mesh = ensure_mesh_from_ply(ply_path)

    render_orbit_video_win(
        mesh,
        out_path="orbit_result.mp4",
        width=1080,
        height=1080,
        seconds=10,            
        fps=30,
        bg_color=(255, 255, 255),
        mesh_rgb=(0.78, 0.81, 0.66),
        zoom=0.72,
        total_yaw_deg=60.0,
        elev_deg=0.0,
        cycles=2.0,            
        yaw_offset_deg=-90.0,
        start_at_center=True,
        pre_rot_deg=(90, -90, 180),
        mesh_show_back_face=True,
        brightness=0.86,       
        use_lighting=True,
    )
