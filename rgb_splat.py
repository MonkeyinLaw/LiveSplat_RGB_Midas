#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# feed_video_to_livesplat.py
#
# Stream a *recorded* file into LiveSplat as if it was a live RGB-D camera.
#
# • If the file is a RealSense  .bag  → colour + depth are read directly.
# • Otherwise any ordinary video (MP4, AVI …) is accepted; depth is inferred
#   on-the-fly with MiDaS / DPT and pushed to LiveSplat together with RGB.
#
# ---------------------------------------------------------------------------
#  ▸ Dependencies
#      pip install livesplat opencv-python torch torchvision timm
#      pip install pyrealsense2        # only when you use .bag playback
#
#  ▸ Usage examples
#      python feed_video_to_livesplat.py --input walk.mp4
#      python feed_video_to_livesplat.py --input capture.bag --bag
#
#  ▸ Keyboard interrupt (Ctrl-C) or closing the LiveSplat viewer stops the run
# ---------------------------------------------------------------------------

import argparse, time, sys, cv2, numpy as np

# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play a prerecorded RGB-D source into LiveSplat")
    p.add_argument("--input", required=True,
                   help="Path to MP4/AVI (RGB) or RealSense .bag")
    p.add_argument("--bag", action="store_true",
                   help="Force RealSense .bag playback (skip MiDaS)")
    p.add_argument("--midas", default="DPT_Large",
                   help="MiDaS model: DPT_Large | DPT_Hybrid | MiDaS_small")
    p.add_argument("--width",  type=int, default=640, help="Output width")
    p.add_argument("--height", type=int, default=480, help="Output height")
    p.add_argument("--fps",    type=int, default=30,  help="Target FPS")
    p.add_argument("--device-id", default="synthetic_rgbd",
                   help="Camera name shown in LiveSplat")
    return p.parse_args()

# ---------------------------------------------------------------------------

def make_intrinsics(w: int, h: int,
                    fx: float | None = None, fy: float | None = None,
                    cx: float | None = None, cy: float | None = None) -> np.ndarray:
    """Return a 3×3 pin-hole intrinsics matrix (float32)."""
    fx, fy = fx or w * 0.9, fy or h * 0.9     # rough guess
    cx, cy = cx or w / 2, cy or h / 2
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    return K

# ---------------------------------------------------------------------------
# ████  RealSense .bag  playback  ████
# ---------------------------------------------------------------------------

def play_bag(path: str, args) -> None:
    import pyrealsense2 as rs, livesplat
    cfg = rs.config()
    cfg.enable_device_from_file(path, repeat_playback=True)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)

    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    w = prof.get_stream(rs.stream.color).as_video_stream_profile().width()
    h = prof.get_stream(rs.stream.color).as_video_stream_profile().height()

    K = make_intrinsics(w, h)
    livesplat.register_camera_params(args.device_id, w, h, w, h,
                                     K, K, np.eye(4, dtype=np.float32), depth_scale)
    livesplat.start_viewer()

    while not livesplat.should_stop_all():
        try:
            frames = pipe.wait_for_frames()
        except RuntimeError:
            break                                   # end-of-file
        livesplat.ingest_rgb  (args.device_id,
                               np.asanyarray(frames.get_color_frame().get_data()))
        livesplat.ingest_depth(args.device_id,
                               np.asanyarray(frames.get_depth_frame().get_data()))

    pipe.stop()

# ---------------------------------------------------------------------------
# ████  Plain RGB video  →  depth via MiDaS  ████
# ---------------------------------------------------------------------------

def play_rgb_video(path: str, args) -> None:
    import torch, livesplat
    tf_repo = "intel-isl/MiDaS"

    # --- MiDaS initialisation ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas  = torch.hub.load(tf_repo, args.midas, trust_repo=True).to(device).eval()
    trans  = torch.hub.load(tf_repo, "transforms", trust_repo=True)
    tfm    = trans.dpt_transform if "DPT" in args.midas else trans.small_transform

    # --- LiveSplat camera registration --------------------------------------
    w, h = args.width, args.height
    K    = make_intrinsics(w, h)
    depth_scale_mm = 0.001                    # our output is uint16 mm
    livesplat.register_camera_params(
        args.device_id, w, h, w, h, K, K, np.eye(4, dtype=np.float32),
        depth_scale_mm
    )
    livesplat.start_viewer()

    # --- Video loop ----------------------------------------------------------
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"✘ cannot open {path}")

    target_dt = 1.0 / args.fps
    while not livesplat.should_stop_all():
        t0 = time.time()
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # depth inference
        inp = tfm(rgb).to(device)
        with torch.no_grad():
            pred = midas(inp).squeeze()
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="bicubic",
                align_corners=False
            )[0, 0]
        d = pred.cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)   # 0-1 normalise
        d_mm = (d * 1000).astype(np.uint16)              # → millimetres

        livesplat.ingest_rgb  (args.device_id, rgb)
        livesplat.ingest_depth(args.device_id, d_mm)

        # simple realtime pacing
        dt = time.time() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)

    cap.release()

# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.bag or args.input.lower().endswith(".bag"):
        play_bag(args.input, args)
    else:
        play_rgb_video(args.input, args)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
