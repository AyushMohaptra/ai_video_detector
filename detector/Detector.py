import os

# Silence TensorFlow / absl / mediapipe logs as much as possible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # TF: only errors
os.environ['GLOG_minloglevel'] = '3'       # glog: errors only

import logging
logging.getLogger().setLevel(logging.ERROR)

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_absl_handler()

import cv2
import mediapipe as mp
import numpy as np
import argparse
import contextlib

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# Mouth center landmarks
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

# A landmark near the nose to approximate head position
NOSE_IDX = 1


@contextlib.contextmanager
def suppress_output():
    """Context manager that redirects stdout and stderr to /dev/null.

    This silences native C/C++ logs (EGL/OpenGL/TensorFlow/MediaPipe) that
    bypass Python's logging system and are printed directly to the
    process' file descriptors.
    """
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
    except Exception:
        # Fallback to using os.devnull path via open() if os.open fails
        devnull = None

    if devnull is None:
        # Best-effort: yield without redirection
        yield
        return

    # Save original fds
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore
        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(devnull)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)


def eye_aspect_ratio(landmarks, eye_idx, image_w, image_h):
    pts = []
    for i in eye_idx:
        lm = landmarks[i]
        pts.append(np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32))

    p0, p1, p2, p3, p4, p5 = pts

    d1 = np.linalg.norm(p1 - p5)
    d2 = np.linalg.norm(p2 - p4)
    d3 = np.linalg.norm(p0 - p3)

    if d3 == 0:
        return 0.0

    ear = (d1 + d2) / (2.0 * d3)
    return ear


def analyze_face_dynamics(video_path, target_fps=10, suppress_native_logs=True):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0:
        video_fps = target_fps
    frame_interval = max(int(round(video_fps / target_fps)), 1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0001

    blink_count = 0
    prev_left_open = True
    prev_right_open = True

    left_ear_list = []
    right_ear_list = []

    mouth_open_ratio_list = []
    head_pos_list = []

    frame_idx = 0

    # Many native libraries (MediaPipe, OpenGL, TensorFlow C++) print info
    # messages directly to the process' stdout/stderr before Python's
    # logging is initialized. Optionally wrap the FaceMesh usage with a
    # suppression context to keep the CLI output clean.
    ctx = suppress_output() if suppress_native_logs else contextlib.nullcontext()
    with ctx:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
                    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)
                    left_ear_list.append(left_ear)
                    right_ear_list.append(right_ear)

                    EAR_THRESH = 0.21
                    left_open = left_ear > EAR_THRESH
                    right_open = right_ear > EAR_THRESH

                    if prev_left_open and not left_open:
                        blink_count += 0.5
                    if prev_right_open and not right_open:
                        blink_count += 0.5

                    prev_left_open = left_open
                    prev_right_open = right_open

                    upper_lip = face_landmarks[UPPER_LIP_IDX]
                    lower_lip = face_landmarks[LOWER_LIP_IDX]
                    nose = face_landmarks[NOSE_IDX]

                    upper = np.array([upper_lip.x * w, upper_lip.y * h])
                    lower = np.array([lower_lip.x * w, lower_lip.y * h])
                    nose_p = np.array([nose.x * w, nose.y * h])

                    mouth_dist = np.linalg.norm(upper - lower)
                    face_scale = np.linalg.norm(nose_p - np.array([w / 2.0, h / 2.0])) + 1e-6
                    mouth_open_ratio = mouth_dist / face_scale
                    mouth_open_ratio_list.append(mouth_open_ratio)

                    head_pos_list.append(nose_p)

                frame_idx += 1

    cap.release()

    if not left_ear_list:
        raise RuntimeError("No face detected in the video for analysis.")

    total_blinks = int(round(blink_count))
    blink_rate_per_min = total_blinks / duration_min

    mouth_var = float(np.var(mouth_open_ratio_list))

    head_pos_arr = np.array(head_pos_list)
    if len(head_pos_arr) > 1:
        diffs = np.diff(head_pos_arr, axis=0)
        step_magnitudes = np.linalg.norm(diffs, axis=1)
        head_motion_std = float(np.std(step_magnitudes))
    else:
        head_motion_std = 0.0

    if blink_rate_per_min < 2:
        blink_suspicion = 1.0
    elif blink_rate_per_min <=30:
        blink_suspicion = 0.1
    else:
        blink_suspicion = 0.4

    if mouth_var < 1e-5:
        mouth_suspicion = 0.9
    elif mouth_var < 5e-5:
        mouth_suspicion = 0.7
    elif mouth_var < 1e-4:
        mouth_suspicion = 0.4
    else:
        mouth_suspicion = 0.1

    if head_motion_std < 0.2:
        head_suspicion = 0.6
    elif head_motion_std < 0.5:
        head_suspicion = 0.4
    else:
        head_suspicion = 0.2

    ai_suspicion = (
        0.5 * blink_suspicion
        + 0.3 * mouth_suspicion
        + 0.2 * head_suspicion
    )
    ai_suspicion = float(min(max(ai_suspicion, 0.0), 1.0))

    return {
        "blink_rate_per_min": float(blink_rate_per_min),
        "total_blinks": int(total_blinks),
        "mouth_variance": mouth_var,
        "head_motion_std": head_motion_std,
        "blink_suspicion": blink_suspicion,
        "mouth_suspicion": mouth_suspicion,
        "head_suspicion": head_suspicion,
        "face_ai_suspicion": ai_suspicion,
    }


def analyze_motion_dynamics(video_path, target_fps=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0:
        video_fps = target_fps
    frame_interval = max(int(round(video_fps / target_fps)), 1)

    motion_mags = []

    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            motion_mags.append(mean_mag)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not motion_mags:
        raise RuntimeError("No motion data extracted from the video.")

    motion_mags = np.array(motion_mags)

    mean_motion = float(np.mean(motion_mags))
    std_motion = float(np.std(motion_mags))

    diffs = np.diff(motion_mags)
    abs_diffs = np.abs(diffs)
    mean_change = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0
    std_change = float(np.std(abs_diffs)) if len(abs_diffs) > 0 else 0.0

    very_low_thresh = 0.2
    low_motion_ratio = float(np.mean(motion_mags < very_low_thresh))

    high_thresh = np.percentile(motion_mags, 90)
    high_motion_ratio = float(np.mean(motion_mags > high_thresh))

    if len(abs_diffs) > 0:
        spike_thresh = np.percentile(abs_diffs, 90)
        spike_ratio = float(np.mean(abs_diffs > spike_thresh)) if spike_thresh > 0 else 0.0
    else:
        spike_ratio = 0.0

    if low_motion_ratio > 0.7:
        calm_suspicion = 1.0
    elif low_motion_ratio > 0.4:
        calm_suspicion = 0.6
    elif low_motion_ratio > 0.2:
        calm_suspicion = 0.3
    else:
        calm_suspicion = 0.1

    if high_motion_ratio > 0.3 and spike_ratio > 0.3:
        fast_suspicion = 1.0
    elif high_motion_ratio > 0.2 and spike_ratio > 0.2:
        fast_suspicion = 0.7
    elif high_motion_ratio > 0.1 and spike_ratio > 0.1:
        fast_suspicion = 0.4
    else:
        fast_suspicion = 0.1

    if std_motion < 0.05:
        smooth_suspicion = 0.8
    elif std_motion < 0.1:
        smooth_suspicion = 0.5
    else:
        smooth_suspicion = 0.2

    ai_motion_suspicion = (
        0.4 * calm_suspicion
        + 0.4 * fast_suspicion
        + 0.2 * smooth_suspicion
    )
    ai_motion_suspicion = float(min(max(ai_motion_suspicion, 0.0), 1.0))

    return {
        "mean_motion": mean_motion,
        "std_motion": std_motion,
        "mean_change": mean_change,
        "std_change": std_change,
        "low_motion_ratio": low_motion_ratio,
        "high_motion_ratio": high_motion_ratio,
        "spike_ratio": spike_ratio,
        "calm_suspicion": calm_suspicion,
        "fast_suspicion": fast_suspicion,
        # legacy name used elsewhere in the code / prints
        "ragdoll_suspicion": fast_suspicion,
        "smooth_suspicion": smooth_suspicion,
        "ai_motion_suspicion": ai_motion_suspicion,
    }


def analyze_temporal_inconsistency(video_path, target_fps=10, resize=(320, 180)):
    """Detect temporal inconsistencies using dense optical flow.

    Returns detailed metrics and an `ai_temporal_suspicion` score in [0,1].

    The detector looks for:
    - jitter (fast changes in mean flow between frames)
    - mismatched directions (low global direction consistency)
    - spatial incoherence (flow vectors within a frame disagree)
    - unnatural smooth stretches (long runs of near-zero flow)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0:
        video_fps = target_fps
    frame_interval = max(int(round(video_fps / target_fps)), 1)

    mean_mags = []
    mean_vecs = []
    dir_consistency_list = []
    spatial_incoherence_list = []
    zero_motion_mask = []

    prev_gray = None
    frame_idx = 0

    w, h = resize

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_small = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )

            fx = flow[..., 0]
            fy = flow[..., 1]
            mag = np.sqrt(fx * fx + fy * fy)
            ang = np.arctan2(fy, fx)  # radians, -pi..pi

            # Mean magnitude and mean flow vector for the frame
            mean_mag = float(np.mean(mag))
            mean_mags.append(mean_mag)
            mean_vec = np.array([float(np.mean(fx)), float(np.mean(fy))], dtype=np.float32)
            mean_vecs.append(mean_vec)

            # Directional consistency: resultant vector length of unit vectors
            eps = 1e-6
            ux = fx / (mag + eps)
            uy = fy / (mag + eps)
            # Ignore zero vectors by masking
            mask = mag > (np.percentile(mag, 10) * 0.1 + 1e-6)
            if np.any(mask):
                sum_x = float(np.sum(ux[mask]))
                sum_y = float(np.sum(uy[mask]))
                R = np.sqrt(sum_x * sum_x + sum_y * sum_y) / float(np.sum(mask))
            else:
                R = 0.0
            dir_consistency_list.append(float(R))

            # Spatial incoherence: per-patch angular std (coarse)
            ph = 8
            pw = 8
            ang_std_patches = []
            H, W = ang.shape
            for y in range(0, H, ph):
                for x in range(0, W, pw):
                    patch = ang[y : min(y + ph, H), x : min(x + pw, W)]
                    if patch.size == 0:
                        continue
                    # circular std approximation: use unit vectors
                    cx = np.cos(patch)
                    cy = np.sin(patch)
                    r = np.sqrt((np.mean(cx)) ** 2 + (np.mean(cy)) ** 2)
                    circ_std = float(np.sqrt(max(0.0, 1.0 - r)))
                    ang_std_patches.append(circ_std)
            spatial_incoherence_list.append(float(np.mean(ang_std_patches)) if ang_std_patches else 0.0)

            # Zero/very-low-motion fraction
            zero_motion_mask.append(float(np.mean(mag < 1e-3)))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not mean_mags:
        raise RuntimeError("No temporal flow data extracted from the video.")

    mean_mags = np.array(mean_mags)
    mean_vecs = np.array(mean_vecs)
    dir_consistency_list = np.array(dir_consistency_list)
    spatial_incoherence_list = np.array(spatial_incoherence_list)
    zero_motion_mask = np.array(zero_motion_mask)

    # Temporal jitter: variability of mean flow vector changes
    vec_diffs = np.linalg.norm(np.diff(mean_vecs, axis=0), axis=1)
    jitter_index = float(np.std(vec_diffs) / (np.mean(mean_mags) + 1e-6))

    # Mismatched direction: low average directional consistency
    mean_dir_consistency = float(np.mean(dir_consistency_list))

    # Spatial incoherence: average patch angular std
    mean_spatial_incoherence = float(np.mean(spatial_incoherence_list))

    # Smooth stretches: fraction of frames with very low mean motion
    low_motion_frac = float(np.mean(mean_mags < 0.01))
    # Long runs of low motion
    low_runs = 0
    run_len = 0
    for v in (mean_mags < 0.01):
        if v:
            run_len += 1
        else:
            if run_len >= 5:
                low_runs += 1
            run_len = 0
    if run_len >= 5:
        low_runs += 1

    # Normalize metrics to 0..1-ish scales
    jitter_score = min(max(jitter_index / 0.5, 0.0), 1.0)
    dir_mismatch_score = min(max(1.0 - mean_dir_consistency, 0.0), 1.0)
    spatial_score = min(max(mean_spatial_incoherence / 0.8, 0.0), 1.0)
    smooth_stretch_score = min(max((low_motion_frac * 0.6 + low_runs * 0.4) , 0.0), 1.0)

    # Combine into a single suspicion score (weights are heuristic)
    ai_temporal_suspicion = (
        0.35 * jitter_score
        + 0.25 * dir_mismatch_score
        + 0.2 * spatial_score
        + 0.2 * smooth_stretch_score
    )
    ai_temporal_suspicion = float(min(max(ai_temporal_suspicion, 0.0), 1.0))

    return {
        "mean_mags_mean": float(np.mean(mean_mags)),
        "mean_mags_std": float(np.std(mean_mags)),
        "jitter_index": jitter_index,
        "dir_consistency_mean": mean_dir_consistency,
        "spatial_incoherence_mean": mean_spatial_incoherence,
        "low_motion_frac": low_motion_frac,
        "low_motion_runs": low_runs,
        "jitter_score": jitter_score,
        "dir_mismatch_score": dir_mismatch_score,
        "spatial_score": spatial_score,
        "smooth_stretch_score": smooth_stretch_score,
        "ai_temporal_suspicion": ai_temporal_suspicion,
    }

def analyze_body_dynamics(video_path, suppress_native_logs=True):
    # Try face dynamics, but don't fail if no face is found
    face_res = None
    try:
        face_res = analyze_face_dynamics(video_path, suppress_native_logs=suppress_native_logs)
        face_score = face_res["face_ai_suspicion"]
        face_available = True
    except RuntimeError as e:
        # Specifically handle the "no face" case
        if "No face detected" in str(e):
            face_available = False
            face_score = 0.0
        else:
            # Any other error should still bubble up
            raise

    motion_res = analyze_motion_dynamics(video_path)
    motion_score = motion_res["ai_motion_suspicion"]
    # Temporal inconsistency analysis (optical-flow focused)
    try:
        temporal_res = analyze_temporal_inconsistency(video_path)
        temporal_score = temporal_res["ai_temporal_suspicion"]
    except Exception:
        temporal_res = None
        temporal_score = 0.0

    # If no face, rely fully on motion
    if face_res is None or not face_available:
        w_face = 0.0
        w_motion = 1.0
    else:
        w_face = 0.3
        w_motion = 0.7

    final_score = w_face * face_score + w_motion * motion_score

    # Combined v2 score including temporal inconsistency
    # We keep the original final_score for backward compatibility,
    # and compute `final_body_suspicion_v2` that includes temporal cues.
    if face_res is None or not face_available:
        w_motion_v2 = 0.6
        w_temporal_v2 = 0.4
        w_face_v2 = 0.0
    else:
        w_face_v2 = 0.25
        w_motion_v2 = 0.5
        w_temporal_v2 = 0.25

    final_score_v2 = (
        w_face_v2 * face_score + w_motion_v2 * motion_score + w_temporal_v2 * temporal_score
    )

    return {
        "face": face_res,  # can be None if no face
        "motion": motion_res,
        "temporal": temporal_res,
        "final_body_suspicion": float(final_score),
        "final_body_suspicion_v2": float(final_score_v2),
        "face_available": bool(face_res is not None and face_available),
    }
def main():
    parser = argparse.ArgumentParser(description="Body dynamics-based AI suspicion (face + motion)")
    parser.add_argument("video_path", type=str, help="Path to video file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quiet', dest='quiet', action='store_true', help='Silence native logs (MediaPipe/TensorFlow)')
    group.add_argument('--no-quiet', dest='quiet', action='store_false', help='Do not silence native logs')
    parser.set_defaults(quiet=True)
    args = parser.parse_args()

    print(f"Analyzing body dynamics for: {args.video_path}")
    try:
        res = analyze_body_dynamics(args.video_path, suppress_native_logs=args.quiet)
    except Exception as e:
        print("Error during analysis:", e)
        return

    face = res["face"]
    motion = res["motion"]
    final_score = res["final_body_suspicion"]
    face_available = res.get("face_available", face is not None)
    
    if face_available and face is not None:
        print("\n--- Face Dynamics ---")
        print(f"Blink rate: {face['blink_rate_per_min']:.2f} blinks/min")
        print(f"Total blinks: {face['total_blinks']}")
        print(f"Mouth variance: {face['mouth_variance']:.6f}")
        print(f"Head motion std: {face['head_motion_std']:.4f}")
        print(f"Face AI suspicion: {face['face_ai_suspicion']:.2f}")
    else:
        print("\n--- Face Dynamics ---")
        print("No face detected in this video. Using motion-only analysis.")

    print("\n--- Motion Dynamics ---")
    print(f"Mean motion: {motion['mean_motion']:.4f}")
    print(f"Std motion: {motion['std_motion']:.4f}")
    print(f"Low-motion ratio: {motion['low_motion_ratio']:.2f}")
    print(f"High-motion ratio: {motion['high_motion_ratio']:.2f}")
    print(f"Spike ratio: {motion['spike_ratio']:.2f}")
    print(f"Calm suspicion: {motion['calm_suspicion']:.2f}")
    print(f"Smooth suspicion: {motion['smooth_suspicion']:.2f}")
    print(f"Ragdoll/fast-move suspicion: {motion['ragdoll_suspicion']:.2f}")
    print(f"Motion AI suspicion: {motion['ai_motion_suspicion']:.2f}")

    temporal = res.get("temporal")
    if temporal is not None:
        print("\n--- Temporal Inconsistency (Optical Flow) ---")
        print(f"Mean mag (mean): {temporal['mean_mags_mean']:.4f}")
        print(f"Mean mag (std): {temporal['mean_mags_std']:.4f}")
        print(f"Jitter index: {temporal['jitter_index']:.4f}")
        print(f"Direction consistency (mean R): {temporal['dir_consistency_mean']:.4f}")
        print(f"Spatial incoherence (mean): {temporal['spatial_incoherence_mean']:.4f}")
        print(f"Low-motion fraction: {temporal['low_motion_frac']:.2f}")
        print(f"Temporal AI suspicion: {temporal['ai_temporal_suspicion']:.2f}")


    print("\n=== Combined Body Dynamics Score ===")
    print(f"➡ Final body-based AI suspicion: {final_score:.2f}")

    if final_score >= 0.4:
        print("Conclusion: BODY DYNAMICS LOOK CLEARLY SUSPICIOUS (LIKELY AI).")
    elif final_score >= 0.36:
        print("Conclusion: BODY DYNAMICS ARE AMBIGUOUS / MIXED.")
    else:
        print("Conclusion: BODY DYNAMICS LOOK NATURAL (LIKELY REAL).")


if __name__ == "__main__":
    main()
