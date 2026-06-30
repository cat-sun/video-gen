import cv2

video_path = "asset/Das_validation/animating_mesh_to_videos/videos/1.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps if fps > 0 else None

cap.release()

print(f"🎞️  帧数: {frame_count}, FPS: {fps}, 时长: {duration:.2f} 秒")
