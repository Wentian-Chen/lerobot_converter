import tarfile
import json
import io
import os
import av
from PIL import Image

def test_sthv2_info(data_dir, num_videos=5):
    tar_path = os.path.join(data_dir, "20bn-something-something-v2.tar.gz")
    labels_zip_path = os.path.join(data_dir, "labels.zip")
    
    print(f"Checking dataset at: {data_dir}")
    
    if not os.path.exists(tar_path):
        print(f"Error: {tar_path} not found.")
        return

    # Extract labels to get video info (optional but good for context)
    # For a quick test, we can just iterate the tar file directly
    
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            # Sthv2 videos are typically stored in subdirectories like 20bn-something-something-v2/12345.webm
            if member.isfile() and (member.name.endswith(".webm") or member.name.endswith(".mp4")):
                video_id = os.path.splitext(os.path.basename(member.name))[0]
                print(f"\n--- Member: {member.name} ---")
                print(f"  Video ID: {video_id}")
                
                # Read video data into memory
                f = tar.extractfile(member)
                video_bytes = f.read()
                
                # Use PyAV to probe video information
                try:
                    container = av.open(io.BytesIO(video_bytes))
                    stream = container.streams.video[0]
                    
                    print(f"  Resolution: {stream.width}x{stream.height}")
                    print(f"  Long Name: {stream.codec_context.codec.long_name}")
                    print(f"  Average FPS: {float(stream.average_rate)}")
                    print(f"  Frames: {stream.frames}")
                    print(f"  Duration: {float(container.duration / 1000000)} seconds")
                    
                    # Check first frame actual size
                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        print(f"  First Frame Actual Size: {img.size}")
                        break
                    
                    container.close()
                except Exception as e:
                    print(f"  Error processing video: {e}")
                
                count += 1
                if count >= num_videos:
                    break

if __name__ == "__main__":
    # Adjust path if needed
    base_dir = r"To/your/datasets/sthv2"
    test_sthv2_info(base_dir)
