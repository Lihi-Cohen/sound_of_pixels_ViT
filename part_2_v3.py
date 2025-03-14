import os
import subprocess

# Define input and output directories
base_dirs = ["downloaded_videos/MUSIC21_solo", "downloaded_videos/MUSIC_duet", "downloaded_videos/MUSIC_solo"]
output_dir = "data"

audio_output_dir = os.path.join(output_dir, "audio")
frames_output_dir = os.path.join(output_dir, "frames")

# Ensure output directories exist
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(frames_output_dir, exist_ok=True)

def extract_audio_and_frames(instrument, video_file, video_path, audio_path, frames_path):
    """
    Extract audio waveforms at 11025Hz and frames at 8fps (resized to 224x224) from a video.
    """
    # Create output paths for this video
    video_id = video_file  # Keep the file name with the extension for folder naming
    video_frames_path = os.path.join(frames_path, instrument, video_id)
    audio_output_file = os.path.join(audio_path, instrument, f"{os.path.splitext(video_file)[0]}.mp3")  # Save audio as MP3

    print(f"Processing instrument: {instrument}, video: {video_file}")
    
    # Ensure directories exist
    os.makedirs(video_frames_path, exist_ok=True)
    os.makedirs(os.path.join(audio_path, instrument), exist_ok=True)
    
    # Extract frames at 8fps and resize to 224x224
    frame_command = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps=8,scale=224:224",  # Add scale filter for resizing frames
        os.path.join(video_frames_path, "%06d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ]
    
    # Extract audio at 11025Hz
    audio_command = [
        "ffmpeg", "-i", video_path,
        "-ar", "11025",
        "-ac", "1",
        "-vn",
        audio_output_file,
        "-hide_banner", "-loglevel", "error"
    ]
    
    try:
        print(f"Extracting frames for {video_file}...")
        subprocess.run(frame_command, check=True)
        
        print(f"Extracting audio for {video_file}...")
        subprocess.run(audio_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_file}: {e}")

# Walk through the downloaded videos
for base_dir in base_dirs:
    for instrument in os.listdir(base_dir):
        instrument_path = os.path.join(base_dir, instrument)
        if not os.path.isdir(instrument_path):
            continue
        video_dir = os.path.join(instrument_path, "videos")
        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, video_file)
                if not video_file.endswith(('.mp4', '.mkv', '.avi')):  # Ensure it's a video file
                    continue
                
                # Process videos to extract frames and audio
                extract_audio_and_frames(
                    instrument,
                    video_file,
                    video_path,
                    audio_output_dir,
                    frames_output_dir
                )
print("Preprocessing completed.")
