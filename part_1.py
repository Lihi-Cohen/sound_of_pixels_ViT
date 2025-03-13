import os
import json
import subprocess

print("Current Working Directory:", os.getcwd())

# Define input JSON files


json_files = [
    "MUSIC_dataset/MUSIC21_solo_videos.json",
    "MUSIC_dataset/MUSIC_duet_videos.json",
    "MUSIC_dataset/MUSIC_solo_videos.json"
]


# Define output directory
base_output_dir = "downloaded_videos"
# Ensure the base output directory exists
os.makedirs(base_output_dir, exist_ok=True)

def download_videos(json_file, folder_name):
    """
    Download videos and audio based on video IDs from a JSON file,
    using video IDs as filenames instead of video titles.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Create a base directory for the current JSON file
    json_output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(json_output_dir, exist_ok=True)
    
    for category, video_ids in data.get("videos", {}).items():
        # Create a directory for each category inside the current JSON folder
        category_dir = os.path.join(json_output_dir, category)
        video_dir = os.path.join(category_dir, "videos")
        audio_dir = os.path.join(category_dir, "audio")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        print(f"Downloading videos and audio for category: {category} in {folder_name}")
        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"
            # Use video_id as filename instead of video title
            video_output_path = os.path.join(video_dir, f"{video_id}.mp4")
            audio_output_path = os.path.join(audio_dir, f"{video_id}.mp3")
            
            # Download video with best quality
            video_command = [
                "yt-dlp",
                "-f", "bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", video_output_path,
                url
            ]
            
            # Download audio only
            audio_command = [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "mp3",
                "-o", audio_output_path,
                url
            ]
            
            try:
                print(f"Downloading video {video_id}: {url} to {video_dir}")
                subprocess.run(video_command, check=True)
                
                print(f"Downloading audio {video_id}: {url} to {audio_dir}")
                subprocess.run(audio_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {video_id} - {url}: {e}")

# Map JSON file names to folder names

json_folder_map = {
    "MUSIC_dataset/MUSIC21_solo_videos.json": "MUSIC21_solo",
    "MUSIC_dataset/MUSIC_duet_videos.json": "MUSIC_duet",
    "MUSIC_dataset/MUSIC_solo_videos.json": "MUSIC_solo"
}

# Download videos and audio for each JSON file
for json_file, folder_name in json_folder_map.items():
    if os.path.exists(json_file):
        download_videos(json_file, folder_name)
    else:
        print(f"JSON file not found: {json_file}")
