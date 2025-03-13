#!/usr/bin/env python3
import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index file")
    args = parser.parse_args()

    print(f"Resolved root_audio: {os.path.abspath(args.root_audio)}")
    print(f"Resolved root_frame: {os.path.abspath(args.root_frame)}")

    # Create validation set with one sample per instrument
    infos = []
    instruments = sorted(os.listdir(args.root_audio))
    print(f"Found {len(instruments)} instruments")

    for instrument in instruments:
        instrument_audio_path = os.path.join(args.root_audio, instrument)
        if not os.path.isdir(instrument_audio_path):
            print(f"Skipping {instrument}: not a directory in audio path")
            continue

        # Find all audio files for this instrument
        audio_files = find_recursive(instrument_audio_path, ext='.mp3')
        if not audio_files:
            print(f"No audio files found for instrument: {instrument}")
            continue

        # Shuffle audio files to try them in random order
        random.shuffle(audio_files)

        selected = False
        for audio_path in audio_files:
            # Convert audio path to frame path
            frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.mp3', '.mp4')

            # Check if frame directory exists
            if not os.path.isdir(frame_path):
                print(f"Frame directory not found: {frame_path}")
                continue

            # Find all jpg frames in the directory
            frame_files = glob.glob(os.path.join(frame_path, '*.jpg'))

            # Check if enough frames
            if len(frame_files) > args.fps * 20:
                infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
                print(f"Selected for {instrument}: {os.path.basename(audio_path)}")
                selected = True
                break
            else:
                print(f"Skipping: {frame_path} (Not enough frames: {len(frame_files)})")

        if not selected:
            print(f"Warning: No suitable samples found for instrument: {instrument}")

    # Save validation CSV
    val_filename = os.path.join(args.path_output, 'val_vis.csv')
    with open(val_filename, 'w') as f:
        for item in infos:
            f.write(item + '\n')
    print(f"{len(infos)} validation items saved to {val_filename}.")
    print("Done!")