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
    parser.add_argument('--root_audio', default='./data/eval_audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/eval_frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    print(f"Resolved root_audio: {os.path.abspath(args.root_audio)}")
    print(f"Resolved root_frame: {os.path.abspath(args.root_frame)}")


    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.mp3')
    # for audio_path in audio_files:
    #     print(f"Processing audio file: {audio_path}")

    #     frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.mp3', '.mp4')
    #     print(f"Looking for frames in: {frame_path}")

    #     frame_files = glob.glob(frame_path + '/*.jpg')
    #     print(f"Found {len(frame_files)} frames for {frame_path}")

    #     if len(frame_files) > args.fps * 20:
    #         infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    #     else:
    #         print(f"Skipping: {frame_path} (Not enough frames: {len(frame_files)})")
    
    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20:
            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(0)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
