import os
import random
from .base import BaseDataset

class MUSICNoMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICNoMixDataset, self).__init__(list_sample, opt, **kwargs)
        self.fps = opt.frameRate

    def __getitem__(self, index):
        # Info to retrieve individual audio and frames
        infos = self.list_sample[index]

        # Load frame and audio info
        path_audio, path_frame, count_frames = infos

        # Select center frame (either randomly for training or middle frame for validation/test)
        idx_margin = max(int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        if self.split == 'train':
            center_frame = random.randint(idx_margin + 1, int(count_frames) - idx_margin)
        else:
            center_frame = int(count_frames) // 2

        # Absolute frame paths
        path_frames = []
        for i in range(self.num_frames):
            idx_offset = (i - self.num_frames // 2) * self.stride_frames
            path_frames.append(
                os.path.join(path_frame, '{:06d}.jpg'.format(center_frame + idx_offset)))

        # Load frames and audio
        try:
            frames = self._load_frames(path_frames)
            center_time = (center_frame - 0.5) / self.fps  # Optional jitter or fixed center
            audio = self._load_audio(path_audio, center_time)
            mag, _ = self._stft(audio)
            mag = mag.unsqueeze(0)

        except Exception as e:
            print(f"Failed loading frame/audio: {e}")
            # Handle error, create dummy data
            frames, audio = self.dummy_frame_audio_data()

        # Return dictionary with frame and audio data
        ret_dict = {'frames': frames, 'audio': audio, 'mags':mag}
        
        if self.split != 'train':
            ret_dict['infos'] = infos  # Metadata if not training

        return ret_dict
