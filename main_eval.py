# System libs

import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
# from scipy.misc import imsave
from mir_eval.separation import bss_eval_sources
import imageio

# Our libs

from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer



##ADDED:

from sklearn.decomposition import PCA
from PIL import Image
from viz_2 import visualize_sound_clustering, visualize_activations
import numpy as np


# Network wrapper, defines forward pass

class NetWrapper(torch.nn.Module):

    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_synthesizer = nets



    def forward(self, batch_data, args):
        mags = batch_data['mag_mix']
        frames = batch_data['frames']
        mags = mags + 1e-10

        B = mags.size(0)
        T = mags.size(3)

        # warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mags = F.grid_sample(mags, grid_warp)

        # LOG magnitude
        log_mags = torch.log(mags).detach()

        # 1. forward net_sound -> BxCxHSxWS
        feat_sound = self.net_sound(log_mags)
        feat_sound = activate(feat_sound, args.sound_activation)

        # 2. forward net_frame -> BxCxHIxHS
        # required pool = False argument

        if args.img_pool:
            print('Evaluation requires pool argument == False !!')

        frames_tensor = frames[0]  
        feat_frames = self.net_frame.forward(frames_tensor, False)  # BxCxTxHIxHS (T = num_frames)
        feat_frames = activate(feat_frames, args.img_activation)

        # averging over temporal dimension

        feat_frames = feat_frames.mean(dim=2)  # New shape: (B, C, HI, WI)

        # 3. sound synthesizer
        pred_masks = self.net_synthesizer.forward_pixelwise(feat_frames, feat_sound)

        #return pred_masks
        return pred_masks, feat_frames, feat_sound

# Calculate metrics

def calc_metrics(batch_data, outputs, args):

    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5

        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)

            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)

            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())


    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]

    for n in range(N):

        if args.log_freq:

            grid_unwarp = torch.from_numpy(

                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)

            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)

            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)

        else:

            pred_masks_linear[n] = pred_masks_[n]

            gt_masks_linear[n] = gt_masks_[n]



    # convert into numpy

    mag_mix = mag_mix.numpy()

    mag_mix_ = mag_mix_.detach().cpu().numpy()

    phase_mix = phase_mix.numpy()

    weight_ = weight_.detach().cpu().numpy()

    for n in range(N):

        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()

        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()

        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()



        # threshold if binary mask

        if args.binary_mask:

            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)

            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)



    # loop over each sample

    for j in range(B):

        row_elements = []



        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imageio.imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        imageio.imsave(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]



        # save each component

        preds_wav = [None for n in range(N)]

        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imageio.imsave(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            imageio.imsave(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            # output spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imageio.imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imageio.imsave(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])


            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            # output video
            frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(args.num_frames)]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(args.vis, prefix, 'video{}.mp4'.format(n+1))
            save_video(path_video, frames_tensor, fps=args.frameRate/args.stride_frames)



            # combine gt video and audio

            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))

            combine_video_audio(

                path_video,

                os.path.join(args.vis, filename_gtwav),
                os.path.join(args.vis, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)



def find_best_channel(visual_features):
    """
    Find channel that best captures multiple sound sources
    """
    channels_scores = []
    for c in range(visual_features.shape[0]):
        channel_map = visual_features[c]
        # Normalize
        channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-8)
        # Count distinct activation regions (threshold > 0.5)
        active_regions = np.sum(channel_map > 0.5)
        # Calculate variance of activations
        spatial_variance = np.var(channel_map)
        # Combine metrics
        score = active_regions * spatial_variance
        channels_scores.append(score)
    return np.argmax(channels_scores)



def visualize_sound_clustering2(pixel_spectrograms, frame, output_path):
    """
    Visualize sound source clustering with balanced emphasis on source locations.
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from sklearn.cluster import KMeans

    
    # Normalize input
    pixel_spectrograms = np.asarray(pixel_spectrograms)
    pixel_spectrograms = (pixel_spectrograms - np.min(pixel_spectrograms)) / \
                        (np.max(pixel_spectrograms) - np.min(pixel_spectrograms) + 1e-8)
    
    # Apply PCA
    pca_obj = PCA(n_components=3)
    sound_features = pca_obj.fit_transform(pixel_spectrograms)
    
    # Normalize features
    sound_features = (sound_features - np.min(sound_features)) / \
                    (np.max(sound_features) - np.min(sound_features) + 1e-8)
    
    # Gentler thresholding
    threshold = 0.4  # Reduced threshold
    mask = sound_features < threshold
    sound_features[mask] *= 0.3  # Less aggressive reduction of weak signals
    
    # Milder non-linear scaling
    sound_features = np.power(sound_features, 1.5)  # Reduced from 2 to 1.5
    
    # Normalize again after scaling
    sound_features = (sound_features - np.min(sound_features)) / \
                    (np.max(sound_features) - np.min(sound_features) + 1e-8)
    
    # Reshape to image grid
    overlay = sound_features.reshape(14, 14, 3)
    
    # Process frame
    frame_display = np.transpose(frame, (1, 2, 0))
    frame_display = (frame_display - np.min(frame_display)) / \
                   (np.max(frame_display) - np.min(frame_display) + 1e-8)
    
    # Resize overlay
    zoom_factors = (frame_display.shape[0]/overlay.shape[0], 
                   frame_display.shape[1]/overlay.shape[1], 
                   1)
    overlay_resized = zoom(overlay, zoom_factors, order=1)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Original frame
    plt.subplot(1, 3, 1)
    plt.imshow(frame_display)
    plt.title('Original Frame')
    plt.axis('off')
    
    # Sound source overlay
    plt.subplot(1, 3, 2)
    plt.imshow(overlay_resized)
    plt.title('Sound Source Clusters')
    plt.axis('off')
    
    # Blended visualization with balanced opacity
    plt.subplot(1, 3, 3)
    alpha = 0.65  # Reduced opacity
    blended = (1 - alpha) * frame_display + alpha * overlay_resized
    plt.imshow(np.clip(blended, 0, 1))
    plt.title('Blended Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()   

    torch.cuda.synchronize()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            with autocast():  # Enable mixed precision
                pred_masks, feat_frames, feat_sound = netWrapper.forward(batch_data, args)      

        # Process each item in batch
        B, HI, WI, HS, WS = pred_masks.shape
        for b in range(B):
            # 1. Sound Source Clustering Visualization
            pixel_spectrograms_flat = pred_masks[b].reshape(HI * WI, HS * WS).detach().cpu().numpy()
            frame = batch_data['frames'][0][b, :, 1].cpu().numpy()

            clustering_path = os.path.join(args.vis, f'sound_clustering_batch{i}_sample{b}.png')
            try:
                visualize_sound_clustering2(
                    pixel_spectrograms_flat,
                    frame,
                    clustering_path
                )

                print(f"Successfully created clustering visualization for batch {i}, sample {b}")
            except Exception as e:
                print(f"Error processing clustering visualization batch {i}, sample {b}: {str(e)}")
                continue


            # 2. Network Activation Visualization
            visual_features = feat_frames[b].detach().cpu().numpy()
            audio_features = feat_sound[b].detach().cpu().numpy()


            # Find the best channel using new method
            best_channel = find_best_channel(visual_features)

            # Select best channel features

            best_visual_features = visual_features[best_channel:best_channel+1]
            best_audio_features = audio_features[best_channel:best_channel+1]

            activation_path = os.path.join(args.vis, f'activations_batch{i}_sample{b}.png')
            try:
                visualize_activations(
                    frame,
                    best_visual_features,
                    best_audio_features,
                    activation_path,
                    channel_idx=best_channel

                )

                print(f"Successfully created activation visualization for batch {i}, sample {b}")
            except Exception as e:
                print(f"Error processing activation visualization batch {i}, sample {b}: {str(e)}")
                continue

        # Explicitly delete the batch data
        del batch_data
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"Memory cleared after batch {i}")

    print('Evaluation visualization completed!')

from torch.cuda.amp import autocast, GradScaler


def main(args):
    # Network Builders
    builder = ModelBuilder()

    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)

    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)

    net_synthesizer = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)
    nets = (net_sound, net_frame, net_synthesizer)

    # Dataset and Loader
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)



    # Wrap networks
    netWrapper = NetWrapper(nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)



    # History of peroformance

    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    scaler = GradScaler()

    # Eval mode
    evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return



if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")


    ###ADDED
    use_img_pool = args.img_pool.lower() not in ["none", "false", ""]
    print(f"Using img_pool: {use_img_pool}")  # Debugging check
    ###

    # experiment name
    args.id += '{}'.format(args.vis_train_mode)
   #args.id += ...
    print('Model ID: {}'.format(args.id))


    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt,'visualization/')


    # dir for output visuals
    makedirs(args.ckpt, remove=False)
    makedirs(args.vis, remove=False)


    args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
    args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
    args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
