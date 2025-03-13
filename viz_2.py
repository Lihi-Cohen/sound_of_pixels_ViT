

##Version 1 = more ephasis on sound source, less on background.
def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
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


  # ##Version two - less dark, less emphasis on sound source.
  # def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
  #   """
  #   Visualize sound source clustering by applying PCA to the spectrograms.
    
  #   Args:
  #       pixel_spectrograms: Array of shape (196, 65536) containing spectrograms
  #       frame: Video frame of shape (3, 224, 224)
  #       output_path: Path to save visualization
  #   """
  #   from sklearn.decomposition import PCA
  #   import numpy as np
  #   import matplotlib.pyplot as plt
  #   from scipy.ndimage import zoom
    
  #   # Ensure input is numpy array and normalized
  #   pixel_spectrograms = np.asarray(pixel_spectrograms)
  #   pixel_spectrograms = (pixel_spectrograms - np.min(pixel_spectrograms)) / \
  #                       (np.max(pixel_spectrograms) - np.min(pixel_spectrograms) + 1e-8)
    
  #   # Create and fit PCA
  #   pca_obj = PCA(n_components=3)
  #   sound_features = pca_obj.fit_transform(pixel_spectrograms)
    
  #   # Normalize PCA results to [0, 1]
  #   sound_features = (sound_features - np.min(sound_features)) / \
  #                   (np.max(sound_features) - np.min(sound_features) + 1e-8)
    
  #   # Reshape sound features to image grid (14x14x3)
  #   overlay = sound_features.reshape(14, 14, 3)
    
  #   # Process frame for display (convert from CHW to HWC format)
  #   frame_display = np.transpose(frame, (1, 2, 0))  # Convert from (3, 224, 224) to (224, 224, 3)
  #   frame_display = (frame_display - np.min(frame_display)) / \
  #                  (np.max(frame_display) - np.min(frame_display) + 1e-8)
    
  #   # Resize overlay to match frame size
  #   zoom_factors = (frame_display.shape[0]/overlay.shape[0], 
  #                  frame_display.shape[1]/overlay.shape[1], 
  #                  1)
  #   overlay_resized = zoom(overlay, zoom_factors, order=1)
    
  #   # Create figure
  #   plt.figure(figsize=(15, 5))
    
  #   # Original frame
  #   plt.subplot(1, 3, 1)
  #   plt.imshow(frame_display)
  #   plt.title('Original Frame')
  #   plt.axis('off')
    
  #   # Sound source overlay
  #   plt.subplot(1, 3, 2)
  #   plt.imshow(overlay_resized)
  #   plt.title('Sound Source Clusters')
  #   plt.axis('off')
    
  #   # Blended visualization
  #   plt.subplot(1, 3, 3)
  #   alpha = 0.6
  #   blended = (1 - alpha) * frame_display + alpha * overlay_resized
  #   plt.imshow(np.clip(blended, 0, 1))
  #   plt.title('Blended Visualization')
  #   plt.axis('off')
    
  #   plt.tight_layout()
  #   plt.savefig(output_path, dpi=300, bbox_inches='tight')
  #   plt.close()

##OPTION 3 - better seperation with duets:
# def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
#     # Normalize input
#     pixel_spectrograms = np.asarray(pixel_spectrograms)
#     pixel_spectrograms = (pixel_spectrograms - np.min(pixel_spectrograms)) / \
#                         (np.max(pixel_spectrograms) - np.min(pixel_spectrograms) + 1e-8)
    
#     # Apply PCA
#     pca_obj = PCA(n_components=3)
#     sound_features = pca_obj.fit_transform(pixel_spectrograms)
    
#     # Apply K-means clustering to separate instruments
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     clusters = kmeans.fit_predict(sound_features)
    
#     # Create separate masks for each instrument
#     mask1 = clusters == 0
#     mask2 = clusters == 1
    
#     # Enhance separation by modifying features based on clusters
#     sound_features_enhanced = sound_features.copy()
    
#     # Assign distinct colors to each cluster
#     sound_features_enhanced[mask1] *= [1.2, 0.8, 0.8]  # Reddish for first instrument
#     sound_features_enhanced[mask2] *= [0.8, 0.8, 1.2]  # Bluish for second instrument
    
#     # Normalize enhanced features
#     sound_features_enhanced = (sound_features_enhanced - np.min(sound_features_enhanced)) / \
#                             (np.max(sound_features_enhanced) - np.min(sound_features_enhanced) + 1e-8)
    
#     # Apply thresholding to focus on strong signals
#     threshold = 0.4
#     weak_signals = np.max(sound_features_enhanced, axis=1) < threshold
#     sound_features_enhanced[weak_signals] *= 0.3
    
#     # Reshape to image grid
#     overlay = sound_features_enhanced.reshape(14, 14, 3)
    
#     # Process frame
#     frame_display = np.transpose(frame, (1, 2, 0))
#     frame_display = (frame_display - np.min(frame_display)) / \
#                    (np.max(frame_display) - np.min(frame_display) + 1e-8)
    
#     # Resize overlay
#     zoom_factors = (frame_display.shape[0]/overlay.shape[0], 
#                    frame_display.shape[1]/overlay.shape[1], 
#                    1)
#     overlay_resized = zoom(overlay, zoom_factors, order=1)
    
#     # Create visualization
#     plt.figure(figsize=(15, 5))
    
#     # Original frame
#     plt.subplot(1, 3, 1)
#     plt.imshow(frame_display)
#     plt.title('Original Frame')
#     plt.axis('off')
    
#     # Sound source overlay
#     plt.subplot(1, 3, 2)
#     plt.imshow(overlay_resized)
#     plt.title('Sound Source Separation')
#     plt.axis('off')
    
#     # Blended visualization
#     plt.subplot(1, 3, 3)
#     alpha = 0.65
#     blended = (1 - alpha) * frame_display + alpha * overlay_resized
#     plt.imshow(np.clip(blended, 0, 1))
#     plt.title('Blended Visualization')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

##Version 1 = more ephasis on sound source, less on background.
def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
    """
    Visualize sound source clustering with balanced emphasis on source locations.
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from sklearn.cluster import KMeans
    from matplotlib.colors import LinearSegmentedColormap


    
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


  # ##Version two - less dark, less emphasis on sound source.
  # def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
  #   """
  #   Visualize sound source clustering by applying PCA to the spectrograms.
    
  #   Args:
  #       pixel_spectrograms: Array of shape (196, 65536) containing spectrograms
  #       frame: Video frame of shape (3, 224, 224)
  #       output_path: Path to save visualization
  #   """
  #   from sklearn.decomposition import PCA
  #   import numpy as np
  #   import matplotlib.pyplot as plt
  #   from scipy.ndimage import zoom
    
  #   # Ensure input is numpy array and normalized
  #   pixel_spectrograms = np.asarray(pixel_spectrograms)
  #   pixel_spectrograms = (pixel_spectrograms - np.min(pixel_spectrograms)) / \
  #                       (np.max(pixel_spectrograms) - np.min(pixel_spectrograms) + 1e-8)
    
  #   # Create and fit PCA
  #   pca_obj = PCA(n_components=3)
  #   sound_features = pca_obj.fit_transform(pixel_spectrograms)
    
  #   # Normalize PCA results to [0, 1]
  #   sound_features = (sound_features - np.min(sound_features)) / \
  #                   (np.max(sound_features) - np.min(sound_features) + 1e-8)
    
  #   # Reshape sound features to image grid (14x14x3)
  #   overlay = sound_features.reshape(14, 14, 3)
    
  #   # Process frame for display (convert from CHW to HWC format)
  #   frame_display = np.transpose(frame, (1, 2, 0))  # Convert from (3, 224, 224) to (224, 224, 3)
  #   frame_display = (frame_display - np.min(frame_display)) / \
  #                  (np.max(frame_display) - np.min(frame_display) + 1e-8)
    
  #   # Resize overlay to match frame size
  #   zoom_factors = (frame_display.shape[0]/overlay.shape[0], 
  #                  frame_display.shape[1]/overlay.shape[1], 
  #                  1)
  #   overlay_resized = zoom(overlay, zoom_factors, order=1)
    
  #   # Create figure
  #   plt.figure(figsize=(15, 5))
    
  #   # Original frame
  #   plt.subplot(1, 3, 1)
  #   plt.imshow(frame_display)
  #   plt.title('Original Frame')
  #   plt.axis('off')
    
  #   # Sound source overlay
  #   plt.subplot(1, 3, 2)
  #   plt.imshow(overlay_resized)
  #   plt.title('Sound Source Clusters')
  #   plt.axis('off')
    
  #   # Blended visualization
  #   plt.subplot(1, 3, 3)
  #   alpha = 0.6
  #   blended = (1 - alpha) * frame_display + alpha * overlay_resized
  #   plt.imshow(np.clip(blended, 0, 1))
  #   plt.title('Blended Visualization')
  #   plt.axis('off')
    
  #   plt.tight_layout()
  #   plt.savefig(output_path, dpi=300, bbox_inches='tight')
  #   plt.close()

##OPTION 3 - better seperation with duets:
# def visualize_sound_clustering(pixel_spectrograms, frame, output_path):
#     # Normalize input
#     pixel_spectrograms = np.asarray(pixel_spectrograms)
#     pixel_spectrograms = (pixel_spectrograms - np.min(pixel_spectrograms)) / \
#                         (np.max(pixel_spectrograms) - np.min(pixel_spectrograms) + 1e-8)
    
#     # Apply PCA
#     pca_obj = PCA(n_components=3)
#     sound_features = pca_obj.fit_transform(pixel_spectrograms)
    
#     # Apply K-means clustering to separate instruments
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     clusters = kmeans.fit_predict(sound_features)
    
#     # Create separate masks for each instrument
#     mask1 = clusters == 0
#     mask2 = clusters == 1
    
#     # Enhance separation by modifying features based on clusters
#     sound_features_enhanced = sound_features.copy()
    
#     # Assign distinct colors to each cluster
#     sound_features_enhanced[mask1] *= [1.2, 0.8, 0.8]  # Reddish for first instrument
#     sound_features_enhanced[mask2] *= [0.8, 0.8, 1.2]  # Bluish for second instrument
    
#     # Normalize enhanced features
#     sound_features_enhanced = (sound_features_enhanced - np.min(sound_features_enhanced)) / \
#                             (np.max(sound_features_enhanced) - np.min(sound_features_enhanced) + 1e-8)
    
#     # Apply thresholding to focus on strong signals
#     threshold = 0.4
#     weak_signals = np.max(sound_features_enhanced, axis=1) < threshold
#     sound_features_enhanced[weak_signals] *= 0.3
    
#     # Reshape to image grid
#     overlay = sound_features_enhanced.reshape(14, 14, 3)
    
#     # Process frame
#     frame_display = np.transpose(frame, (1, 2, 0))
#     frame_display = (frame_display - np.min(frame_display)) / \
#                    (np.max(frame_display) - np.min(frame_display) + 1e-8)
    
#     # Resize overlay
#     zoom_factors = (frame_display.shape[0]/overlay.shape[0], 
#                    frame_display.shape[1]/overlay.shape[1], 
#                    1)
#     overlay_resized = zoom(overlay, zoom_factors, order=1)
    
#     # Create visualization
#     plt.figure(figsize=(15, 5))
    
#     # Original frame
#     plt.subplot(1, 3, 1)
#     plt.imshow(frame_display)
#     plt.title('Original Frame')
#     plt.axis('off')
    
#     # Sound source overlay
#     plt.subplot(1, 3, 2)
#     plt.imshow(overlay_resized)
#     plt.title('Sound Source Separation')
#     plt.axis('off')
    
#     # Blended visualization
#     plt.subplot(1, 3, 3)
#     alpha = 0.65
#     blended = (1 - alpha) * frame_display + alpha * overlay_resized
#     plt.imshow(np.clip(blended, 0, 1))
#     plt.title('Blended Visualization')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()


def visualize_activations(frame, visual_features, audio_features, output_path, channel_idx=None):
    """
    Create visualization with more transparent overlay for better visibility of original image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import zoom
    
    # Create custom colormap similar to the paper
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Set up the figure
    fig = plt.figure(figsize=(4, 10))
    
    # Add channel number as title
    if channel_idx is not None:
        plt.suptitle(f'channel {channel_idx}', y=0.98, fontsize=10)
    
    # Create subplots with custom heights
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.15)
    
    # 1. Video frame
    ax1 = plt.subplot(gs[0])
    frame_display = frame.transpose(1, 2, 0)
    frame_display = (frame_display - frame_display.min()) / (frame_display.max() - frame_display.min())
    ax1.imshow(frame_display)
    ax1.set_title('Video frame', pad=5, fontsize=10)
    ax1.axis('off')
    
    # 2. Visual activations with reduced alpha
    ax2 = plt.subplot(gs[1])
    visual_activation = np.mean(visual_features, axis=0)
    visual_activation = (visual_activation - visual_activation.min()) / \
                       (visual_activation.max() - visual_activation.min() + 1e-8)
    zoom_factors = (frame_display.shape[0] / visual_activation.shape[0],
                   frame_display.shape[1] / visual_activation.shape[1])
    visual_activation_resized = zoom(visual_activation, zoom_factors, order=1)
    
    # Show original image
    ax2.imshow(frame_display)
    # Overlay heatmap with reduced alpha (0.5 instead of 0.7)
    ax2.imshow(visual_activation_resized, cmap=cmap, alpha=0.5)  # Reduced alpha here
    ax2.set_title('Visual activations', pad=5, fontsize=10)
    ax2.axis('off')
    
    # 3. Audio activations
    ax3 = plt.subplot(gs[2])
    audio_activation = np.mean(audio_features, axis=0)
    audio_activation = (audio_activation - audio_activation.min()) / \
                      (audio_activation.max() - audio_activation.min() + 1e-8)
    ax3.imshow(audio_activation, cmap=cmap, aspect='auto')
    ax3.set_title('Audio activations', pad=5, fontsize=10)
    ax3.axis('off')
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()