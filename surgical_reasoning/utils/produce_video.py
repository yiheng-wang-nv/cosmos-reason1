# !pip install zarr matplotlib numpy simplejpeg

import zarr
import numpy as np
import matplotlib.pyplot as plt
import simplejpeg
import numcodecs
from numcodecs import register_codec
import os
import cv2

def _assert_shape(arr: np.ndarray, expected_shape: tuple[int | None, ...]):
    """Asserts that the shape of an array matches the expected shape."""
    assert len(arr.shape) == len(expected_shape), (arr.shape, expected_shape)
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, (arr.shape, expected_shape)


class JpegCodec(numcodecs.abc.Codec):
    """Codec for JPEG compression.
    Encodes image chunks as JPEGs. Assumes that chunks are uint8 with shape (1, H, W, 3).
    """
    codec_id = "pi_jpeg"

    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality

    def encode(self, buf):
        _assert_shape(buf, (1, None, None, 3))
        assert buf.dtype == "uint8"
        return simplejpeg.encode_jpeg(buf[0], quality=self.quality)

    def decode(self, buf, out=None):
        img = simplejpeg.decode_jpeg(buf, buffer=out)
        return img[np.newaxis, ...]

register_codec(JpegCodec)

def save_frames_and_video(zarr_array, image_folder_path, video_path, fps=30):
    print(f"Zarr array shape: {zarr_array.shape}")
    
    # Get video dimensions from first frame
    first_frame = zarr_array[0]
    height, width = first_frame.shape[:2]
    
    # Initialize video writer (OpenCV uses BGR, so we'll convert)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(len(zarr_array)):
        frame = zarr_array[i]
        
        # Check if frame has correct dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"ERROR: Frame {i} has invalid shape {frame.shape}")
            continue
        
        # Convert to numpy array and ensure it's contiguous
        frame = np.ascontiguousarray(frame)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save image using OpenCV
        img_path = os.path.join(image_folder_path, f"{i:05d}.png")
        cv2.imwrite(img_path, frame_bgr)
        
        # Write frame to video
        video_writer.write(frame_bgr)
        
    video_writer.release()
    print(f"Saved {len(zarr_array)} frames to {image_folder_path} and video to {video_path}")

root_dir = "/localhome/local-vennw/code/surgical_data/"
tissues = os.listdir(root_dir)
tissues = [t for t in tissues if "tissue" in t]

for tissue in tissues:
    tasks = os.listdir(f'{root_dir}/{tissue}')
    for task in tasks:
        for sample in os.listdir(f'{root_dir}/{tissue}/{task}'):
            print(f'Processing sample: {sample}')
            output_video_path = f'{root_dir}/{tissue}/{task}/{sample}/videos'
            os.makedirs(output_video_path, exist_ok=True)
            for data in ['left', 'right', 'endo_psm1', 'endo_psm2']:
                pth = f'{root_dir}/{tissue}/{task}/{sample}/{data}'
                if os.path.exists(pth):
                    data_array = zarr.open(pth, mode='r')

                    data_images_path = f'{root_dir}/{tissue}/{task}/{sample}/{data}_images'
                    os.makedirs(data_images_path, exist_ok=True)

                    save_frames_and_video(data_array, data_images_path, f'{output_video_path}/{data}.mp4')
