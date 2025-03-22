import os
from typing import List, Generator, Dict, Union, Tuple
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny
# local imports
import utils



@dataclass
class Segment:
    segment: np.ndarray
    start_row: int
    end_row: int # might not keep this
    is_standalone: bool
    split_after: bool


@dataclass
class ImageSegments:
    """ store all high-information segments from an image, the indices of the remaining blank spaces, and information regarding the split points """
    filename: str
    segments: List[Segment]
    blank_row_indices: List[int]
    blank_row_colors: List[Tuple[int, int, int]]  # RGB row averages
    height: int
    width: int




class ImageSegmentGenerator:
    def __init__(self, image_files: List[np.ndarray], image_dir: os.PathLike, target_width: int, min_blank_run: int = 10):
        self.image_files = image_files
        self.image_dir = image_dir
        self.target_width = target_width
        self.min_blank_run = min_blank_run

    def _detect_split_points(self, image: np.ndarray) -> List[int]:
        """ detects eligible blank rows for vertical splits in the image using Canny edge detection """
        # TODO: add more advanced logic to avoid cuts in the middle of comic panels that don't have much detail (i.e. homogeneous background regions)
        img_gray = rgb2gray(image)
        edges_canny = canny(img_gray, sigma=1.0)
        split_points = []
        blank_candidates = []
        for row in range(edges_canny.shape[0]):
            if np.any(edges_canny[row, :]):
                if len(blank_candidates) >= self.min_blank_run:
                    median_blank = int(np.median(blank_candidates))
                    split_points.append(median_blank)
                blank_candidates = []
            else:
                blank_candidates.append(row)
        if len(blank_candidates) >= self.min_blank_run:
            median_blank = int(np.median(blank_candidates))
            split_points.append(median_blank)
        return split_points

    def _get_segment_results(self, img: np.ndarray, img_type: str, split: bool) -> List[Dict[str, Union[str, np.ndarray, bool]]]:
        return {
            "type": img_type,
            "image": img,
            "split_after": split
        }

    def __iter__(self) -> Generator[Dict[str, Union[str, np.ndarray, bool]], None, None]:
        for file in tqdm(self.image_files, desc="Processing images"):
            img = imread(os.path.join(self.image_dir, file))
            resized = utils.resize_image(img, self.target_width)
            if utils.should_treat_as_standalone(resized):
                yield self._get_segment_results(resized, "standalone", False)
                continue
            split_points = self._detect_split_points(resized)
            prev = 0
            for i, split in enumerate(split_points + [resized.shape[0]]):
                segment = resized[prev:split, :, :]
                is_valid_break = (i < len(split_points))
                yield self._get_segment_results(segment, "segment", is_valid_break)
                prev = split



def accumulate_segments(segment_stream: ImageSegmentGenerator, min_height: int) -> Generator[List[np.ndarray], None, None]:
    """ Accumulates segments from the generator into stacked images based on min_height and yields a stacked image """
    buffer = None
    buffer_height = 0
    for item in segment_stream:
        if item["type"] == "standalone":
            if buffer is not None:
                yield buffer
                buffer = None
                buffer_height = 0
            yield item["image"]
            continue
        segment = item["image"]
        split_after = item["split_after"]
        if buffer is None:
            buffer = segment
            buffer_height = segment.shape[0]
        else:
            buffer = np.vstack((buffer, segment))
            buffer_height += segment.shape[0]
        if buffer_height >= min_height and split_after:
            yield buffer
            buffer = None
            buffer_height = 0
    if buffer is not None:
        yield buffer



#~ TODO: create a similar utility for paginated document formats like pdf that still splits images but attempts to limit height to a letter form factor (8.5x11) or something similar
    #~ TODO - add a --page_size argument that takes in a string like "A4" or "Letter" and then sets the target height accordingly
    #~ the new utility should attempt to fit as many actual comic panels in a page as possible by checking if reducing the blank rows between panels is possible without cutting off any panels
        #~ this should be tempered by requiring a minimum number of blank rows between panels to avoid them running into each other visually (i.e. 5 blank rows minimum)
        #~ failing that, it should do the opposite and try to add more blank rows to empty regions to reach the target page height

#~ TODO: add a --dpi argument to set the dpi of the output images (for printing purposes)

### TRIAL 3

# def stack_segments(segments_info, min_gap):
#     """Stack segments tightly with min_gap spacing, return result and its height."""
#     segments = [seg["image"] for seg in segments_info]
#     gaps = [np.ones((min_gap, segments[0].shape[1], 3), dtype=np.uint8) * 255] * (len(segments) - 1)
#     stacked = []
#     for i, seg in enumerate(segments):
#         stacked.append(seg)
#         if i < len(gaps):
#             stacked.append(gaps[i])
#     result = np.vstack(stacked)
#     return result, result.shape[0]


# def build_strict_page(segments_info, page_height, min_gap, max_gap):
#     """Assembles a page from segments, padding or resizing only if necessary."""
#     stacked_img, current_height = stack_segments(segments_info, min_gap)
#     if current_height == page_height:
#         return stacked_img
#     elif current_height < page_height:
#         pad_total = page_height - current_height
#         if pad_total <= max_gap:
#             top_pad = pad_total // 2
#             bottom_pad = pad_total - top_pad
#             h, w = stacked_img.shape[:2]
#             top = np.ones((top_pad, w, 3), dtype=np.uint8) * 255
#             bottom = np.ones((bottom_pad, w, 3), dtype=np.uint8) * 255
#             return np.vstack([top, stacked_img, bottom])
#         else:
#             # resize image to fit page height
#             from skimage.transform import resize
#             h, w = stacked_img.shape[:2]
#             resized = resize(stacked_img, (page_height, w), preserve_range=True, anti_aliasing=True)
#             return resized.astype(np.uint8)
#     else:
#         # safety fallback (shouldn’t hit with proper logic)
#         print("Warning: page exceeds target height unexpectedly. Resizing.")
#         from skimage.transform import resize
#         h, w = stacked_img.shape[:2]
#         resized = resize(stacked_img, (page_height, w), preserve_range=True, anti_aliasing=True)
#         return resized.astype(np.uint8)


# def paginate_segments(segment_stream, page_height, min_gap=5, max_gap=40):
#     """Paginate comic segments while preserving panel integrity and visual balance."""
#     buffer = []
#     for item in segment_stream:
#         if item["type"] == "standalone":
#             if buffer:
#                 yield build_strict_page(buffer, page_height, min_gap, max_gap)
#                 buffer = []
#             yield item["image"]
#             continue
#         buffer.append(item)
#         # Check if we can safely finalize this page
#         stacked, total_height = stack_segments(buffer, min_gap)
#         if total_height > page_height:
#             # remove last segment and try again
#             last = buffer.pop()
#             if buffer:
#                 yield build_strict_page(buffer, page_height, min_gap, max_gap)
#             buffer = [last]
#     if buffer:
#         yield build_strict_page(buffer, page_height, min_gap, max_gap)


def simulate_height(segments_info, min_gap):
    """Compute height of segments stacked with min_gap between each (no real data ops)."""
    heights = [s["image"].shape[0] for s in segments_info]
    return sum(heights) + min_gap * (len(heights) - 1)


def construct_page(segments_info, target_height, min_gap, max_gap):
    """Build the actual page image with padding or resizing only at this step."""
    from skimage.transform import resize
    segments = [s["image"] for s in segments_info]
    h_list = [seg.shape[0] for seg in segments]
    content_height = sum(h_list)
    num_gaps = len(segments) - 1
    gap_total = min_gap * num_gaps
    full_height = content_height + gap_total
    # Compose image with min_gap first
    stacked = []
    for i, seg in enumerate(segments):
        stacked.append(seg)
        if i < num_gaps:
            gap = np.ones((min_gap, seg.shape[1], 3), dtype=np.uint8) * 255
            stacked.append(gap)
    composed = np.vstack(stacked)
    if full_height == target_height:
        return composed
    elif full_height < target_height:
        pad_needed = target_height - full_height
        if pad_needed <= max_gap:
            top_pad = pad_needed // 2
            bottom_pad = pad_needed - top_pad
            w = composed.shape[1]
            top = np.ones((top_pad, w, 3), dtype=np.uint8) * 255
            bottom = np.ones((bottom_pad, w, 3), dtype=np.uint8) * 255
            return np.vstack([top, composed, bottom])
        else:
            # resize vertically
            w = composed.shape[1]
            resized = resize(composed, (target_height, w), preserve_range=True, anti_aliasing=True)
            return resized.astype(np.uint8)
    else:
        # resize because stack exceeded allowed height
        w = composed.shape[1]
        resized = resize(composed, (target_height, w), preserve_range=True, anti_aliasing=True)
        return resized.astype(np.uint8)


def paginate_segments(segment_stream, page_height, min_gap=5, max_gap=40):
    """ Yield finalized page images from segments, using simulation-based pagination. """
    buffer = []
    for item in segment_stream:
        if item["type"] == "standalone":
            if buffer:
                yield construct_page(buffer, page_height, min_gap, max_gap)
                buffer = []
            yield item["image"]
            continue
        buffer.append(item)
        simulated_height = simulate_height(buffer, min_gap)
        if simulated_height > page_height:
            last = buffer.pop()
            if buffer:
                yield construct_page(buffer, page_height, min_gap, max_gap)
            buffer = [last]
    if buffer:
        yield construct_page(buffer, page_height, min_gap, max_gap)
