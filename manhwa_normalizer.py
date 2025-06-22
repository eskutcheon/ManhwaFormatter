import os
from typing import List, Generator, Dict, Union, Tuple
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import resize
# local imports
import utils


# --- Helper functions for clarity and efficiency ---

def _detect_edges(gray: np.ndarray,
                  low_pct: float = 0.1,
                  high_pct: float = 0.5,
                  sigma: float = 1.0) -> np.ndarray:
    """ Compute adaptive hysteresis thresholds and run Canny on a grayscale image """
    lo, hi = utils.compute_adaptive_thresholds(gray, low_pct=low_pct, high_pct=high_pct)
    return canny(gray, sigma=sigma, low_threshold=lo, high_threshold=hi)


def _find_blank_runs(blank_mask: np.ndarray, min_run: int) -> List[Tuple[int,int]]:
    """ Given a 1D boolean mask of blank rows, return (start, end) indices for each contiguous run of length >= min_run """
    # pad and diff to find run boundaries
    mask_int = blank_mask.view(np.int8)
    diffs = np.diff(np.concatenate(([0], mask_int, [0])))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_run]


def _region_median_color(image: np.ndarray, start: int, end: int) -> Tuple[int,int,int]:
    """ compute median RGB color over rows [start:end] in the image """
    med = np.median(image[start:end], axis=(0,1)).astype(int)
    return tuple(med)


def _plan_and_apply_trimming(stacked: np.ndarray,
                             blank_rows: List[int],
                             total_to_remove: int,
                             min_gap: int) -> np.ndarray:
    """ trim exactly `total_to_remove` rows from stacked image using blank_rows,
        respecting a minimum gap in each blank region, else fallback to resizing the image
    """
    H, W = stacked.shape[:2]
    # group blank_rows into contiguous regions
    regions = []
    current = []
    for idx in sorted(blank_rows):
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            regions.append(current)
            current = [idx]
    if current:
        regions.append(current)
    # filter regions that would be too small after removal
    regions = [r for r in regions if len(r) - total_to_remove >= min_gap] #& UPDATED to len(r) - total_to_remove >= min_gap
    if not regions:
        return resize(stacked, (H - total_to_remove, W), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    # proportional removal plan
    removable_total = sum(max(0, len(r) - min_gap) for r in regions)
    if removable_total < total_to_remove:
        # if not enough removable rows, fallback to resizing
        return resize(stacked, (H - total_to_remove, W), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    # distribute the removals proportionally across regions
    # remove_plan = {}
    # removable_total = sum(max(0, len(r) - self.min_gap) for r in regions)
    # if removable_total < total_to_remove:
    #     return resize(stacked, (target_height, W), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    # for region in regions:
    #     max_removable = len(region) - self.min_gap
    #     prop = max_removable / removable_total
    #     to_remove = int(round(prop * total_to_remove))
    #     if to_remove > 0:
    #         remove_plan[tuple(region)] = region[:to_remove]
    # rows_to_strip = set()
    # for group in remove_plan.values():
    #     rows_to_strip.update(group)
    rows_to_strip = set()
    for region in regions:
        max_rem = len(region) - min_gap
        to_rem = int(round((max_rem / removable_total) * total_to_remove))
        rows_to_strip.update(region[:to_rem])
    keep_idx = np.array([i for i in range(H) if i not in rows_to_strip])
    return stacked[keep_idx, :, :]



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



class ImageSegmentGenerator:
    def __init__(self, image_files: List[np.ndarray], image_dir: os.PathLike, target_width: int, min_blank_run: int = 10):
        self.image_files = image_files
        self.image_dir = image_dir
        self.target_width = target_width
        self.min_blank_run = min_blank_run

    def _add_split_point(self, blank_candidates: List[int], split_points: List[int]):
        if len(blank_candidates) >= self.min_blank_run:
            midpt = int(np.median(blank_candidates))
            split_points.append(midpt)

    def _detect_blank_regions(self, image: np.ndarray) -> Tuple[List[int], List[int], List[Tuple[int, int, int]]]:
        """ detects eligible blank rows for vertical splits in the image using Canny edge detection """
        # TODO: add more advanced logic to avoid cuts in the middle of comic panels that don't have much detail (i.e. homogeneous background regions)
        img_gray = rgb2gray(image)
        edges = _detect_edges(img_gray)
        blank_mask = ~edges.any(axis=1)
        runs = _find_blank_runs(blank_mask, self.min_blank_run)
        split_points = []
        blank_row_indices = []
        blank_row_colors = []
        # row = 0
        # while row < edges.shape[0]:
        #     # Find the start of a blank run
        #     if not np.any(edges[row]):
        #         start_row = row
        #         # count the number of blank rows before an edge is detected
        #         while row < edges.shape[0] and not np.any(edges[row]):
        #             row += 1
        #         end_row = row  # exclusive in slicing
        #         # if the run is long enough, process it
        #         run_length = end_row - start_row
        #         #& NEW: check for low variance in the blank region
        #         if run_length >= self.min_blank_run and utils.is_low_variance_region(image, (start_row, end_row)):
        #             midpoint = start_row + run_length // 2
        #             split_points.append(midpoint)
        #             avg_rgb = np.median(image[start_row:end_row], axis=(0, 1)).astype(int)
        #             # append blank row indices and average colors for the run
        #             blank_row_indices.extend(range(start_row, end_row))
        #             blank_row_colors.extend([tuple(avg_rgb)] * run_length)
        #     else:
        #         row += 1
        for start, end in runs:
            if utils.is_low_variance_region(image, (start, end)):
                split_points.append((start + end) // 2)
                blank_row_indices.extend(range(start, end))
                median_color = _region_median_color(image, start, end)
                blank_row_colors.extend([median_color] * (end - start))
        return split_points, blank_row_indices, blank_row_colors

    def __iter__(self) -> Generator[Dict[str, Union[str, np.ndarray, bool]], None, None]:
        for file in tqdm(self.image_files, desc="Processing images"):
            img_path = os.path.join(self.image_dir, file)
            img = imread(img_path)
            resized = utils.resize_image(img, self.target_width)
            # may not keep the height and width since I could just read the segment shape directly
            H, W = resized.shape[:2]
            if utils.should_treat_as_standalone(resized):
                segment = Segment(resized, start_row = 0, end_row = H, is_standalone = True, split_after = False)
                yield ImageSegments(file, [segment], [], [], 0)
                continue
            split_points, blank_rows, blank_colors = self._detect_blank_regions(resized)
            segments = []
            prev = 0
            for i, split in enumerate(split_points + [resized.shape[0]]):
                img_segment = resized[prev:split, ...]
                is_valid_break = (i < len(split_points)) # - 1)
                segments.append(Segment(img_segment, start_row=prev, end_row=split, is_standalone=False, split_after=is_valid_break))
                prev = split
            # yield self._get_segment_results(file, segments, blank_rows, blank_colors)
            yield ImageSegments(
                filename=file,
                segments=segments,
                blank_row_indices=blank_rows,
                blank_row_colors=blank_colors,
                height=len(blank_rows)
            )



class SegmentStitcher:
    def __init__(self, min_gap: int = 5, max_gap: int = 40, resize_threshold_ratio: float = 0.05):
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.resize_threshold_ratio = resize_threshold_ratio
        self.buffer: List[Segment] = []
        self.buffer_height = 0

    def _reset_buffer(self):
        self.buffer = []
        self.buffer_height = 0

    def _stack_segments(self) -> np.ndarray:
        return np.vstack([seg.segment for seg in self.buffer])

    def _finalize_page(self,
                       target_height: int,
                       top_color=(255, 255, 255),
                       bottom_color=(255, 255, 255),
                       blank_row_indices: List[int] = None) -> np.ndarray:
        stacked = self._stack_segments()
        current_height = stacked.shape[0]
        return self._adjust_page_to_height(stacked, current_height, target_height, top_color, bottom_color, blank_row_indices)

    def _adjust_page_to_height(self, stacked: np.ndarray, current_height: int, target_height: int,
                                top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int],
                                blank_row_indices: List[int] = None) -> np.ndarray:
        diff = target_height - current_height
        H, W = stacked.shape[:2]
        resize_threshold = int(target_height * self.resize_threshold_ratio)
        if abs(diff) <= resize_threshold:
            return resize(stacked, (target_height, W), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        if diff > 0:
            top_pad = diff // 2
            bottom_pad = diff - top_pad
            top = np.ones((top_pad, W, 3), dtype=np.uint8) * np.array(top_color, dtype=np.uint8)
            bottom = np.ones((bottom_pad, W, 3), dtype=np.uint8) * np.array(bottom_color, dtype=np.uint8)
            return np.vstack([top, stacked, bottom])
        if diff < 0 and blank_row_indices:
            return _plan_and_apply_trimming(stacked, blank_row_indices, total_to_remove=abs(diff), min_gap=self.min_gap)
        # fallback approach: resize to target height
        return resize(stacked, (target_height, W), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    def accumulate_segments(self, segment_stream: Generator[ImageSegments, None, None], min_height: int):
        self._reset_buffer()
        for img_obj in segment_stream:
            for seg in img_obj.segments:
                if seg.is_standalone:
                    if self.buffer:
                        yield self._stack_segments()
                        self._reset_buffer()
                    yield seg.segment
                    continue
                self.buffer.append(seg)
                self.buffer_height += seg.segment.shape[0] + (self.min_gap if len(self.buffer) > 1 else 0)
                if self.buffer_height >= min_height and seg.split_after:
                    yield self._stack_segments()
                    self._reset_buffer()
        if self.buffer:
            yield self._stack_segments()

    def paginate_segments(self, segment_stream: Generator[ImageSegments, None, None], page_height: int):
        self._reset_buffer()
        for img_obj in segment_stream:
            top_color = img_obj.blank_row_colors[0] if img_obj.blank_row_colors else (255, 255, 255)
            bottom_color = img_obj.blank_row_colors[-1] if img_obj.blank_row_colors else (255, 255, 255)
            blank_rows = img_obj.blank_row_indices
            for seg in img_obj.segments:
                if seg.is_standalone:
                    if self.buffer:
                        yield self._finalize_page(page_height, top_color, bottom_color, blank_rows)
                        self._reset_buffer()
                    yield seg.segment
                    continue
                seg_height = seg.segment.shape[0]
                gap = self.min_gap if self.buffer else 0
                projected_height = self.buffer_height + seg_height + gap
                if projected_height > page_height and self.buffer and self.buffer[-1].split_after:
                    yield self._finalize_page(page_height, top_color, bottom_color, blank_rows)
                    self._reset_buffer()
                self.buffer.append(seg)
                self.buffer_height += seg_height + (self.min_gap if len(self.buffer) > 1 else 0)
        if self.buffer:
            yield self._finalize_page(page_height, top_color, bottom_color, blank_rows)