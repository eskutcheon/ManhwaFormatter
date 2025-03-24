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
        # TODO: experiment with different hysteresis thresholds for edge detection - in an effort to address the TODO above
        edges = canny(img_gray, sigma=1.0)
        split_points = []
        blank_row_indices = []
        blank_row_colors = []
        row = 0
        while row < edges.shape[0]:
            # Find the start of a blank run
            if not np.any(edges[row]):
                start_row = row
                # count the number of blank rows before an edge is detected
                while row < edges.shape[0] and not np.any(edges[row]):
                    row += 1
                end_row = row  # exclusive in slicing
                # if the run is long enough, process it
                run_length = end_row - start_row
                if run_length >= self.min_blank_run:
                    midpoint = start_row + run_length // 2
                    split_points.append(midpoint)
                    # append blank row indices and average colors for the run
                    blank_row_indices.extend(range(start_row, end_row))
                    avg_rgb = np.mean(image[start_row:end_row], axis=(0, 1)).astype(int)
                    blank_row_colors.extend([tuple(avg_rgb)] * run_length)
            else:
                row += 1
        return split_points, blank_row_indices, blank_row_colors

    def _get_segment_results(self, filename: str, segments: List[Segment], blank_indices: List[int], row_colors: List[Tuple[int, int, int]]) -> ImageSegments:
        return ImageSegments(filename, segments, blank_row_indices=blank_indices, blank_row_colors=row_colors, height=len(blank_indices))

    def __iter__(self) -> Generator[Dict[str, Union[str, np.ndarray, bool]], None, None]:
        for file in tqdm(self.image_files, desc="Processing images"):
            img_path = os.path.join(self.image_dir, file)
            img = imread(img_path)
            resized = utils.resize_image(img, self.target_width)
            # may not keep the height and width since I could just read the segment shape directly
            H, W = resized.shape[:2]
            if utils.should_treat_as_standalone(resized):
                segment = Segment(resized, start_row = 0, end_row = H, is_standalone = True, split_after = False)
                yield self._get_segment_results(file, [segment], [], [])
                continue
            split_points, blank_rows, blank_colors = self._detect_blank_regions(resized)
            segments = []
            prev = 0
            for i, split in enumerate(split_points + [resized.shape[0]]):
                img_segment = resized[prev:split, ...]
                is_valid_break = (i < len(split_points)) # - 1)
                segments.append(Segment(img_segment, start_row=prev, end_row=split, is_standalone=False, split_after=is_valid_break))
                prev = split
            yield self._get_segment_results(file, segments, blank_rows, blank_colors)



class SegmentStitcher:
    def __init__(self, min_gap: int = 5, max_gap: int = 40):
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.buffer: List[Segment] = []
        self.buffer_height = 0

    def _stack_segments(self) -> np.ndarray:
        """ Stacks the segments in the buffer with the specified gap """
        return np.vstack([seg.segment for seg in self.buffer])

    def _finalize_page(self, target_height: int) -> np.ndarray:
        """ Finalizes a page by stacking segments and adding padding if necessary """
        stacked = self._stack_segments()
        h, w = stacked.shape[:2]
        pad_needed = target_height - h
        if pad_needed == 0:
            return stacked
        elif pad_needed > 0 and pad_needed <= self.max_gap:
            #! FIXME: use the new row color averages to pad with a more visually consistent color
            ###~ save previous page end color as a class variable to reference for padding; end color should just reference the final segment colors
            top = np.ones((pad_needed // 2, w, 3), dtype=np.uint8) * 255
            bottom = np.ones((pad_needed - pad_needed // 2, w, 3), dtype=np.uint8) * 255
            return np.vstack([top, stacked, bottom])
        else:
            # if too tall — force resize
            return resize(stacked, (target_height, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    def accumulate_segments(self, segment_stream: Generator[ImageSegments, None, None], min_height: int) -> Generator[np.ndarray, None, None]:
        """ Accumulates segments into stacked images based on min_height and yields a stacked image """
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
                ###~ region below is where this function differs from paginate_segments
                if self.buffer_height >= min_height and seg.split_after:
                    yield self._stack_segments()
                    self._reset_buffer()
        if self.buffer:
            yield self._stack_segments()

    def paginate_segments(self, segment_stream: Generator[ImageSegments, None, None], page_height: int) -> Generator[np.ndarray, None, None]:
        """ Paginates segments into pages based on page_height and yields a finalized page """
        self._reset_buffer()
        for img_obj in segment_stream:
            for seg in img_obj.segments:
                if seg.is_standalone:
                    if self.buffer:
                        yield self._finalize_page(page_height)
                        self._reset_buffer()
                    #??? Should this be wrapped in an else statement since it would otherwise immediately yield the next segment the next iteration?
                    yield seg.segment
                    continue
                # # add heights of each segment in the buffer and add the min_gap for each segment except the last one
                # total_height = sum(seg.segment.shape[0] for seg in buffer) + min_gap * (len(buffer) - 1)
                seg_height = seg.segment.shape[0]
                gap_height = self.min_gap if self.buffer else 0
                projected_height = self.buffer_height + seg_height + gap_height
                # if adding this segment would exceed page height, and we have a valid break
                if projected_height > page_height and self.buffer and self.buffer[-1].split_after:
                    yield self._finalize_page(page_height)
                    self._reset_buffer()
                # add the current segment regardless
                self.buffer.append(seg)
                self.buffer_height += seg_height + (self.min_gap if len(self.buffer) > 1 else 0)
        if self.buffer:
            yield self._finalize_page(page_height)

    def _reset_buffer(self):
        """ Resets the buffer and height for a new page """
        self.buffer = []
        self.buffer_height = 0