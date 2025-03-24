import os
import struct
from typing import List, Tuple, Generator
import numpy as np
from collections import Counter
from skimage.io import imread, imsave
from skimage.transform import resize


def get_page_height(page_size: str, dpi: int = 300) -> int:
    """ return target image height in pixels for a given page size """
    sizes = {"letter": (8.5, 11), "a4": (8.27, 11.69)}
    # TODO: might need to resize width as well to maintain aspect ratio when creating pdfs
    width_in, height_in = sizes.get(page_size.lower(), sizes["letter"])
    return int(height_in * dpi)

def should_treat_as_standalone(image: np.ndarray) -> bool:
    """ Return True if the image is landscape (wider than tall). """
    h, w = image.shape[:2]
    return w > h

def mode_width(dimensions) -> int:
    widths = [w for _, w in dimensions]
    return Counter(widths).most_common(1)[0][0]

def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """ Resizes image while preserving aspect ratio. """
    h, w = image.shape[:2]
    if w == target_width or should_treat_as_standalone(image):
        return image
    new_height = int((target_width / w) * h)
    resized = resize(image, (new_height, target_width), anti_aliasing=True, preserve_range=True)
    return resized.astype(np.uint8)


def save_new_images(images: Generator[List[np.ndarray], None, None], output_dir: os.PathLike):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save each image yielded from the generator to the output directory
    for i, img in enumerate(images):
        page_num = get_padded_page_number(i + 1)
        imsave(os.path.join(output_dir, f"page_{page_num}.png"), img)


def filter_duplicate_files(image_files, supported_filetypes):
    #? NOTE: this function may end up skipping some title pages that have the same name as the first (non-title) page but different extensions
        # ex: when scraping from MangaDex, the first page is often "{...}001.jpg" and the title page is "{...}001.gif"
    # create a mapping of file extensions to their priority based on the order in supported_filetypes
    filetype_priority = {ext: i for i, ext in enumerate(supported_filetypes)}
    # dictionary to store the highest-priority file for each basename
    file_dict = {}
    duplicates_detected = False  # Flag to track if duplicates are found
    for file in image_files:
        basename, ext = os.path.splitext(file)
        ext = ext.lower().lstrip('.')
        if ext in filetype_priority:  # Only consider supported file types
            if basename not in file_dict:
                file_dict[basename] = file
            elif filetype_priority[ext] < filetype_priority[os.path.splitext(file_dict[basename])[1].lstrip('.')]:
                duplicates_detected = True
                file_dict[basename] = file
    # print a warning if duplicates were detected and inform the user about the priority
    if duplicates_detected:
        print(f"WARNING: Duplicate filenames detected. Reference files are chosen based on the following priority: {', '.join(supported_filetypes)}")
    # return the list of files, keeping only the highest-priority duplicates
    return list(file_dict.values())


#? NOTE: tested the use of the old version and `get_image_dimensions_from_metadata` on a directory of 132 images with total size 52.1 MB
    # RESULTS: average over 10 runs was 3.442 seconds for old version, 0.016 seconds for new version (using metadata)
def get_image_dimensions(image_files: List[str], image_dir: os.PathLike) -> List[Tuple[int, int]]:
    dimensions = []
    for file in image_files:
        file_path = os.path.join(image_dir, file)
        try:
            width, height = get_image_dimensions_from_metadata(file_path)
            dimensions.append((height, width))
        except ValueError:
            img = imread(file_path)
            dimensions.append(img.shape[:2])  # (height, width)
    return dimensions

def get_image_dimensions_from_metadata(file_path):
    with open(file_path, 'rb') as f:
        data = f.read(24)
        # check if the file is a PNG file
        if data.startswith(b'\211PNG\r\n\032\n') and data[12:16] == b'IHDR':
            width, height = struct.unpack('>II', data[16:24])
            return width, height
        # check if the file is a JPEG file
        elif data[0:2] == b'\xff\xd8':
            f.seek(0)
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf:
                f.seek(size, 1)
                byte = f.read(1)
                while ord(byte) == 0xff:
                    byte = f.read(1)
                ftype = ord(byte)
                size = struct.unpack('>H', f.read(2))[0] - 2
            f.seek(1, 1)  # skip precision byte
            height, width = struct.unpack('>HH', f.read(4))
            return width, height
        else:
            raise ValueError("Unsupported image format")


def get_padded_page_number(page_number: int) -> str:
    """ returns a zero-padded page number for consistent naming """
    return f"{page_number:03d}"