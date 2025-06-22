import os
from tqdm import tqdm
import zipfile
import shutil
import subprocess
from PIL import Image
from typing import List


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')
ARCHIVE_TOOLS = ('rar', '7z', 'winrar')



def _get_supported_tool(tool_preference: str = None):
    """ returns the name of the first available supported archiving tool (rar, 7z, winrar) or None if none are in PATH """
    candidates = []
    if tool_preference:
        candidates.append(tool_preference)
    candidates += list(ARCHIVE_TOOLS)
    for tool in candidates:
        tool_path = shutil.which(tool)
        if not tool_path:
            continue
        return tool
    return None


def _get_image_filenames(src_dir):
    """ returns a sorted list of image filenames in the source directory """
    return sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS))


# TODO: may move to utils.py later; though it's pretty similar to `resize_image`, but just uses PIL instead of skimage
def rescale_image(img: Image.Image, target_width: int) -> Image.Image:
    """ resizes image while preserving aspect ratio using PIL """
    width, height = img.size
    if width == target_width:
        return img
    new_height = int((target_width / width) * height)
    return img.resize((target_width, new_height), Image.LANCZOS)


def create_cbz(
    src_dir,
    archive_name = "output.cbz",
    clear_directory = False,
    compression_level: int = 5,
    as_fallback: bool = False
) -> None:
    """ creates a .cbz file (ZIP archive) from images in the output directory or .cbr (RAR archive) if archive_name ends with .cbr """
    # if archive_name.endswith('.cbr'): #? I think zipfile can correctly create .rar files and thus should be able to do .cbr
    #     archive_name = archive_name.replace('.cbr', '.cbz')
    cbz_path = os.path.join(src_dir, "..", archive_name)
    with zipfile.ZipFile(cbz_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
        for filename in _get_image_filenames(src_dir):
            zipf.write(os.path.join(src_dir, filename), arcname=filename)
    print(f"Archive file created: {cbz_path}")
    if clear_directory and not as_fallback:
        _cleanup_src_directory(src_dir)


#! FIXME: doesn't work on Windows - need to test on Linux
    #! requires WinRAR to be installed and available in the system PATH
def create_cbr(
    src_dir,
    archive_name="output.cbr",
    clear_directory = False,
    tool_preference: str = None,
    compression_level: int = 5
) -> None:
    """ creates a .cbr file (RAR archive) from images in `src_dir` probes for rar, 7z, winrar in PATH; or falls back to zip-with-.cbr
        Args:
            src_dir: directory containing images to archive
            archive_name: name of the output archive file (default: "output.cbr")
            clear_directory: if True, deletes all files in `src_dir` after creating the archive
            tool_preference: 'rar'|'7z'|'winrar'
            compression_level: 0-9 (mapped to tool syntax)
    """
    cbr_path = os.path.join(src_dir, "..", archive_name)
    image_files = _get_image_filenames(src_dir)
    tool = _get_supported_tool(tool_preference)
    try:
        if tool in ARCHIVE_TOOLS:
            if tool in ('rar', 'winrar'):
                cmd = [tool, 'a', '-ep', f"-m{compression_level}", cbr_path] + image_files
            elif tool == '7z':
                cmd = ['7z', 'a', f"-tRAR", f"-m0=on", f"-mx={compression_level}", cbr_path] + image_files
            subprocess.run(cmd, cwd=src_dir, check=True)
            print(f"CBR created with {tool}: {cbr_path}")
        else: # fallback option: zip and rename
            print("No RAR tool found; falling back to CBZ-style format (ZIP format) with RAR extension...")
            create_cbz(src_dir, archive_name, clear_directory, compression_level, as_fallback=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to create .cbr (requires 'rar' be installed and available in PATH).")
        raise e
    if clear_directory:
        _cleanup_src_directory(src_dir)


def create_pdf_from_pages(
    src_dir,
    archive_name="output.pdf",
    dpi=300,
    compression_level: int = 1,  # 0-9, where 0 is no compression and 9 is maximum compression
    clear_directory = False,
    auto_scale: bool = True,
    target_width: int = None
) -> None:
    """ Build a PDF from image pages in `src_dir`. Optionally auto-scale to target_width px """
    pages: List[Image.Image] = []
    for filename in _get_image_filenames(src_dir):
        img_path = os.path.join(src_dir, filename)
        img = Image.open(img_path).convert("RGB")
        if auto_scale and target_width:
            img = rescale_image(img, target_width)
        pages.append(img)
    if not pages:
        print("No pages for PDF.")
        return
    try:
        pdf_path = os.path.abspath(os.path.join(src_dir, "..", archive_name))
        pdf_quality = int(10 * (10 - compression_level))  # map compression level 0-9 to quality level 100-10
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:], dpi=(dpi, dpi), quality=pdf_quality) #resolution=dpi)
        print(f"PDF created: {pdf_path}")
    except Exception as e:
        print(f"ERROR: Failed to create PDF. {e}")
        return
    if clear_directory:
        _cleanup_src_directory(src_dir)


def _cleanup_src_directory(src_dir, remove_empty: bool = True):
    """ delete all files in the output directory after creating the archive - called optionally by the archive creation functions """
    assert os.path.isdir(src_dir), f"Source directory {src_dir} does not exist or is not a directory."
    print("Clearing image source directory...")
    all_files = _get_image_filenames(src_dir)
    for filename in tqdm(all_files, desc="Deleting files"):
        file_path = os.path.join(src_dir, filename)
        try:
            # check if the file exists before trying to delete it or remove recursively if it's a subdirectory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    # remove the source directory itself if empty (avoids pruning the whole tree from the get go to avoid deleting non-image files)
    if remove_empty and not os.listdir(src_dir):
        print(f"Removing empty source directory: {src_dir}")
        os.rmdir(src_dir)