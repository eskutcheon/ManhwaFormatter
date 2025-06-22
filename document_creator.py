import os
from tqdm import tqdm
import zipfile
import shutil
import subprocess
from PIL import Image


def create_cbz(src_dir, archive_name = "output.cbz", clear_directory = False):
    """ creates a .cbz file (ZIP archive) from images in the output directory """
    cbz_path = os.path.join(src_dir, "..", archive_name)
    with zipfile.ZipFile(cbz_path, 'w') as zipf:
        for filename in sorted(os.listdir(src_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                zipf.write(
                    os.path.join(src_dir, filename),
                    arcname=filename
                )
    print(f"CBZ file created: {cbz_path}")
    if clear_directory:
        cleanup_src_directory(src_dir)


#! FIXME: doesn't work on Windows - need to test on Linux
    #! requires WinRAR to be installed and available in the system PATH
def create_cbr(src_dir, archive_name="output.cbr", clear_directory = False):
    """ creates a .cbr file (RAR archive) from images using WinRAR or system 'rar' if available """
    cbr_path = os.path.join(src_dir, "..", archive_name)
    image_files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    try:
        # Try using system 'rar' command
        command = ['rar', 'a', '-ep', cbr_path] + image_files
        subprocess.run(command, cwd=src_dir, check=True)
        print(f"CBR file created: {cbr_path}")
    except Exception as e:
        print("Failed to create .cbr (requires 'rar' be installed and available in PATH).")
        raise e
    if clear_directory:
        cleanup_src_directory(src_dir)


def create_pdf_from_pages(src_dir, archive_name="output.pdf", dpi=300, clear_directory = False):
    pages = []
    for filename in sorted(os.listdir(src_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(src_dir, filename)
            img = Image.open(img_path).convert("RGB")
            pages.append(img)
    if pages:
        pdf_path = os.path.join(src_dir, "..", archive_name)
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=dpi)
        print(f"PDF created: {pdf_path}")
    if clear_directory:
        cleanup_src_directory(src_dir)



#! WARNING: written by AI - untested
def cleanup_src_directory(src_dir):
    """ delete all files in the output directory after creating the archive - called optionally by the archive creation functions """
    print("Clearing image source directory...")
    for filename in tqdm(os.listdir(src_dir), desc="Deleting files"):
        file_path = os.path.join(src_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")