import os
#from typing import List, Generator, Dict, Union
# local imports
import utils
from manhwa_normalizer import ImageSegmentGenerator, SegmentStitcher
from document_creator import create_cbz, create_cbr, create_pdf_from_pages
from config import ParserArguments, parse_input_args



def split_and_stack_images(segment_stream: ImageSegmentGenerator, args: ParserArguments):
    """ splits and stacks images for paginated output with a maximum height determined by the page size that is reached through padding and trimming """
    cfg = args.stitcher_config
    stitcher = SegmentStitcher(cfg.min_gap, cfg.max_gap, cfg.resize_threshold_ratio)
    if args.archive_config.archive_type == "pdf":
        page_height = args.get_page_height()
        accumulator = stitcher.paginate_segments(segment_stream, page_height)
    else:
        accumulator = stitcher.accumulate_segments(segment_stream, args.stitcher_config.min_height)
    # passes a generator to save_new_images to save memory
    utils.save_new_images(accumulator, args.output_dir)



def create_archive(args: ParserArguments):
    cfg = args.archive_config
    archive_name = f"{os.path.basename(args.output_dir)}.{cfg.archive_type}"
    # archiving functions take many of the same arguments, so we construct common kwargs below
    kwargs = {
        "archive_name": archive_name,
        "clear_directory": cfg.cleanup,
        "compression_level": cfg.level,
    }
    match cfg.archive_type:
        case "pdf":
            create_pdf_from_pages(args.output_dir, dpi=cfg.dpi, auto_scale=cfg.auto_scale, **kwargs)
        case "cbz":
            create_cbz(args.output_dir, **kwargs)
        case "cbr":
            create_cbr(args.output_dir, tool_preference=cfg.tool, **kwargs)
        case _:
            print(f"ERROR: Unsupported archive type '{cfg.archive_type}'. Supported types are 'pdf', 'cbz', and 'cbr'.")



# TODO: consider adding a new run mode that spawns a GUI to set input arguments interactively and allows arranging images before stitching


def main():
    #! Not actually sure if webp is supported by skimage - check later
    supported_filetypes = ('png', 'jpg', 'jpeg', 'bmp', 'webp','gif')
    parser_args = parse_input_args()
    image_files = sorted([f for f in os.listdir(parser_args.input_dir) if f.lower().endswith(supported_filetypes)])
    if not image_files:
        print("No images found.")
        return
    image_files = utils.filter_duplicate_files(image_files, supported_filetypes)
    dimensions = utils.get_image_dimensions(image_files, parser_args.input_dir)
    # check if the target width is set, otherwise set it to the most frequent image width in the input directory
    target_width = parser_args.stitcher_config.target_width or utils.mode_width(dimensions)
    print("Splitting and stacking images...")
    # object's __iter__ method acts as the generator
    segment_stream = ImageSegmentGenerator(image_files, parser_args.input_dir, target_width, parser_args.stitcher_config.min_blank_run)
    split_and_stack_images(segment_stream, parser_args)
    # create the archive (or document) if requested
    if parser_args.archive_config.archive_type is not None:
        create_archive(parser_args)



if __name__ == "__main__":
    main()