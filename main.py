import os
#from typing import List, Generator, Dict, Union
import argparse
from dataclasses import dataclass
# local imports
import utils
from manhwa_normalizer import ImageSegmentGenerator, SegmentStitcher
from document_creator import create_cbz, create_cbr, create_pdf_from_pages


@dataclass
class ParserArguments:
    input_dir: str
    output_dir: str
    min_height: int
    archive: str = None
    cleanup: bool = False
    page_size: str = "letter"
    dpi: int = 300

    def get_page_height(self) -> int:
        """ return target image height in pixels for a given page size """
        return utils.get_page_height(self.page_size, self.dpi)

    @staticmethod
    def from_args(args) -> 'ParserArguments':
        return ParserArguments(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_height=args.min_height,
            archive=args.archive,
            cleanup=args.cleanup,
            page_size=args.page_size,
            dpi=args.dpi
        )

def parse_input_args() -> ParserArguments:
    """ Parse command line arguments and return them as a ParserArguments object """
    parser = argparse.ArgumentParser(description="Convert a folder of images into formatted .cbr pages.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save processed images")
    parser.add_argument("--min_height", type=int, default=1600, help="Minimum height for stacked images")
    # might want to add a check to make clear to the user that cleanup is only allowed when archiving
    parser.add_argument("--archive", choices=["cbz", "cbr", "pdf"], help="Optionally archive the output as CBZ or CBR")
    parser.add_argument("--cleanup", action="store_true", help="Delete output images after archiving")
    parser.add_argument("--page_size", choices=["letter", "a4"], default="letter", help=f"Optional page size for output images - supports 'letter' or 'a4'")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF output")
    # TODO: add a range for the gap size allowed between panels
    args = parser.parse_args()
    return ParserArguments.from_args(args)




def split_and_stack_images(segment_stream: ImageSegmentGenerator, args: ParserArguments):
    """ splits and stacks images for paginated output with a maximum height determined by the page size that is reached through padding and trimming """
    stitcher = SegmentStitcher()
    if args.archive == "pdf":
        page_height = args.get_page_height()
        # TODO: add arguments for the gap size range allowed between panels
        accumulator = stitcher.paginate_segments(segment_stream, page_height)
    else:
        accumulator = stitcher.accumulate_segments(segment_stream, args.min_height)
    # passes a generator to save_new_images to save memory
    utils.save_new_images(accumulator, args.output_dir)



def create_archive(args: ParserArguments):
    archive_name = f"{os.path.basename(args.output_dir)}.{args.archive}"
    if args.archive == "pdf":
        create_pdf_from_pages(args.output_dir, archive_name, dpi=args.dpi, clear_directory=args.cleanup)
    if args.archive == "cbz":
        create_cbz(args.output_dir, archive_name, clear_directory=args.cleanup)
    elif args.archive == "cbr":
        create_cbr(args.output_dir, archive_name, clear_directory=args.cleanup)



def main():
    #! Not actually sure if webp is supported by skimage - check later
    supported_filetypes = ('png', 'jpg', 'jpeg', 'bmp', 'webp','gif')
    parser_args = parse_input_args()
    #print(parser_args)
    image_files = sorted([f for f in os.listdir(parser_args.input_dir) if f.lower().endswith(supported_filetypes)])
    if not image_files:
        print("No images found.")
        return
    image_files = utils.filter_duplicate_files(image_files, supported_filetypes)
    dimensions = utils.get_image_dimensions(image_files, parser_args.input_dir)
    target_width = utils.mode_width(dimensions)
    print(f"Target width: {target_width}")
    print("Splitting and stacking images...")
    # object's __iter__ method acts as the generator
    segment_stream = ImageSegmentGenerator(image_files, parser_args.input_dir, target_width)
    split_and_stack_images(segment_stream, parser_args)
    # create the archive (or document) if requested
    if parser_args.archive:
        create_archive(parser_args)



if __name__ == "__main__":
    main()