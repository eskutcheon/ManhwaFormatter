
import argparse
from dataclasses import dataclass, field
from typing import Optional, Literal
# local imports
from utils import get_page_height


@dataclass
class ArchiveConfig:
    """ config options for CBZ/CBR/PDF archive creation """
    archive_type: Optional[Literal["cbz", "cbr", "pdf"]] = None,
    tool: Optional[Literal["rar", "7z", "winrar"]] = None      # Force 'rar', '7z', or 'winrar'
    level: int = 5                                             # Compression level (0-9)
    auto_scale: bool = False                                   # Whether to scale images uniformly before PDF creation
    cleanup: bool = False                                      # Whether to delete images after archiving
    page_size: Literal["letter", "a4"] = "letter"              # only used for PDF creation when used as an argument in ParserArguments.get_page_height()
    dpi: int = 300


@dataclass
class StitcherConfig:
    """ configuration for normalization and segment stitching """
    min_height: int = 1600              # Minimum height for stacked images
    target_width: Optional[int] = None  # Target width for resizing, if applicable
    min_gap: int = 5
    max_gap: int = 40
    resize_threshold_ratio: float = 0.05
    min_blank_run: int = 10         # Minimum contiguous blank rows for a split


@dataclass
class ParserArguments:
    input_dir: str
    output_dir: str
    # new config options from new dataclasses
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    stitcher_config: StitcherConfig = field(default_factory=StitcherConfig)

    def get_page_height(self) -> int:
        """ return target image height in pixels for a given page size """
        return get_page_height(self.archive_config.page_size, self.archive_config.dpi)

    @staticmethod
    def from_args(args) -> 'ParserArguments':
        archive_cfg = ArchiveConfig(
            archive_type=args.archive,
            tool=args.archive_tool,
            level=args.compression,
            auto_scale=args.auto_scale,
            cleanup=args.cleanup,
            page_size=args.page_size,
            dpi=args.dpi
        )
        stitcher_cfg = StitcherConfig(
            min_height=args.min_height,
            target_width=args.target_width,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            resize_threshold_ratio=args.resize_threshold,
            min_blank_run=args.min_blank_run
        )
        return ParserArguments(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            archive_config=archive_cfg,
            stitcher_config=stitcher_cfg
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
    parser.add_argument("--archive_tool", choices=["rar", "7z", "winrar"], help="Force use of this RAR creation tool (otherwise auto-detect)")
    parser.add_argument("--compression", type=int, default=0, help="Compression level 0-9 for CBR/CBZ (default: no compression)")
    parser.add_argument("--auto_scale", action="store_true", help="Scale images uniformly before PDF creation")
    parser.add_argument("--min_gap", type=int, default=5, help="Minimum gap size between panels in pixels")
    parser.add_argument("--max_gap", type=int, default=40, help="Maximum gap size between panels in pixels")
    #! FIXME: may not actually be a good choice of ratio - investigate later
    parser.add_argument("--resize_threshold", type=float, default=0.05, help="Threshold ratio for resizing images (default 0.05)")
    parser.add_argument("--min_blank_run", type=int, default=10, help="Minimum contiguous blank rows for a split (default 10)")
    parser.add_argument("--target_width", type=int, help="Target width for resizing images (optional)")
    args = parser.parse_args()
    return ParserArguments.from_args(args)