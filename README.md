# ManhwaFormatter: Webtoon Image Formatter and Archive Generator

ManhwaFormatter is a Python tool for processing vertical webtoon-style comics (like manhwa, manhua, and Korean webcomics) into images which fixes poor crops and cut-off panel content in high-information regions. It also allows for the creation of structured document formats such as .cbz, .cbr, and .pdf using these reconstituted images. It intelligently resizes, segments, and recomposes webtoon panels for optimized viewing across various platforms - from comic book readers to e-readers and print-ready PDFs.


This started as a simple script for my own use with scraped manhwa that were really poorly formatted. I've extended the scripts since then to more intelligently create paginated images to form pdf files and other paginated document types.


## Features
- Auto-resize images to a uniform width using the most common value across a folder
  - Skips resizing landscape-oriented title pages or special pages
- Intelligent vertical segmentation by detecting blank horizontal regions with low variance
  - Prevents cutting through panels or dialogue bubbles
- Supports vertical stacking into long-scroll .cbz or .cbr comics
- New paginated mode for .pdf output optimized for page sizes like Letter or A4
- Adaptive spacing logic:
  - Reduces or expands blank space between panels
  - Matches padding color with detected panel gaps
- Handles duplicate images of different file extensions by default
- Efficient streamed batch processing
- Fast dimension detection using image metadata parsing


## Use Cases

| Use Case                  | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| **Mobile readers**        | Create .cbz or .cbr files optimized for vertical scrolling     |
| E-reader export           | Convert webcomics to paginated .pdf with letter/A4 page sizes  |
| Print layout              | Generate standardized PDFs for print-ready formatting          |
| Localization / fansubbing | Re-segment and recompose webcomics before translation overlays |


## Example Results

The following examples show the problem of poor image cuts on one of the most popular manhwa of all time, "Solo Leveling" hosted on one of the most popular manga/manhwa/manhua readers of all time, MangaDex. Even with the intense popularity of both, we see that proper formatting of the images may be an afterthought for many, as they instead rely on features of the comic reader to support continuous vertical read modes. The images below are ordered left-to-right in their original read order.

![OLD IMAGES](assets/examples/old_img_grid.png)

While the images above illustrate the problem, note that we can expect much worse readability when the cuts are made in the middle of a speech bubble or region of very dense comic panels. Below, we see the fixed version that ensures cuts are made in whitespace.

![NEW IMAGES](assets/examples/new_img_grid.png)


This particular trial was done with a minimum height of 3200, using the original image heights, though this can be set arbitrarily with the arguments shown in the [Usage section](##Usage).



## Installation
```bash
git clone https://github.com/eskutcheon/ManhwaFormatter.git
cd ManhwaFormatter
pip install -r requirements.txt
```

## Usage
```bash
python main.py input_dir output_dir [options]
```
#### Arguments

| Option         | Description                                                                          |
| -------------- | ------------------------------------------------------------------------------------ |
| `--archive`    | One of ("cbz", "cbr", "pdf")                                                         |
| `--cleanup`    | Delete all intermediate images after archiving (does nothing if `--archive` is None) |
| `--min_height` | Minimum height before a new (non-paginated) stacked page is created - Default: 1600  |
| `--page_size`  | Paginated version's page size from ("letter", "a4") - Default: "letter"              |
| `--dpi`        | DPI used for PDF output - Default: 300                                               |

#### Example
```bash
python main.py ./chapter_001 ./output --min_height 3200
python main.py ./chapter_001 ./output --archive cbz --cleanup
python main.py ./chapter_002 ./output --archive pdf --page_size a4
```


## Roadmap
- [ ] EPUB support
- [ ] GUI wrapper
