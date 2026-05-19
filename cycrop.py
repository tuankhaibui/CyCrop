#!/usr/bin/env python3
"""
╔═══════════════════════════════════════╗
║          🔴 CyCrop 👁️                ║
║  Extract figures with Cyclops vision  ║
║                                       ║
║  ✨ TEXT-BASED CAPTION DETECTION:     ║
║  • Reads real PDF text + coordinates  ║
║  • Matches "Figure N:" / "Fig. N"     ║
║  • Column-aware crop of image ABOVE   ║
╚═══════════════════════════════════════╝

Usage: python cycrop.py --input <input_dir> --output <output_dir> [options]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional
import logging

try:
    import cv2
    import numpy as np
except ImportError:
    print("Installing packages...")
    import os
    os.system("pip install opencv-python numpy pillow pymupdf --break-system-packages")
    import cv2
    import numpy as np

try:
    import fitz
except ImportError:
    import os
    os.system("pip install pymupdf --break-system-packages")
    import fitz

from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Matches a *caption* line: starts with Fig / Fig. / Figure, a number,
# then a separator (":" "." "-" en/em dash). The separator requirement is
# what rejects in-text references like "Figure 1 shows ...".
CAPTION_RE = re.compile(
    r'^\s*(?:Fig(?:ure)?)\.?\s*(\d+)\s*[:.\-‒–—]',
    re.IGNORECASE,
)
# Used to skip other captions (figure/table) when looking for the text
# barrier that marks the top of a figure.
ANY_CAPTION_RE = re.compile(r'^\s*(?:Fig(?:ure)?|Table|Tab)\.?\s*\d+\s*[:.\-‒–—]',
                            re.IGNORECASE)


class CyCrop:
    def __init__(self, margin: int = 10, min_size: int = 50,
                 zoom: float = 4.0, min_text_chars: int = 25,
                 output_dir: str = None):
        self.margin = margin
        self.min_size = min_size
        self.zoom = zoom
        # A text block must have at least this many characters to count as a
        # "barrier" (body text / heading) that bounds the top of a figure.
        # Keeps short figure-internal labels (axis titles) from cutting the
        # figure short.
        self.min_text_chars = min_text_chars
        self.output_dir = Path(output_dir) if output_dir else Path("./cropped_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'files_processed': 0, 'images_found': 0,
                      'images_cropped': 0, 'errors': 0}

    def process_file(self, file_path: str) -> int:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            self.stats['errors'] += 1
            return 0

        logger.info(f"Processing: {file_path.name}")

        try:
            if file_path.suffix.lower() == '.pdf':
                count = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg',
                                              '.bmp', '.webp', '.tiff']:
                count = self._process_image(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return 0

            self.stats['files_processed'] += 1
            return count

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            self.stats['errors'] += 1
            return 0

    # ------------------------------------------------------------------ PDF

    def _process_pdf(self, file_path: Path) -> int:
        """Render each page, then crop the figure above every real caption."""
        try:
            pdf = fitz.open(file_path)
        except Exception as e:
            logger.error(f"PDF error: {str(e)}")
            return 0

        count = 0
        seen = set()  # (page, fig_number) -> avoid duplicates

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            page_rect = page.rect

            # Word-level lines: block bboxes are inflated/quirky on the
            # top & bottom edges and would clip or leak text into figures.
            lines = self._get_lines(page)

            captions = self._find_captions(lines)
            if not captions:
                logger.debug(f"  Page {page_num + 1}: no figure captions")
                continue

            # Render only pages that actually have captions.
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            logger.info(f"  Page {page_num + 1}: {len(captions)} caption(s) "
                        f"-> {[c['number'] for c in captions]}")

            for cap in captions:
                key = (page_num + 1, cap['number'])
                if key in seen:
                    continue
                seen.add(key)

                region = self._crop_figure(img, cap, lines, page_rect)
                if region is None:
                    logger.warning(f"  Could not extract Figure {cap['number']} "
                                    f"on page {page_num + 1}")
                    continue

                out = (self.output_dir /
                       f"{file_path.stem}_page{page_num + 1}_fig{cap['number']}.png")
                cv2.imwrite(str(out), region)
                self.stats['images_cropped'] += 1
                count += 1
                logger.info(f"  ✓ Saved: {out.name} "
                            f"({region.shape[1]}x{region.shape[0]}px)")

        pdf.close()
        self.stats['images_found'] += count
        return count

    @staticmethod
    def _get_lines(page) -> List[dict]:
        """
        Group words into text lines with tight glyph-level bounding boxes.

        Block bboxes from get_text("blocks") are inflated on the top edge
        (first-line leading) and underestimate the bottom edge, which clips
        a figure's bottom labels or leaks body text into its top. Word boxes
        are tight to glyphs, so everything is keyed off these.
        """
        # words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        grouped = {}
        for w in page.get_text("words"):
            grouped.setdefault((w[5], w[6]), []).append(w)

        lines = []
        for words in grouped.values():
            words.sort(key=lambda w: w[7])
            lines.append({
                'text': " ".join(w[4] for w in words),
                'x0': min(w[0] for w in words),
                'y0': min(w[1] for w in words),
                'x1': max(w[2] for w in words),
                'y1': max(w[3] for w in words),
            })
        for ln in lines:
            ln['cx'] = (ln['x0'] + ln['x1']) / 2.0
        return lines

    def _find_captions(self, lines: List[dict]) -> List[dict]:
        """Return real figure captions with word-level PDF-point coords."""
        captions = []
        for ln in lines:
            m = CAPTION_RE.match(ln['text'])
            if not m:
                continue
            captions.append({
                'number': int(m.group(1)),
                'x0': ln['x0'], 'y0': ln['y0'],  # y0 = true top of caption
                'x1': ln['x1'], 'y1': ln['y1'],
                'cx': ln['cx'],
            })
        return sorted(captions, key=lambda c: (c['y0'], c['x0']))

    def _crop_figure(self, img: np.ndarray, cap: dict,
                     lines: List[dict], page_rect) -> Optional[np.ndarray]:
        """
        Crop the figure that belongs to `cap`.

        1. Figure sits ABOVE the caption, inside the same text column.
        2. Top boundary = bottom of the nearest substantial text line
           above the caption in that column (body paragraph / heading).
        3. Within that band, trim to the tight bounding box of non-white
           pixels and add a margin.
        """
        page_w = page_rect.width
        page_mid = page_w / 2.0
        cap_left_col = cap['cx'] < page_mid

        def is_spanning(x0, x1):
            # Title / running header / doi line spanning both columns.
            return (x0 < page_mid < x1) and (x1 - x0) > 0.6 * page_w

        def is_barrier(stripped):
            # Body paragraph, OR a short ALL-CAPS section heading. Short
            # mixed-case blocks (figure-internal axis labels) are NOT barriers.
            if len(stripped) >= self.min_text_chars:
                return True
            letters = [c for c in stripped if c.isalpha()]
            if len(stripped) >= 4 and letters and \
                    sum(c.isupper() for c in letters) / len(letters) > 0.7:
                return True
            return False

        # ---- horizontal column extent (body-like, single-column lines) ----
        col_x0, col_x1 = None, None
        for ln in lines:
            s = ln['text'].strip()
            if len(s) < 8 or is_spanning(ln['x0'], ln['x1']):
                continue
            if (ln['x1'] - ln['x0']) < 0.20 * page_w:  # margin watermark / labels
                continue
            if (ln['cx'] < page_mid) != cap_left_col:
                continue
            col_x0 = ln['x0'] if col_x0 is None else min(col_x0, ln['x0'])
            col_x1 = ln['x1'] if col_x1 is None else max(col_x1, ln['x1'])
        if col_x0 is None:
            col_x0, col_x1 = cap['x0'], cap['x1']

        # ---- top boundary: bottom of the body paragraph above caption ----
        # Group in-column lines above the caption into paragraphs (vertically
        # adjacent lines). A paragraph is a "barrier" if any of its lines is
        # long body text / an ALL-CAPS heading. We take the paragraph's full
        # bottom, so a short final line ("losses.") cannot leak into the crop,
        # while sparse figure-internal labels (their own non-barrier cluster)
        # are ignored.
        cand = []
        for ln in lines:
            s = ln['text'].strip()
            if not s or ln['y1'] > cap['y0'] - 2:
                continue
            if ANY_CAPTION_RE.match(s):
                continue
            if is_spanning(ln['x0'], ln['x1']) or \
                    ((ln['cx'] < page_mid) == cap_left_col):
                cand.append(ln)
        cand.sort(key=lambda l: l['y0'])

        heights = sorted(l['y1'] - l['y0'] for l in cand)
        lh = heights[len(heights) // 2] if heights else 10.0
        gap_thr = max(5.0, 0.8 * lh)

        top_pt = float(page_rect.y0) + 36.0  # default: below page header
        cluster = []

        def flush(cl):
            nonlocal top_pt
            if cl and any(is_barrier(l['text'].strip()) for l in cl):
                top_pt = max(top_pt, max(l['y1'] for l in cl))

        for ln in cand:
            if cluster:
                cl_bottom = max(l['y1'] for l in cluster)
                if (ln['y0'] - cl_bottom) > gap_thr:
                    flush(cluster)
                    cluster = []
            cluster.append(ln)
        flush(cluster)

        bottom_pt = cap['y0'] - 2.0
        if bottom_pt <= top_pt:
            return None

        # ---- points -> pixels ----
        z = self.zoom
        H, W = img.shape[:2]
        # Hard clamp to the caption's page half: a figure never legitimately
        # crosses the column gutter, so this prevents any cross-column bleed.
        gutter = page_mid * z
        y_top = max(0, int(round(top_pt * z)))
        y_bot = min(H, int(round(bottom_pt * z)))
        x_lo = max(0, int(round(col_x0 * z)))
        x_hi = min(W, int(round(col_x1 * z)))
        if cap_left_col:
            x_hi = min(x_hi, int(round(gutter)))
        else:
            x_lo = max(x_lo, int(round(gutter)))
        if y_bot - y_top < 5 or x_hi - x_lo < 5:
            return None

        band = img[y_top:y_bot, x_lo:x_hi]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        if np.count_nonzero(binary) == 0:
            return None

        idx = np.where(np.any(binary, axis=1))[0]
        if len(idx) == 0:
            return None

        # Split content rows into runs (content separated by whitespace).
        brk = np.where(np.diff(idx) > 1)[0]
        run_start = np.r_[idx[0], idx[brk + 1]]
        run_end = np.r_[idx[brk], idx[-1]]  # inclusive

        # Drop a thin strip at the very top that is separated from the figure
        # body by clear whitespace: glyph descenders of the last body line
        # bleed ~1px below the word box. A real figure top is thick/connected.
        strip_max = max(3, int(round(0.25 * lh * z)))
        gap_min = max(4, int(round(0.15 * lh * z)))
        s = 0
        while s < len(run_start) - 1:
            h = run_end[s] - run_start[s] + 1
            gap = run_start[s + 1] - run_end[s] - 1
            if h <= strip_max and gap >= gap_min:
                s += 1
            else:
                break

        top_row, bot_row = run_start[s], run_end[-1]
        sub = binary[top_row:bot_row + 1, :]
        cols = np.where(np.any(sub, axis=0))[0]
        if len(cols) == 0:
            return None

        t = max(0, top_row - self.margin)
        b = min(band.shape[0], bot_row + self.margin)
        l = max(0, cols[0] - self.margin)
        r = min(band.shape[1], cols[-1] + self.margin)

        if (b - t) < self.min_size or (r - l) < self.min_size:
            return None

        return band[t:b, l:r]

    # ---------------------------------------------------------------- image

    def _process_image(self, file_path: Path) -> int:
        """Single image: needs OCR to locate captions. Falls back gracefully."""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                pil = Image.open(file_path)
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Image error: {str(e)}")
            return 0

        try:
            import pytesseract
        except ImportError:
            logger.warning(
                f"  Skipping {file_path.name}: caption detection on raster "
                f"images needs OCR. Install with: pip install pytesseract "
                f"(and the tesseract binary).")
            return 0

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        count = 0
        n = len(data['text'])
        i = 0
        while i < n:
            word = (data['text'][i] or "").strip()
            nxt = (data['text'][i + 1] if i + 1 < n else "").strip()
            joined = f"{word} {nxt}".strip()
            m = CAPTION_RE.match(joined) or CAPTION_RE.match(word)
            if m and data['conf'][i] != '-1':
                cap_y = data['top'][i]
                region = self._crop_image_above(img, cap_y)
                if region is not None:
                    out = (self.output_dir /
                           f"{file_path.stem}_fig{m.group(1)}.png")
                    cv2.imwrite(str(out), region)
                    self.stats['images_cropped'] += 1
                    count += 1
                    logger.info(f"  ✓ Saved: {out.name}")
            i += 1

        self.stats['images_found'] += count
        return count

    def _crop_image_above(self, img: np.ndarray, cap_y: int) -> Optional[np.ndarray]:
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        region = binary[0:cap_y, :]
        if region.size == 0 or np.count_nonzero(region) == 0:
            return None
        rows = np.where(np.any(region, axis=1))[0]
        cols = np.where(np.any(region, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        t = rows[0]
        b = min(rows[-1] + self.margin, cap_y)
        l = max(0, cols[0] - self.margin)
        r = min(width, cols[-1] + self.margin)
        if (b - t) < self.min_size or (r - l) < self.min_size:
            return None
        return img[t:b, l:r]

    # ------------------------------------------------------------ batch/cli

    def process_directory(self, input_dir: str) -> dict:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return self.stats

        exts = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
        files = [f for f in input_path.rglob('*') if f.suffix.lower() in exts]
        if not files:
            logger.warning("No supported files found")
            return self.stats

        logger.info(f"Found {len(files)} files")
        for file_path in sorted(files):
            self.process_file(file_path)
        return self.stats

    def print_stats(self):
        print("\n" + "=" * 60)
        print("🔴 CYCROP - EXTRACTION COMPLETE 👁️")
        print("=" * 60)
        print(f"Files processed:     {self.stats['files_processed']}")
        print(f"Figures extracted:   {self.stats['images_found']}")
        print(f"Files saved:         {self.stats['images_cropped']}")
        print(f"Errors:              {self.stats['errors']}")
        print(f"Output:              {self.output_dir}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='🔴 CyCrop - Extract figures by reading real captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cycrop.py --input paper.pdf --output ./crops
  python cycrop.py --input ./papers --output ./crops --margin 15 --zoom 6
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', default='./cropped_images', help='Output directory')
    parser.add_argument('--margin', '-m', type=int, default=10, help='Pixel margin (default: 10)')
    parser.add_argument('--min-size', '-s', type=int, default=50, help='Min crop size px (default: 50)')
    parser.add_argument('--zoom', '-z', type=float, default=4.0, help='Render zoom factor (default: 4)')
    parser.add_argument('--min-text-chars', type=int, default=25,
                        help='Min chars for a text block to bound a figure top (default: 25)')
    args = parser.parse_args()

    cropper = CyCrop(margin=args.margin, min_size=args.min_size, zoom=args.zoom,
                     min_text_chars=args.min_text_chars, output_dir=args.output)

    input_path = Path(args.input)
    if input_path.is_file():
        cropper.process_file(str(input_path))
    elif input_path.is_dir():
        cropper.process_directory(str(input_path))
    else:
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    cropper.print_stats()


if __name__ == '__main__':
    main()
