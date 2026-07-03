#!/usr/bin/env python3
"""Extract figures from PDF papers as images for blog embedding.

Uses PyMuPDF (fitz) to render PDF pages or page regions as PNG/JPG.
Output goes to static/images/paperreading/<slug>/.

Usage examples:
  # Render full page 5 at 200 DPI
  python3 scripts/extract_pdf_figures.py assets/pdf/2025-cheops-llm.pdf \
      --page 5 --name fig-iops-comparison --slug cheops-llm

  # Crop a specific region (coordinates in PDF points, origin top-left)
  python3 scripts/extract_pdf_figures.py assets/pdf/2025-cheops-llm.pdf \
      --page 3 --bbox 50 100 550 400 --name fig-arch --slug cheops-llm --dpi 300

  # Render pages 3-5 (each page becomes a separate file)
  python3 scripts/extract_pdf_figures.py assets/pdf/2025-cheops-llm.pdf \
      --pages 3-5 --name fig-results --slug cheops-llm

  # List embedded images and their positions (to find crop coordinates)
  python3 scripts/extract_pdf_figures.py assets/pdf/2025-cheops-llm.pdf --list
"""
import argparse
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    print("Error: PyMuPDF not installed. Run: pip3 install PyMuPDF", file=sys.stderr)
    sys.exit(1)


def render_page(pdf_path, page_num, bbox, output_path, dpi=200, fmt='png'):
    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        print(f"Error: page {page_num} out of range (1-{len(doc)})", file=sys.stderr)
        doc.close()
        sys.exit(1)
    page = doc[page_num - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    if bbox:
        clip = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=mat, clip=clip)
    else:
        pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()


def list_images(pdf_path):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        page_rect = page.rect
        print(f"--- Page {i + 1} ({page_rect.width:.0f}x{page_rect.height:.0f} pt) ---")
        imgs = page.get_images(full=True)
        if not imgs:
            print("  (no embedded raster images — likely vector/diagram)")
            continue
        for img in imgs:
            xref = img[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                print(f"  x0={r.x0:.0f} y0={r.y0:.0f} x1={r.x1:.0f} y1={r.y1:.0f} "
                      f"({r.width:.0f}x{r.height:.0f} pt) xref={xref}")
    doc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract figures from PDF for blog embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('pdf', nargs='?', help='Path to PDF file')
    parser.add_argument('--page', type=int, help='Page number (1-indexed)')
    parser.add_argument('--pages', help='Page range, e.g. 3-5')
    parser.add_argument('--bbox', nargs=4, type=float,
                        metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='Crop box in PDF points (origin top-left)')
    parser.add_argument('--name', help='Output filename (without extension)')
    parser.add_argument('--slug', help='Paper slug for output directory, e.g. cheops-llm')
    parser.add_argument('--dpi', type=int, default=200, help='Output DPI (default: 200)')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--list', action='store_true',
                        help='List embedded images and their positions')
    args = parser.parse_args()

    if args.list:
        if not args.pdf:
            parser.error('--list requires a PDF path')
        list_images(args.pdf)
        return

    if not args.pdf:
        parser.error('a PDF path is required')
    if not args.name:
        parser.error('--name is required')
    if not args.slug:
        parser.error('--slug is required')
    if not args.page and not args.pages:
        parser.error('either --page or --pages is required')

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path('static/images/paperreading') / args.slug
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pages:
        start, end = map(int, args.pages.split('-'))
        for p in range(start, end + 1):
            out = output_dir / f'{args.name}-p{p}.{args.format}'
            render_page(str(pdf_path), p, args.bbox, str(out), args.dpi, args.format)
            print(f'Saved: {out}')
    else:
        out = output_dir / f'{args.name}.{args.format}'
        render_page(str(pdf_path), args.page, args.bbox, str(out), args.dpi, args.format)
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
