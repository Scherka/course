import sys
import os
from PIL import Image

def process_pixel(value, bits, mode):
    mask = (1 << bits) - 1
    low = value & mask

    if mode == '1':
        new_low = mask
    elif mode == '0':
        new_low = 0
    else:  
        new_low = low ^ mask

    return (value & ~mask) | new_low

def process_image(img, bits, mode):
    pix = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pix[x, y]
            pix[x, y] = (
                process_pixel(r, bits, mode),
                process_pixel(g, bits, mode),
                process_pixel(b, bits, mode),
            )

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image>")
        return

    img_path = sys.argv[1]
    img = Image.open(img_path)
    original_format = img.format
    base_dir = os.path.dirname(img_path)
    name, ext = os.path.splitext(os.path.basename(img_path))
    ext = ext.lower()

    modes = ["0", "1", "reverse"]

    for bits in range(1, 9):
        for mode in modes:
            img_copy = img.copy().convert("RGB")

            process_image(img_copy, bits, mode)

            out_name = f"{name}_bits_{bits}_{mode}{ext}"
            out_path = os.path.join(base_dir, out_name)

            img_copy.save(out_path, format=original_format)
            print("Saved:", out_path)

if __name__ == "__main__":
    main()
