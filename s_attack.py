import sys
from PIL import Image

def process_pixel(value, bits, mode):
    mask = (1 << bits) - 1           # напр. bits=3 → mask=0b111
    low = value & mask               # младшие биты пикселя

    if mode == '1':
        new_low = mask               # все биты → 1
    elif mode == '0':
        new_low = 0                  # все биты → 0
    else:  # reverse
        new_low = low ^ mask         # инвертировать только указанные биты

    return (value & ~mask) | new_low # заменить младшие биты

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <image> <bits 1-8> [1|0|reverse]")
        return

    img_path = sys.argv[1]
    bits = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "reverse"

    img = Image.open(img_path).convert("RGB")
    pix = img.load()

    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = pix[x, y]
            pix[x, y] = (
                process_pixel(r, bits, mode),
                process_pixel(g, bits, mode),
                process_pixel(b, bits, mode)
            )

    out = f"out_{bits}_{mode}{'.' + img.format.lower() if img.format else ''}"
    out = f"out_{bits}_{mode}.bmp"

    img.save(out, format=img.format)
    print("Saved:", out)

if __name__ == "__main__":
    main()
