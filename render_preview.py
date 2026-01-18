#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import datetime as dt
from PIL import Image, ImageDraw, ImageFont

W, H = 800, 480

ASSETS_BASE = "/home/alex/E-Paper-tibber-Preisanzeige/assets"
SCENES_DIR = os.path.join(ASSETS_BASE, "scenes")
ICONS_DIR  = os.path.join(ASSETS_BASE, "icons")

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def ordered_dither_bayer8(img_l):
    # Bayer 8x8 thresholds 0..63
    b = [
        [0,32,8,40,2,34,10,42],
        [48,16,56,24,50,18,58,26],
        [12,44,4,36,14,46,6,38],
        [60,28,52,20,62,30,54,22],
        [3,35,11,43,1,33,9,41],
        [51,19,59,27,49,17,57,25],
        [15,47,7,39,13,45,5,37],
        [63,31,55,23,61,29,53,21],
    ]
    if img_l.mode != "L":
        img_l = img_l.convert("L")
    w, h = img_l.size
    out = Image.new("1", (w, h), 1)
    src = img_l.load()
    dst = out.load()
    for y in range(h):
        row = b[y % 8]
        for x in range(w):
            v = src[x, y]
            thr = int((row[x % 8] + 0.5) * (255.0 / 64.0))
            dst[x, y] = 0 if v < thr else 255
    return out

def draw_card(draw, x, y, w, h, title, font_title, font_body):
    draw.rectangle((x, y, x+w, y+h), outline=40, width=2, fill=245)
    tw, th = text_size(draw, title, font_title)
    draw.text((x+12, y+8), title, font=font_title, fill=30)
    # simple rows
    rows = ["Line 1", "Line 2", "Line 3"]
    yy = y + 8 + th + 6
    for r in rows:
        draw.text((x+12, yy), r, font=font_body, fill=40)
        yy += 16

def draw_chart_mock(draw, x0, y0, x1, y1, font_tiny):
    # background chart frame
    draw.rectangle((x0, y0, x1, y1), outline=40, width=2, fill=255)

    mid = (x0 + x1) // 2
    draw.line((mid, y0, mid, y1), fill=60, width=2)

    # sample curves (price step)
    import math
    def panel(px0, px1, label):
        w = px1 - px0
        h = y1 - y0
        # fake area fill (PV)
        points = []
        for i in range(0, 49):
            t = i / 48.0
            xx = px0 + int(t * w)
            pv = (math.sin((t*math.pi)) ** 1.5) * 0.6
            yy = y1 - int(pv * h)
            points.append((xx, yy))
        poly = points + [(points[-1][0], y1), (points[0][0], y1)]
        draw.polygon(poly, fill=220)

        # fake consumption (darker)
        points2 = []
        for i in range(0, 49):
            t = i / 48.0
            xx = px0 + int(t * w)
            cons = 0.25 + 0.10*math.sin(6*t*math.pi)
            yy = y1 - int(cons * h)
            points2.append((xx, yy))
        poly2 = points2 + [(points2[-1][0], y1), (points2[0][0], y1)]
        draw.polygon(poly2, fill=200)

        # step price line
        lasty = None
        for i in range(0, 25):
            t = i / 24.0
            xx = px0 + int(t * w)
            price = 0.35 + 0.25*math.sin((t*2*math.pi) + 1.3)
            yy = y1 - int(price * h)
            if lasty is not None:
                draw.line((lastx, lasty, xx, lasty), fill=20, width=2)
                draw.line((xx, lasty, xx, yy), fill=20, width=2)
            lastx, lasty = xx, yy

        draw.text((px0+10, y1+10), label, font=font_tiny, fill=30)

    panel(x0, mid, "Links")
    panel(mid, x1, "Rechts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="", help="Scene PNG filename in assets/scenes, z.B. rain.png")
    ap.add_argument("--out", default="/home/alex/E-Paper-tibber-Preisanzeige/screen_sim.png")
    ap.add_argument("--dither", choices=["none", "bayer8", "floyd"], default="none")
    args = ap.parse_args()

    # Base canvas in grayscale
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    # Optional scene background
    if args.scene:
        scene_path = args.scene
        if not os.path.isabs(scene_path):
            scene_path = os.path.join(SCENES_DIR, args.scene)
        if os.path.exists(scene_path):
            scene = Image.open(scene_path).convert("L").resize((W, H), Image.LANCZOS)
            img.paste(scene, (0, 0))
        else:
            print(f"Scene not found: {scene_path}")

    # Fonts
    f_bold = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    f_reg  = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    f_tiny = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)

    # Header
    header_h = 40
    draw.rectangle((0, 0, W, header_h), fill=245)
    draw.line((0, header_h, W, header_h), fill=60, width=2)
    draw.text((12, 10), "TRMNL Preview", font=f_bold, fill=20)
    ts = dt.datetime.now().strftime("%H:%M  %d.%m.%Y")
    tw, _ = text_size(draw, ts, f_tiny)
    draw.text((W-12-tw, 12), ts, font=f_tiny, fill=40)

    # Cards top
    margin = 12
    top_y = header_h + margin
    card_h = 76
    card_w = (W - margin*3) // 2
    draw_card(draw, margin, top_y, card_w, card_h, "Wetter", f_reg, f_tiny)
    draw_card(draw, margin*2 + card_w, top_y, card_w, card_h, "EcoFlow", f_reg, f_tiny)

    # Info line
    info_y = top_y + card_h + 10
    draw.rectangle((margin, info_y, W-margin, info_y+28), outline=60, width=2, fill=250)
    draw.text((margin+10, info_y+6), "Preis jetzt: 0.28  |  Tief: 0.19 @ 03:00  |  Hoch: 0.41",
              font=f_tiny, fill=30)

    # Chart area
    chart_top = info_y + 42
    chart_bottom = H - 40
    chart_left = margin
    chart_right = W - margin
    draw_chart_mock(draw, chart_left, chart_top, chart_right, chart_bottom-20, f_tiny)

    # Footer
    draw.text((12, H-24), "Preview only (no EPD)", font=f_tiny, fill=60)

    # Export
    if args.dither == "none":
        img.save(args.out)
    elif args.dither == "floyd":
        img.convert("1", dither=Image.FLOYDSTEINBERG).save(args.out)
    else:  # bayer8
        ordered_dither_bayer8(img).save(args.out)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
