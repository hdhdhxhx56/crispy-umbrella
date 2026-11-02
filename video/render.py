# -*- coding: utf-8 -*-
import sys, math, wave, subprocess, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
from script import SCENES, CHARS

W, H, FPS = 1280, 720, 25
SR = 24000
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
f = lambda s, b=False: ImageFont.truetype(FONTB if b else FONT, s)
F_SUB, F_NAME, F_TITLE, F_BUL, F_BIG = f(25), f(19, True), f(30, True), f(23), f(52, True)
PAUSE = 0.45
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out.mp4'

# ---------------- audio timeline ----------------
def load(path):
    w = wave.open(path); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a

timeline = []   # (start_sec, end_sec, scene_idx, line_idx)
chunks = []
t = 0.0
INTRO, OUTRO = 4.0, 6.0
chunks.append(np.zeros(int(INTRO * SR), np.float32)); t = INTRO
for si, sc in enumerate(SCENES):
    if si > 0:
        chunks.append(np.zeros(int(1.2 * SR), np.float32)); t += 1.2   # scene transition
    for li, ln in enumerate(sc['lines']):
        a = load(f'a/{si}_{li}.wav')
        timeline.append((t, t + len(a) / SR, si, li))
        chunks.append(a); t += len(a) / SR
        chunks.append(np.zeros(int(PAUSE * SR), np.float32)); t += PAUSE
chunks.append(np.zeros(int(OUTRO * SR), np.float32)); t += OUTRO
audio = np.concatenate(chunks)
TOTAL = len(audio) / SR
pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
with wave.open('narration.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
# per-frame loudness for lip sync
NF = int(TOTAL * FPS) + 1
hop = SR // FPS
env = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(audio) else 0 for i in range(NF)])
env = np.clip(env / (np.percentile(env[env > 0.003], 90) + 1e-6), 0, 1.2)

# ---------------- drawing helpers ----------------
SKIN = (241, 200, 165)
def rr(d, box, r, fill, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)

def wrap(text, font, maxw, d):
    words, lines, cur = text.split(), [], ''
    for wd in words:
        test = (cur + ' ' + wd).strip()
        if d.textlength(test, font=font) <= maxw: cur = test
        else: lines.append(cur); cur = wd
    if cur: lines.append(cur)
    return lines

STYLE = {
    'prof':  dict(body=(110, 80, 55), hair=(170, 170, 175), legs=(60, 55, 50), tie=(150, 30, 40), glasses=True, beard=True),
    'masha': dict(body=(70, 150, 110), hair=(120, 70, 35), legs=(50, 60, 100), long_hair=True, female=True),
    'sled':  dict(body=(40, 55, 90), hair=(35, 30, 30), legs=(35, 40, 60), tie=(80, 90, 110), folder=True),
    'judge': dict(body=(25, 25, 30), hair=(90, 60, 40), legs=(25, 25, 30), collar=True, female=True, bun=True),
}

def draw_char(d, kind, cx, by, mouth, blink, talking, ft, face_dir):
    st = STYLE[kind]
    bob = math.sin(ft * 0.25) * 3 if talking else math.sin(ft * 0.05) * 1.5
    by = by + bob
    # legs
    d.rounded_rectangle([cx - 45, by - 120, cx - 8, by], 8, fill=st['legs'])
    d.rounded_rectangle([cx + 8, by - 120, cx + 45, by], 8, fill=st['legs'])
    d.ellipse([cx - 55, by - 14, cx - 2, by + 6], fill=(30, 30, 30)); d.ellipse([cx + 2, by - 14, cx + 55, by + 6], fill=(30, 30, 30))
    # hair back (long)
    hx, hy = cx, by - 330
    if st.get('long_hair'):
        d.rounded_rectangle([hx - 62, hy - 40, hx + 62, hy + 90], 40, fill=st['hair'])
    # body
    top = by - 270
    if kind == 'judge':
        d.polygon([(cx - 75, top + 10), (cx + 75, top + 10), (cx + 95, by - 60), (cx - 95, by - 60)], fill=st['body'])
    d.rounded_rectangle([cx - 75, top, cx + 75, by - 105], 30, fill=st['body'])
    # neck
    d.rectangle([cx - 14, top - 22, cx + 14, top + 8], fill=SKIN)
    if st.get('collar'):
        d.polygon([(cx - 30, top), (cx + 30, top), (cx, top + 40)], fill=(245, 245, 245))
    if st.get('tie'):
        d.polygon([(cx - 22, top), (cx + 22, top), (cx, top + 28)], fill=(240, 240, 240))
        d.polygon([(cx - 8, top + 10), (cx + 8, top + 10), (cx + 11, top + 95), (cx, top + 108), (cx - 11, top + 95)], fill=st['tie'])
    # arms (gesture while talking)
    g = (math.sin(ft * 0.18) * 0.5 + 0.5) if talking else 0
    for side in (-1, 1):
        sx = cx + side * 75
        gest = g if side == face_dir else 0
        ex, ey = sx + side * 22, top + 120 - gest * 70
        hx2, hy2 = ex + side * (8 + gest * 30), ey + 70 - gest * 90
        d.line([(sx, top + 20), (ex, ey)], fill=st['body'], width=30)
        d.line([(ex, ey), (hx2, hy2)], fill=st['body'], width=26)
        d.ellipse([hx2 - 15, hy2 - 15, hx2 + 15, hy2 + 15], fill=SKIN)
    if st.get('folder'):
        d.rounded_rectangle([cx - face_dir * 110 - 30, top + 110, cx - face_dir * 110 + 30, top + 190], 4, fill=(200, 160, 60), outline=(120, 90, 30), width=2)
    # head
    d.ellipse([hx - 55, hy - 62, hx + 55, hy + 62], fill=SKIN)
    # ears
    d.ellipse([hx - 63, hy - 8, hx - 47, hy + 16], fill=SKIN); d.ellipse([hx + 47, hy - 8, hx + 63, hy + 16], fill=SKIN)
    # hair
    if st.get('female'):
        d.chord([hx - 60, hy - 70, hx + 60, hy + 30], 180, 360, fill=st['hair'])
        d.pieslice([hx - 60, hy - 66, hx + 10, hy + 10], 180, 300, fill=st['hair'])
        if st.get('bun'):
            d.ellipse([hx - 25, hy - 100, hx + 25, hy - 55], fill=st['hair'])
    else:
        d.chord([hx - 57, hy - 66, hx + 57, hy + 10], 180, 360, fill=st['hair'])
        d.rectangle([hx - 57, hy - 30, hx - 48, hy + 5], fill=st['hair']); d.rectangle([hx + 48, hy - 30, hx + 57, hy + 5], fill=st['hair'])
    if st.get('beard'):
        d.chord([hx - 48, hy - 5, hx + 48, hy + 66], 0, 180, fill=st['hair'])
    # eyes (look toward face_dir)
    ox = face_dir * 4
    for ex0 in (hx - 20, hx + 20):
        if blink:
            d.line([(ex0 - 8, hy), (ex0 + 8, hy)], fill=(40, 30, 30), width=3)
        else:
            d.ellipse([ex0 - 9, hy - 9, ex0 + 9, hy + 9], fill='white')
            d.ellipse([ex0 - 4 + ox, hy - 4, ex0 + 4 + ox, hy + 5], fill=(40, 30, 30))
    # brows
    for ex0 in (hx - 20, hx + 20):
        d.line([(ex0 - 10, hy - 17), (ex0 + 10, hy - 18)], fill=(70, 50, 40) if kind != 'prof' else (120, 120, 120), width=3)
    if st.get('glasses'):
        for ex0 in (hx - 20, hx + 20):
            d.ellipse([ex0 - 15, hy - 14, ex0 + 15, hy + 14], outline=(40, 40, 40), width=3)
        d.line([(hx - 5, hy - 2), (hx + 5, hy - 2)], fill=(40, 40, 40), width=3)
    # nose
    d.line([(hx + ox, hy + 6), (hx + ox + 3, hy + 20)], fill=(205, 155, 120), width=3)
    # mouth
    my = hy + 36
    if mouth > 0.08:
        h = 3 + mouth * 16
        d.ellipse([hx - 13, my - h / 2, hx + 13, my + h / 2], fill=(120, 30, 40))
    else:
        d.arc([hx - 14, my - 10, hx + 14, my + 6], 20, 160, fill=(120, 40, 40), width=3)
    if st.get('female'):
        pass

def name_tag(d, kind, cx, y):
    txt = {'prof':'Преподаватель','masha':'Студентка Маша','sled':'Следователь','judge':'Судья'}[kind]
    w = d.textlength(txt, font=F_NAME) + 24
    x0 = max(8, min(W - w - 8, cx - w / 2))
    rr(d, [x0, y, x0 + w, y + 32], 14, (255, 255, 255), (60, 60, 60), 2)
    d.text((x0 + 12, y + 5), txt, font=F_NAME, fill=(30, 30, 30))

# ---------------- backgrounds ----------------
BOARD = (300, 40, 980, 420)
def bg_class():
    im = Image.new('RGB', (W, H), (236, 224, 200)); d = ImageDraw.Draw(im)
    d.rectangle([0, 470, W, H], fill=(170, 130, 90))
    for x in range(0, W, 80): d.line([(x, 470), (x - 60, H)], fill=(155, 115, 80), width=2)
    d.rectangle([0, 455, W, 470], fill=(200, 185, 160))
    x0, y0, x1, y1 = BOARD
    d.rectangle([x0 - 14, y0 - 14, x1 + 14, y1 + 14], fill=(120, 85, 50))
    d.rectangle([x0, y0, x1, y1], fill=(40, 80, 60))
    d.rectangle([x0 + 40, y1 + 14, x1 - 40, y1 + 24], fill=(120, 85, 50))
    # window
    d.rectangle([40, 60, 230, 300], fill=(180, 215, 240), outline=(250, 250, 250), width=8); d.line([(135, 60), (135, 300)], fill=(250, 250, 250), width=6)
    d.rectangle([1050, 60, 1240, 300], fill=(180, 215, 240), outline=(250, 250, 250), width=8); d.line([(1145, 60), (1145, 300)], fill=(250, 250, 250), width=6)
    return im, (235, 245, 235), (255, 230, 120)
def bg_office():
    im = Image.new('RGB', (W, H), (200, 210, 220)); d = ImageDraw.Draw(im)
    d.rectangle([0, 470, W, H], fill=(110, 100, 95))
    # bookshelf
    d.rectangle([30, 80, 230, 460], fill=(120, 80, 50))
    import random; random.seed(3)
    for sy in (100, 190, 280, 370):
        x = 40
        while x < 215:
            bw = random.randint(10, 22); col = random.choice([(150, 40, 40), (40, 70, 140), (50, 110, 60), (190, 160, 70), (90, 60, 120)])
            d.rectangle([x, sy + random.randint(0, 20), x + bw, sy + 80], fill=col); x += bw + 2
        d.rectangle([30, sy + 80, 230, sy + 88], fill=(90, 60, 35))
    # wall screen
    x0, y0, x1, y1 = BOARD
    d.rectangle([x0 - 10, y0 - 10, x1 + 10, y1 + 10], fill=(30, 30, 35))
    d.rectangle([x0, y0, x1, y1], fill=(245, 248, 252))
    # desk
    d.rectangle([980, 430, 1270, 470], fill=(100, 70, 45))
    d.rectangle([1050, 380, 1170, 432], fill=(40, 40, 45)); d.rectangle([1100, 432, 1120, 440], fill=(40, 40, 45))
    return im, (25, 40, 70), (180, 40, 40)
def bg_court():
    im = Image.new('RGB', (W, H), (150, 105, 70)); d = ImageDraw.Draw(im)
    for x in range(0, W, 90): d.rectangle([x, 0, x + 86, 470], fill=(160, 113, 76))
    d.rectangle([0, 470, W, H], fill=(120, 40, 40))
    x0, y0, x1, y1 = BOARD
    d.rectangle([x0 - 10, y0 - 10, x1 + 10, y1 + 10], fill=(70, 45, 30))
    d.rectangle([x0, y0, x1, y1], fill=(250, 246, 235))
    return im, (60, 35, 20), (150, 30, 30)
def court_front(d):
    # judge bench in front of judge (right side)
    d.rectangle([985, 430, 1275, 620], fill=(110, 70, 40)); d.rectangle([975, 420, 1285, 440], fill=(85, 52, 30))
    for x in range(1005, 1260, 60): d.rectangle([x, 460, x + 40, 600], outline=(90, 55, 32), width=3)

BGS = {'class': bg_class, 'office': bg_office, 'court': bg_court}

def render_slide(base, slide, fg, accent):
    im = base.copy(); d = ImageDraw.Draw(im)
    if not slide: return im
    x0, y0, x1, y1 = BOARD
    d.text((x0 + 30, y0 + 22), slide['title'], font=F_TITLE, fill=accent)
    d.line([(x0 + 30, y0 + 66), (x1 - 30, y0 + 66)], fill=accent, width=2)
    if 'chart' in slide:
        data = slide['chart']; mx = max(v for _, v in data)
        cw = (x1 - x0 - 80) / len(data)
        for i, (lab, v) in enumerate(data):
            bx = x0 + 40 + i * cw + 12; bh = (y1 - y0 - 170) * v / mx
            col = accent if i == 0 else (150, 190, 160) if fg[0] > 200 else (120, 140, 170)
            d.rectangle([bx, y1 - 60 - bh, bx + cw - 24, y1 - 60], fill=col)
            d.text((bx + (cw - 24) / 2, y1 - 66 - bh), str(v), font=f(22, True), fill=fg, anchor='md')
            d.text((bx + (cw - 24) / 2, y1 - 52), lab, font=f(20), fill=fg, anchor='ma')
        d.text((x0 + 30, y1 - 24), slide['note'], font=f(16), fill=fg)
        return im
    y = y0 + 84
    for b in slide['bullets']:
        for k, ln in enumerate(wrap(b, F_BUL, x1 - x0 - 90, d)):
            if k == 0 and not b.startswith(('идеологии', 'нуждающимися', 'направленных', 'убеждённости', 'представления', 'принудительные', 'телекоммуникационные', '→', '+', 'публичное оправдание', 'террористического', '(+')):
                d.ellipse([x0 + 34, y + 9, x0 + 44, y + 19], fill=accent)
            d.text((x0 + 56, y), ln, font=F_BUL, fill=fg); y += 34
        y += 6
    return im

# subtitle chunks for each line
def split_sub(text):
    toks = re.split(r'(\s+)', text); parts, cur = [], ''
    ABBR = ('ст.', 'ч.', 'тыс.', 'млн', 'т.ч.', 'прим.', 'гл.', '(ст.', 'п.')
    for tk in toks:
        cur += tk
        if tk.strip() and tk[-1] in '.!?:' and tk.strip() not in ABBR and not re.fullmatch(r'\d+\.?\d*\.', tk.strip()):
            parts.append(cur.strip()); cur = ''
    if cur.strip(): parts.append(cur.strip())
    out = []
    for p in parts:
        while len(p) > 120:
            cut = p.rfind(', ', 0, 120)
            if cut < 40: cut = p.rfind(' ', 0, 110)
            out.append(p[:cut + 1].strip()); p = p[cut + 1:].strip()
        if p: out.append(p)
    # merge very short pieces
    m = []
    for p in out:
        if m and len(m[-1]) + len(p) < 150 and len(p) < 45: m[-1] += ' ' + p
        else: m.append(p)
    return m

def subtitle_img(text):
    im = Image.new('RGBA', (W, 110), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    lines = wrap(text, F_SUB, W - 160, d)
    hgt = len(lines) * 34 + 20; y0 = 110 - hgt - 6
    d.rounded_rectangle([60, y0, W - 60, 110 - 6], 14, fill=(0, 0, 0, 170))
    for i, ln in enumerate(lines):
        d.text((W / 2, y0 + 12 + i * 34), ln, font=F_SUB, fill=(255, 255, 255), anchor='ma')
    return im

# title / outro cards
def card(lines, sub):
    im = Image.new('RGB', (W, H), (25, 40, 70)); d = ImageDraw.Draw(im)
    d.rectangle([0, H - 12, W, H], fill=(200, 160, 60))
    y = 230
    for ln in lines:
        d.text((W / 2, y), ln, font=F_BIG, fill=(255, 255, 255), anchor='ma'); y += 72
    y += 20
    for ln in sub:
        d.text((W / 2, y), ln, font=f(24), fill=(210, 220, 235), anchor='ma'); y += 36
    return im
TITLE = card(['Статья 205.2 УК РФ'], ['Публичные призывы к осуществлению террористической деятельности,', 'публичное оправдание терроризма или пропаганда терроризма', '', 'Учебный разбор'])
OUTRO_IM = card(['Спасибо за внимание'], ['Источники: УК РФ; УПК РФ; ФЗ «Об оперативно-розыскной деятельности»;',
                 'Постановления Пленума ВС РФ от 09.02.2012 № 1 и от 28.06.2011 № 11;',
                 'данные Судебного департамента при ВС РФ и МВД России.', '', 'Видео подготовлено в учебных целях'])

# precompute per-line visual state
line_at = []  # list of (start, end, si, li, slide, subchunks)
cur_slide = {}
for (s, e, si, li) in timeline:
    who, say, sub, sl = SCENES[si]['lines'][li]
    if li == 0: cur_slide[si] = None
    if sl is not None: cur_slide[si] = sl
    chunks_ = split_sub(sub or say)
    tot = sum(len(c) for c in chunks_)
    acc, timed = s, []
    for c in chunks_:
        d_ = (e - s) * len(c) / tot; timed.append((acc, acc + d_, c)); acc += d_
    line_at.append((s, e, si, li, cur_slide[si], timed))

bases = {}
slide_cache, sub_cache = {}, {}
def scene_base(si):
    if si not in bases: bases[si] = BGS[SCENES[si]['bg']]()
    return bases[si]

import os
SNAP = [int(x) for x in os.environ.get('SNAP','').split(',') if x]
ff = None if SNAP else subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-i', 'narration.wav', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p',
                       '-c:a', 'aac', '-b:a', '128k', '-shortest', OUT], stdin=subprocess.PIPE)

idx = 0
last_line = None
for fi in (SNAP or range(NF)):
    tt = fi / FPS
    if tt < INTRO:
        frame = TITLE
    elif tt >= TOTAL - OUTRO:
        frame = OUTRO_IM
    else:
        while idx + 1 < len(line_at) and line_at[idx + 1][0] - (1.2 if line_at[idx + 1][3] == 0 else 0) <= tt: idx += 1
        s, e, si, li, slide, timed = line_at[idx]
        if tt < s: # in transition gap before first line of scene -> show previous state
            pass
        base, fg, accent = scene_base(si)
        key = (si, id(slide))
        if key not in slide_cache: slide_cache[key] = render_slide(base, slide, fg, accent)
        frame = slide_cache[key].copy(); d = ImageDraw.Draw(frame)
        sc = SCENES[si]; speaker = sc['lines'][li][0]
        talking_now = s <= tt <= e
        for pos, kind in (('left', sc['left']), ('right', sc['right'])):
            cx = 150 if pos == 'left' else 1130
            by = 640
            face = 1 if pos == 'left' else -1
            is_sp = (kind == speaker)
            mouth = env[fi] if (is_sp and talking_now) else 0
            blink = (fi + (0 if pos == 'left' else 37)) % 95 < 3
            draw_char(d, kind, cx, by, mouth, blink, is_sp and talking_now, fi, face)
            if kind == 'judge': court_front(d)
        # name tag of speaker
        spx = 150 if sc['left'] == speaker else 1130
        name_tag(d, speaker, spx, 168)
        # subtitles
        for (a_, b_, txt) in timed:
            if a_ <= tt < b_ + PAUSE:
                if txt not in sub_cache: sub_cache[txt] = subtitle_img(txt)
                frame.paste(sub_cache[txt], (0, H - 110), sub_cache[txt]); break
        # transition fade-in at scene start
        if li == 0 and tt < s:
            k = max(0.0, (s - tt) / 1.2)
            frame = Image.blend(frame, Image.new('RGB', (W, H), (0, 0, 0)), k * 0.85)
    if SNAP: frame.save(f'snap_{fi}.png'); continue
    ff.stdin.write(frame.tobytes())
if ff: ff.stdin.close(); ff.wait()
print('done', round(TOTAL, 1), 's')
