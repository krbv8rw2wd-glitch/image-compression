import gradio as gr
import cv2
import numpy as np
import pywt
import time
import concurrent.futures
import os
import tempfile
import gc
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from scipy.stats import entropy
from scipy.fftpack import dct, idct
from PIL import Image
import io
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
# ==================== COMPRESSION FUNCTIONS (unchanged) ====================
def compute_coeff_entropy(coeffs):
    flat = coeffs.flatten()
    hist, _ = np.histogram(flat, bins=256, density=True)
    hist = hist + 1e-10
    return entropy(hist, base=2)

def compute_image_stats(img, title=""):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    hist = np.histogram(gray, bins=256, range=(0,256), density=True)[0]
    hist = hist + 1e-10
    entropy_val = entropy(hist, base=2)
    brightness = np.mean(gray)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(gray.ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
    ax.set_title(f"Histogram - {title}" if title else "Histogram")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    hist_img = Image.open(buf)

    return {
        "entropy": entropy_val,
        "brightness": brightness,
        "histogram": hist_img
    }

def resize_to_multiple_of_8(img):
    h, w = img.shape[:2]
    return img[:h - (h % 8), :w - (w % 8)]

QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

QC = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float32)

def scale_quant_matrix(Q, quality):
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q_scaled = np.floor((Q * scale + 50) / 100)
    Q_scaled[Q_scaled == 0] = 1
    return Q_scaled.astype(np.float32)

def get_true_jpeg_size(image_np, quality):
    pil_img = Image.fromarray(image_np)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.tell() / 1024.0

def process_blocks_vectorized(channel, qmat):
    h, w = channel.shape
    blocks = (channel.reshape(h // 8, 8, w // 8, 8)
              .swapaxes(1, 2)
              .reshape(-1, 8, 8))
    blocks = blocks - 128
    dct_blocks = dct(dct(blocks, axis=2, norm='ortho'), axis=1, norm='ortho')
    quant_blocks = np.round(dct_blocks / qmat)
    coeff_entropy = compute_coeff_entropy(quant_blocks)
    dequant_blocks = quant_blocks * qmat
    idct_blocks = idct(idct(dequant_blocks, axis=2, norm='ortho'), axis=1, norm='ortho')
    idct_blocks = np.clip(idct_blocks + 128, 0, 255).astype(np.uint8)
    reconstructed = (idct_blocks.reshape(h // 8, w // 8, 8, 8)
                     .swapaxes(1, 2)
                     .reshape(h, w))
    return reconstructed, coeff_entropy

def dct_compress_seq(img_rgb, quality=75, return_details=False):
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(img_ycc)
    QY_scaled = scale_quant_matrix(QY, quality)
    QC_scaled = scale_quant_matrix(QC, quality)

    if return_details:
        Y_disp = Y.astype(np.uint8)
        Cb_disp = Cb.astype(np.uint8)
        Cr_disp = Cr.astype(np.uint8)
        sample_y_block = Y[0:8, 0:8].copy() - 128
        dct_sample = cv2.dct(sample_y_block)
        quant_sample = np.round(dct_sample / QY_scaled).astype(int)

    start = time.time()
    Y_rec, entY = process_blocks_vectorized(Y, QY_scaled)
    Cr_rec, entCr = process_blocks_vectorized(Cr, QC_scaled)
    Cb_rec, entCb = process_blocks_vectorized(Cb, QC_scaled)
    elapsed = time.time() - start

    coeff_entropy = (entY + entCr + entCb) / 3
    ycc_rec = cv2.merge([Y_rec, Cr_rec, Cb_rec]).astype(np.uint8)
    rgb_rec = cv2.cvtColor(ycc_rec, cv2.COLOR_YCrCb2RGB)
    compressed_size_kb = get_true_jpeg_size(rgb_rec, quality)

    if return_details:
        details = {
            'Y': Y_disp,
            'Cb': Cb_disp,
            'Cr': Cr_disp,
            'sample_quant_block': quant_sample,
            'quant_matrix': QY_scaled,
            'quality': quality,
            'size_kb': compressed_size_kb,
            'coeff_entropy': coeff_entropy
        }
        return rgb_rec, elapsed, details, compressed_size_kb, coeff_entropy
    return rgb_rec, elapsed, compressed_size_kb, coeff_entropy
  

# ==================== OPTIMISED PARALLEL DCT (processes + tiles + shared memory) ====================
def _dct_compress_tile(tile_info):
    """Process a single tile: DCT compress and return reconstructed tile."""
    shm_name, shape, dtype, offset_rows, offset_cols, tile_h, tile_w, QY_scaled, QC_scaled = tile_info
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        img = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        # Extract tile (RGB)
        tile = img[offset_rows:offset_rows+tile_h, offset_cols:offset_cols+tile_w].copy()
        existing_shm.close()
    except:
        # Fallback if shared memory fails (should not happen)
        return None, offset_rows, offset_cols

    # Convert to YCC
    tile_ycc = cv2.cvtColor(tile, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(tile_ycc)
    # Process each channel
    Y_rec, _ = process_blocks_vectorized(Y, QY_scaled)
    Cr_rec, _ = process_blocks_vectorized(Cr, QC_scaled)
    Cb_rec, _ = process_blocks_vectorized(Cb, QC_scaled)
    ycc_rec = cv2.merge([Y_rec, Cr_rec, Cb_rec]).astype(np.uint8)
    tile_rec = cv2.cvtColor(ycc_rec, cv2.COLOR_YCrCb2RGB)
    return tile_rec, offset_rows, offset_cols

def dct_compress_parallel_optimized(img_rgb, quality=75, max_workers=None):
    """Parallel DCT using tiles and shared memory – faster than sequential for large images."""
    if max_workers is None:
        max_workers = mp.cpu_count()
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    h, w = img.shape[:2]

    # Only use parallel if image is large enough ( > 2MP )
    if h * w < 2_000_000:
        # For small images, fall back to sequential
        return dct_compress_seq(img_rgb, quality)

    # Determine tile size (512x512 works well)
    tile_size = 512
    # Ensure tile dimensions are multiples of 8
    tile_h = ((min(tile_size, h) + 7) // 8) * 8
    tile_w = ((min(tile_size, w) + 7) // 8) * 8

    # Prepare tile list
    tiles = []
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            th = min(tile_h, h - i)
            tw = min(tile_w, w - j)
            tiles.append((i, j, th, tw))

    # Precompute quantization matrices
    QY_scaled = scale_quant_matrix(QY, quality)
    QC_scaled = scale_quant_matrix(QC, quality)

    # Create shared memory for the image
    img_nbytes = img.nbytes
    shm = shared_memory.SharedMemory(create=True, size=img_nbytes)
    shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
    shared_img[:] = img[:]  # copy into shared memory

    # Prepare arguments for each tile
    args = []
    for i, j, th, tw in tiles:
        args.append((shm.name, img.shape, img.dtype, i, j, th, tw, QY_scaled, QC_scaled))

    start = time.time()
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(_dct_compress_tile, args)
    elapsed = time.time() - start

    # Reconstruct full image from tiles
    rec_img = np.zeros_like(img)
    for tile_rec, i, j in results:
        if tile_rec is None:
            continue
        th, tw = tile_rec.shape[:2]
        rec_img[i:i+th, j:j+tw] = tile_rec

    # Clean up shared memory
    shm.close()
    shm.unlink()

    # Compute compressed size and coefficient entropy (optional)
    compressed_size_kb = get_true_jpeg_size(rec_img, quality)
    # For simplicity, skip exact coeff entropy calculation (or compute from whole image)
    coeff_entropy = 0.0  # You can compute if needed, but it's slow
    return rec_img, elapsed, compressed_size_kb, coeff_entropy

# Replace your old dct_compress_parallel with the optimised one
dct_compress_parallel = dct_compress_parallel_optimized



def get_original_coeff_entropy(img_rgb):
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(img_ycc)

    def get_channel_coeffs(ch):
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        return np.concatenate([LH.flatten(), HL.flatten(), HH.flatten()])

    all_coeffs = np.concatenate([get_channel_coeffs(Y), get_channel_coeffs(Cr), get_channel_coeffs(Cb)])
    return compute_coeff_entropy(all_coeffs)

def get_compressed_coeff_entropy(img_rgb):
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(img_ycc)
    def get_details(ch):
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        return np.concatenate([LH.flatten(), HL.flatten(), HH.flatten()])
    all_details = np.concatenate([get_details(Y), get_details(Cr), get_details(Cb)])
    return compute_coeff_entropy(all_details)

def dwt_compress_seq(img_rgb, threshold=30, return_details=False):
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(img_ycc)

    def process_channel(ch, thresh):
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        LH[np.abs(LH) < thresh] = 0
        HL[np.abs(HL) < thresh] = 0
        HH[np.abs(HH) < thresh] = 0
        rec = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        return np.clip(rec, 0, 255).astype(np.uint8), LL, LH, HL, HH

    start = time.time()
    Y_rec, LLy, LHy, HLy, HHy = process_channel(Y, threshold)
    Cr_rec, LLcr, LHcr, HLcr, HHcr = process_channel(Cr, threshold)
    Cb_rec, LLcb, LHcb, HLcb, HHcb = process_channel(Cb, threshold)
    elapsed = time.time() - start

    all_details = np.concatenate([LHy.flatten(), HLy.flatten(), HHy.flatten(),
                                  LHcr.flatten(), HLcr.flatten(), HHcr.flatten(),
                                  LHcb.flatten(), HLcb.flatten(), HHcb.flatten()])
    coeff_entropy = compute_coeff_entropy(all_details)

    ycc_rec = cv2.merge([Y_rec, Cr_rec, Cb_rec])
    rgb_rec = cv2.cvtColor(ycc_rec, cv2.COLOR_YCrCb2RGB)
    compressed_size_kb = get_true_jpeg_size(rgb_rec, 75)

    if return_details:
        def norm(arr):
            arr = np.abs(arr)
            if arr.max() > 0:
                arr = 255 * arr / arr.max()
            return arr.astype(np.uint8)
        subbands = {
            'LL': norm(LLy),
            'LH': norm(LHy),
            'HL': norm(HLy),
            'HH': norm(HHy),
            'zeroed_percent': 100.0 * (1 - (np.count_nonzero(LHy)+np.count_nonzero(HLy)+np.count_nonzero(HHy)) / (LHy.size+HLy.size+HHy.size))
        }
        return rgb_rec, elapsed, subbands, compressed_size_kb, coeff_entropy
    return rgb_rec, elapsed, compressed_size_kb, coeff_entropy

def _process_tile_dwt(tile_data):
    tile, threshold = tile_data
    tile_ycc = cv2.cvtColor(tile, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y_t, Cr_t, Cb_t = cv2.split(tile_ycc)

    def proc(ch, thr):
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        LH[np.abs(LH) < thr] = 0
        HL[np.abs(HL) < thr] = 0
        HH[np.abs(HH) < thr] = 0
        return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    Y_rec = np.clip(proc(Y_t, threshold), 0, 255).astype(np.uint8)
    Cr_rec = np.clip(proc(Cr_t, threshold), 0, 255).astype(np.uint8)
    Cb_rec = np.clip(proc(Cb_t, threshold), 0, 255).astype(np.uint8)

    rec_tile = cv2.cvtColor(cv2.merge([Y_rec, Cr_rec, Cb_rec]), cv2.COLOR_YCrCb2RGB)
    return rec_tile

# ==================== OPTIMISED PARALLEL DWT (processes + tiles + shared memory) ====================
def _dwt_compress_tile(tile_info):
    """Process a single tile: DWT compress and return reconstructed tile."""
    shm_name, shape, dtype, offset_rows, offset_cols, tile_h, tile_w, threshold = tile_info
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        img = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        tile = img[offset_rows:offset_rows+tile_h, offset_cols:offset_cols+tile_w].copy()
        existing_shm.close()
    except:
        return None, offset_rows, offset_cols

    # DWT compression on tile
    tile_ycc = cv2.cvtColor(tile, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(tile_ycc)

    def proc_channel(ch, thr):
        coeffs = pywt.dwt2(ch, 'haar')
        LL, (LH, HL, HH) = coeffs
        LH[np.abs(LH) < thr] = 0
        HL[np.abs(HL) < thr] = 0
        HH[np.abs(HH) < thr] = 0
        return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    Y_rec = np.clip(proc_channel(Y, threshold), 0, 255).astype(np.uint8)
    Cr_rec = np.clip(proc_channel(Cr, threshold), 0, 255).astype(np.uint8)
    Cb_rec = np.clip(proc_channel(Cb, threshold), 0, 255).astype(np.uint8)

    ycc_rec = cv2.merge([Y_rec, Cr_rec, Cb_rec]).astype(np.uint8)
    tile_rec = cv2.cvtColor(ycc_rec, cv2.COLOR_YCrCb2RGB)
    return tile_rec, offset_rows, offset_cols

def dwt_compress_parallel_optimized(img_rgb, threshold=30, max_workers=None):
    """Parallel DWT using tiles and shared memory – faster than sequential for large images."""
    if max_workers is None:
        max_workers = mp.cpu_count()
    img = resize_to_multiple_of_8(img_rgb).astype(np.uint8)
    h, w = img.shape[:2]

    # Only use parallel if image is large enough (> 2MP)
    if h * w < 2_000_000:
        compressed, elapsed, comp_size, _ = dwt_compress_seq(img_rgb, threshold)
        return compressed, elapsed, comp_size

    tile_size = 512
    tile_h = ((min(tile_size, h) + 7) // 8) * 8
    tile_w = ((min(tile_size, w) + 7) // 8) * 8

    tiles = []
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            th = min(tile_h, h - i)
            tw = min(tile_w, w - j)
            tiles.append((i, j, th, tw))

    # Shared memory
    img_nbytes = img.nbytes
    shm = shared_memory.SharedMemory(create=True, size=img_nbytes)
    shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
    shared_img[:] = img[:]

    args = [(shm.name, img.shape, img.dtype, i, j, th, tw, threshold) for (i, j, th, tw) in tiles]

    start = time.time()
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(_dwt_compress_tile, args)
    elapsed = time.time() - start

    # Reconstruct
    rec_img = np.zeros_like(img)
    for tile_rec, i, j in results:
        if tile_rec is None:
            continue
        th, tw = tile_rec.shape[:2]
        rec_img[i:i+th, j:j+tw] = tile_rec

    shm.close()
    shm.unlink()

    compressed_size_kb = get_true_jpeg_size(rec_img, 75)
    return rec_img, elapsed, compressed_size_kb

# Replace old dwt_compress_parallel with optimised one
dwt_compress_parallel = dwt_compress_parallel_optimized

# ==================== VISUALISATION HELPERS ====================
def quant_matrix_to_image(Q, title="Quantization Matrix"):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(Q, cmap='viridis', aspect='equal')
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(int(Q[i,j])), ha='center', va='center',
                   color='white' if Q[i,j] < 50 else 'black', fontsize=8)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def dct_block_heatmap(block, title="Quantized DCT Block"):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(block, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(8):
        for j in range(8):
            val = block[i, j]
            color = 'white' if abs(val) > 20 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center', fontsize=7, color=color)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def zigzag_diagram():
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((j, 7-i), 1, 1, fill=False, edgecolor='black'))
    order = []
    for s in range(15):
        for i in range(max(0, s-7), min(7, s)+1):
            j = s - i
            if (s % 2 == 0):
                order.append((i, j))
            else:
                order.append((j, i))
    for idx, (r, c) in enumerate(order):
        ax.text(c+0.5, 7-r+0.5, str(idx+1), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(0,8)
    ax.set_ylim(0,8)
    ax.set_xticks(np.arange(0.5, 8.5, 1))
    ax.set_yticks(np.arange(0.5, 8.5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Zigzag Scanning Order (JPEG)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ==================== LINKED ZOOM ====================
def linked_zoom(orig_img, comp_img, evt: gr.SelectData):
    if orig_img is None or comp_img is None:
        return None, None
    x, y = evt.index
    h, w = orig_img.shape[:2]
    radius = 200
    x1 = max(0, x - radius)
    x2 = min(w, x + radius)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius)
    orig_crop = orig_img[y1:y2, x1:x2].copy()
    comp_crop = comp_img[y1:y2, x1:x2].copy()
    return orig_crop, comp_crop

# ==================== COMPRESSION + DISPLAY ====================
def compress_and_show(filepath, method, parallel_mode, show_details):
    if filepath is None:
        return (None, None, "No image uploaded", 0, [], "", "", None, None, gr.update(visible=False))

    image = cv2.imread(filepath)
    if image is None:
        return (None, None, "Could not decode image", 0, [], "", "", None, None, gr.update(visible=False))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = resize_to_multiple_of_8(image)
    orig_file_size_kb = os.path.getsize(filepath) / 1024.0

    orig_stats = compute_image_stats(img_resized, "Original")
    orig_pixel_entropy = orig_stats["entropy"]
    orig_brightness = orig_stats["brightness"]
    orig_coeff_entropy = get_original_coeff_entropy(img_resized)

    detail_gallery = []
    detail_caption = ""
    compressed_size_kb = 0
    coeff_entropy = 0.0
    compressed_np = None
    processing_time = 0.0

    if show_details and parallel_mode == "Parallel":
        parallel_mode = "Sequential"
        detail_caption = "ℹ️ Details are shown only in Sequential mode. Switched to Sequential."

    if method == "JPEG (DCT)":
        quality = 75
        if parallel_mode == "Sequential":
            if show_details:
                compressed_np, t, details, comp_size, ce = dct_compress_seq(img_resized, quality, return_details=True)
                coeff_entropy = ce
                detail_gallery.extend([
                    (Image.fromarray(details['Y']), "🔆 Y Luminance"),
                    (Image.fromarray(details['Cb']), "🔵 Cb Chrominance"),
                    (Image.fromarray(details['Cr']), "🔴 Cr Chrominance")
                ])
                qimg = quant_matrix_to_image(details['quant_matrix'], f"Luminance Quantization Matrix (Q={quality})")
                detail_gallery.append((qimg, "📊 Quantization Matrix"))
                zero_count = np.sum(details['sample_quant_block'] == 0)
                heatmap_img = dct_block_heatmap(details['sample_quant_block'],
                                               f"Sample Quantized DCT Block (Y)\nZero coeffs: {zero_count}/64")
                detail_gallery.append((heatmap_img, "🎯 Quantized DCT Block"))
                detail_gallery.append((zigzag_diagram(), "🔀 Zigzag Scanning Order"))
                detail_caption = f"**JPEG (DCT)** Quality={quality} | Original: {orig_file_size_kb:.2f} KB | Compressed: {comp_size:.2f} KB | Coeff entropy: {ce:.3f}"
            else:
                compressed_np, t, comp_size, ce = dct_compress_seq(img_resized, quality)
                coeff_entropy = ce
            compressed_size_kb = comp_size
            processing_time = t
        else:
            compressed_np, t, comp_size, ce = dct_compress_parallel(img_resized, quality)
            coeff_entropy = ce
            compressed_size_kb = comp_size
            processing_time = t
            if show_details:
                _, _, details, _, _ = dct_compress_seq(img_resized, quality, return_details=True)
                detail_gallery.extend([
                    (Image.fromarray(details['Y']), "🔆 Y Luminance"),
                    (Image.fromarray(details['Cb']), "🔵 Cb Chrominance"),
                    (Image.fromarray(details['Cr']), "🔴 Cr Chrominance"),
                    (quant_matrix_to_image(details['quant_matrix'], f"Luminance Matrix"), "📊 Matrix"),
                    (dct_block_heatmap(details['sample_quant_block'], "Quantized DCT Block"), "🎯 DCT Block"),
                    (zigzag_diagram(), "🔀 Zigzag")
                ])
                detail_caption = f"Parallel | Original: {orig_file_size_kb:.2f} KB | Compressed: {compressed_size_kb:.2f} KB"

    else:
        threshold = 30
        if parallel_mode == "Sequential":
            if show_details:
                compressed_np, t, subbands, comp_size, ce = dwt_compress_seq(img_resized, threshold, return_details=True)
                coeff_entropy = ce
                detail_gallery.extend([
                    (Image.fromarray(subbands['LL']), "📐 LL Approximation (Y)"),
                    (Image.fromarray(subbands['LH']), "➡️ LH Horizontal details (Y)"),
                    (Image.fromarray(subbands['HL']), "⬇️ HL Vertical details (Y)"),
                    (Image.fromarray(subbands['HH']), "🔲 HH Diagonal details (Y)")
                ])
                detail_caption = f"**JPEG (DWT)** Threshold={threshold} | Original: {orig_file_size_kb:.2f} KB | Compressed: {comp_size:.2f} KB | Coeff entropy: {ce:.3f}"
            else:
                compressed_np, t, comp_size, ce = dwt_compress_seq(img_resized, threshold)
                coeff_entropy = ce
            compressed_size_kb = comp_size
            processing_time = t
        else:
            compressed_np, t, comp_size = dwt_compress_parallel(img_resized, threshold)
            coeff_entropy = get_compressed_coeff_entropy(compressed_np)
            compressed_size_kb = comp_size
            processing_time = t
            if show_details:
                _, _, subbands, _, _ = dwt_compress_seq(img_resized, threshold, return_details=True)
                detail_gallery.extend([
                    (Image.fromarray(subbands['LL']), "📐 LL Approximation (Y)"),
                    (Image.fromarray(subbands['LH']), "➡️ LH Horizontal (Y)"),
                    (Image.fromarray(subbands['HL']), "⬇️ HL Vertical (Y)"),
                    (Image.fromarray(subbands['HH']), "🔲 HH Diagonal (Y)")
                ])
                detail_caption = f"Parallel | Original: {orig_file_size_kb:.2f} KB | Compressed: {compressed_size_kb:.2f} KB"

    temp_jpeg = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(compressed_np).save(temp_jpeg.name, "JPEG", quality=75, optimize=True)
    temp_jpeg.close()
    jpeg_filepath = temp_jpeg.name

    comp_stats = compute_image_stats(compressed_np, "Compressed")
    comp_pixel_entropy = comp_stats["entropy"]
    comp_brightness = comp_stats["brightness"]

    psnr_val = peak_signal_noise_ratio(img_resized, compressed_np, data_range=255)
    ssim_val = structural_similarity(img_resized, compressed_np, channel_axis=2, data_range=255)
    mse_val = mean_squared_error(img_resized, compressed_np)

    compression_ratio = orig_file_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0
    space_gained_kb = orig_file_size_kb - compressed_size_kb
    space_gained_percent = (space_gained_kb / orig_file_size_kb) * 100 if orig_file_size_kb > 0 else 0

    metrics = f"""
### 📊 Image Statistics & Metrics

| Property | Original | Compressed |
|----------|----------|------------|
| **Size (KB)** | {orig_file_size_kb:.1f} | {compressed_size_kb:.1f} |
| **Pixel Entropy** | {orig_pixel_entropy:.1f} | {comp_pixel_entropy:.1f} |
| **Coeff Entropy** | {orig_coeff_entropy:.1f} | {coeff_entropy:.1f} |
| **Brightness (mean)** | {orig_brightness:.1f} | {comp_brightness:.1f} |
| **Compression Ratio** | - | {compression_ratio:.1f}:1 |
| **Space Gained** | - | {space_gained_kb:.1f} KB ({space_gained_percent:.1f}%) |

### 📈 Quality Metrics

| Metric | Value |
|--------|-------|
| **PSNR** | {psnr_val:.1f} dB |
| **MSE** | {mse_val:.1f} |
| **SSIM** | {ssim_val:.4f} |
"""

    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].hist(cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
    axes[0].set_title("Original Histogram")
    axes[1].hist(cv2.cvtColor(compressed_np, cv2.COLOR_RGB2GRAY).ravel(), bins=256, range=(0,256), color='red', alpha=0.7)
    axes[1].set_title("Compressed Histogram")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    hist_comparison = Image.open(buf)

    if show_details:
        detail_gallery.insert(0, (hist_comparison, "📊 Histogram Comparison"))

    gc.collect()

    return (img_resized, compressed_np, metrics, round(processing_time, 4),
            detail_gallery, detail_caption, jpeg_filepath, f"{compressed_size_kb:.1f} KB",
            gr.update(visible=True))

# ==================== UI ====================
CUSTOM_CSS = """
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
* {
    transition: all 0.3s ease;
}
.gradio-container { 
    max-width: 1600px !important; 
    margin: auto !important; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
}
.main-card { 
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px; 
    padding: 25px; 
    margin: 15px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    animation: fadeIn 0.6s ease-out;
}
.main-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 48px rgba(0,0,0,0.15);
}
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    animation: slideInLeft 0.6s ease-out;
}
h3 {
    color: #667eea !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
}
.team-info {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0e6ff 100%);
    border-radius: 15px;
    padding: 20px;
    margin-top: 10px;
    color: #2d3748;
    text-align: center;
    animation: slideInRight 0.6s ease-out;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.team-title {
    font-size: 1.4em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #667eea !important;
    text-decoration-thickness: 3px !important;
    margin-bottom: 20px !important;
    color: #4a5568 !important;
}
.student-title {
    font-size: 1.15em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #48bb78 !important;
    text-decoration-thickness: 2px !important;
    color: #2f855a !important;
    margin-bottom: 10px !important;
}
.supervisor-title {
    font-size: 1.15em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #ed8936 !important;
    text-decoration-thickness: 2px !important;
    color: #c05621 !important;
    margin-bottom: 10px !important;
}
.team-info .students {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-top: 10px;
    margin-bottom: 15px;
}
.team-info .student {
    background: rgba(102, 126, 234, 0.1);
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 500;
    color: #4a5568;
    transition: all 0.3s ease;
}
.team-info .student:hover {
    background: rgba(102, 126, 234, 0.2);
    transform: translateY(-2px);
}
.team-info .supervisor {
    margin-top: 15px;
    font-size: 1em;
    font-weight: 500;
}
.supervisor-name {
    font-size: 1.05em;
    font-weight: bold;
    color: #c05621;
    background: rgba(237, 137, 54, 0.1);
    display: inline-block;
    padding: 5px 20px;
    border-radius: 25px;
    margin-top: 5px;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    border-radius: 50px !important;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    animation: pulse 0.5s ease;
}
.gr-button-primary:active {
    transform: translateY(0);
}
table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px;
    font-weight: 600;
}
td {
    padding: 10px;
    border-bottom: 1px solid #e0e0e0;
}
tr:hover {
    background: #f8f9ff;
    transform: scale(1.01);
}
.loading {
    position: relative;
    overflow: hidden;
}
.loading::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 1.5s infinite;
}
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    .main-card {
        padding: 15px;
    }
    h1 {
        font-size: 1.8em !important;
    }
    .team-info .students {
        flex-direction: column;
        gap: 10px;
    }
}
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555;
}
.size-display {
    font-weight: 600;
    text-align: center;
}
/* White background for zoomed images */
.zoomed-img {
    background: white !important;
    padding: 10px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}
/* Labels for Original and Compressed */
.image-label {
    background: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 10px;
    font-size: 1.1em;
    color: #4a5568;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: inline-block;
    width: auto;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
* {
    transition: all 0.3s ease;
}
.gradio-container { 
    max-width: 1600px !important; 
    margin: auto !important; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
}
.main-card { 
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px; 
    padding: 25px; 
    margin: 15px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    animation: fadeIn 0.6s ease-out;
}
.main-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 48px rgba(0,0,0,0.15);
}
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    animation: slideInLeft 0.6s ease-out;
}
h3 {
    color: #667eea !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
}
.team-info {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0e6ff 100%);
    border-radius: 15px;
    padding: 20px;
    margin-top: 10px;
    color: #2d3748;
    text-align: center;
    animation: slideInRight 0.6s ease-out;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.team-title {
    font-size: 1.4em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #667eea !important;
    text-decoration-thickness: 3px !important;
    margin-bottom: 20px !important;
    color: #4a5568 !important;
}
.student-title {
    font-size: 1.15em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #48bb78 !important;
    text-decoration-thickness: 2px !important;
    color: #2f855a !important;
    margin-bottom: 10px !important;
}
.supervisor-title {
    font-size: 1.15em !important;
    font-weight: bold !important;
    text-decoration: underline !important;
    text-decoration-color: #ed8936 !important;
    text-decoration-thickness: 2px !important;
    color: #c05621 !important;
    margin-bottom: 10px !important;
}
.team-info .students {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-top: 10px;
    margin-bottom: 15px;
}
.team-info .student {
    background: rgba(102, 126, 234, 0.1);
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 500;
    color: #4a5568;
    transition: all 0.3s ease;
}
.team-info .student:hover {
    background: rgba(102, 126, 234, 0.2);
    transform: translateY(-2px);
}
.team-info .supervisor {
    margin-top: 15px;
    font-size: 1em;
    font-weight: 500;
}
.supervisor-name {
    font-size: 1.05em;
    font-weight: bold;
    color: #c05621;
    background: rgba(237, 137, 54, 0.1);
    display: inline-block;
    padding: 5px 20px;
    border-radius: 25px;
    margin-top: 5px;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    border-radius: 50px !important;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    animation: pulse 0.5s ease;
}
.gr-button-primary:active {
    transform: translateY(0);
}
table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px;
    font-weight: 600;
}
td {
    padding: 10px;
    border-bottom: 1px solid #e0e0e0;
}
tr:hover {
    background: #f8f9ff;
    transform: scale(1.01);
}
.loading {
    position: relative;
    overflow: hidden;
}
.loading::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 1.5s infinite;
}
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    .main-card {
        padding: 15px;
    }
    h1 {
        font-size: 1.8em !important;
    }
    .team-info .students {
        flex-direction: column;
        gap: 10px;
    }
}
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555;
}
.size-display {
    font-weight: 600;
    text-align: center;
}
.zoomed-img {
    background: white !important;
    padding: 10px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}
.image-label {
    background: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 10px;
    font-size: 1.1em;
    color: #4a5568;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: inline-block;
    width: auto;
}
"""
with gr.Blocks(title="Image Compression with Parallel Processing") as demo:
    with gr.Column(elem_classes=["main-card"]):
        gr.Markdown("# 🖼️ Image Compression using JPEG DCT & DWT")
        gr.Markdown("### Master 2 RSD - University of Oran 1")

        with gr.Column(elem_classes=["team-info"]):
            gr.Markdown("### 👨‍🎓 Project Team", elem_classes=["team-title"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Students**", elem_classes=["student-title"])
                    gr.Markdown("• **Hamza Touria Fatima Zohra**\n• **Benhaddou Nour El Houda**", elem_classes=["student"])
                with gr.Column():
                    gr.Markdown("**Supervisor**", elem_classes=["supervisor-title"])
                    gr.Markdown("**Dr. Naoui Oumelkheir**", elem_classes=["supervisor-name"])

        gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["main-card"]):
            gr.Markdown("### 📤 Upload & Settings")
            input_image = gr.Image(type="filepath", label="Choose an image")
            method = gr.Radio(["JPEG (DCT)", "JPEG (DWT)"], value="JPEG (DCT)", label="Compression Method")
            parallel_mode = gr.Radio(["Sequential", "Parallel"], value="Sequential", label="Processing Mode")
            show_details = gr.Checkbox(label="🔍 Show intermediate steps", value=False)
            compress_btn = gr.Button("🚀 Compress", variant="primary")

        with gr.Column(scale=1, elem_classes=["main-card"]):
            gr.Markdown("### 📥 Results")
            output_image = gr.Image(type="filepath", label="Compressed Image (download)", height=200)
            with gr.Row():
                comp_size_disp = gr.Textbox(label="Compressed Size", interactive=False, elem_classes=["size-display"])
            metrics_text = gr.Markdown()
            time_display = gr.Number(label="⏱️ Processing Time (seconds)", precision=4)

    # Hidden row that appears after compression
    with gr.Column(visible=False) as images_row:
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="image-label">Original image</div>')
                # Removed unsupported arguments: sources, show_download_button, show_share_button
                orig_display = gr.Image(
                    type="numpy", label=None, height=300,
                    interactive=True, show_label=False
                )
            with gr.Column():
                gr.HTML('<div class="image-label">Compressed image</div>')
                comp_display = gr.Image(
                    type="numpy", label=None, height=300,
                    interactive=True, show_label=False
                )

        with gr.Row():
            with gr.Column(elem_classes=["zoomed-img"]):
                gr.HTML('<div class="image-label">Zoom Original image</div>')
                zoomed_orig = gr.Image(type="numpy", label=None, height=350, show_label=False, interactive=False)
            with gr.Column(elem_classes=["zoomed-img"]):
                gr.HTML('<div class="image-label">Zoom Compressed image</div>')
                zoomed_comp = gr.Image(type="numpy", label=None, height=350, show_label=False, interactive=False)

    with gr.Column(elem_classes=["main-card"]):
        detail_gallery = gr.Gallery(
            label="🔍 Detailed Processing Steps",
            columns=3,
            height="auto",
            object_fit="contain",
            visible=False,
            allow_preview=True
        )
        detail_text = gr.Markdown(visible=False)

    def update_visibility(show):
        return {detail_gallery: gr.update(visible=show), detail_text: gr.update(visible=show)}

    show_details.change(
        update_visibility,
        inputs=show_details,
        outputs=[detail_gallery, detail_text]
    )

    # Linked zoom callbacks
    def on_click_original(orig, comp, evt: gr.SelectData):
        return linked_zoom(orig, comp, evt)

    def on_click_compressed(orig, comp, evt: gr.SelectData):
        return linked_zoom(orig, comp, evt)

    orig_display.select(on_click_original, inputs=[orig_display, comp_display], outputs=[zoomed_orig, zoomed_comp])
    comp_display.select(on_click_compressed, inputs=[orig_display, comp_display], outputs=[zoomed_orig, zoomed_comp])

    compress_btn.click(
        compress_and_show,
        inputs=[input_image, method, parallel_mode, show_details],
        outputs=[
            orig_display,
            comp_display,
            metrics_text,
            time_display,
            detail_gallery,
            detail_text,
            output_image,
            comp_size_disp,
            images_row
        ]
    )

if __name__ == "__main__":
    import os
    # Get the port assigned by Render (or default to 10000 for local testing)
    port = int(os.environ.get("PORT", 10000))
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set or not needed
    demo.launch(
        server_name="0.0.0.0",   # listen on all network interfaces
        server_port=port,        # use Render's dynamic port
        css=CUSTOM_CSS          # keep your custom styling
    )
