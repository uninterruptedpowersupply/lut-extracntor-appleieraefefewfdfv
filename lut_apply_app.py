import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import time # For demonstration
import queue
import threading
import concurrent.futures
import math
import multiprocessing # <--- IMPORT ADDED HERE

# --- Core LUT Application Logic (Modified for chunking) ---

def _trilinear_interpolate_pixel(r_in, g_in, b_in, lut_data_np, lut_n_size):
    """
    Performs trilinear interpolation for a single pixel.
    r_in, g_in, b_in: Input pixel color components (0-255).
    lut_data_np: NumPy array of the LUT image data (height, width, channels).
                 Expected shape: (N, N*N, 3)
    lut_n_size: The dimension of the LUT cube (e.g., 16, 32, 64).
    """
    # Normalize input color components to 0-1 range
    r_norm = r_in / 255.0
    g_norm = g_in / 255.0
    b_norm = b_in / 255.0

    # Calculate fractional indices within the LUT grid (0 to N-1 range)
    blue_idx_f = b_norm * (lut_n_size - 1)
    green_idx_f = g_norm * (lut_n_size - 1)
    red_idx_f = r_norm * (lut_n_size - 1)

    # Get integer base indices (floor) for the "cube"
    b0 = int(np.floor(blue_idx_f))
    g0 = int(np.floor(green_idx_f))
    r0 = int(np.floor(red_idx_f))

    # Ensure indices are within bounds [0, N-1]
    b0 = np.clip(b0, 0, lut_n_size - 1)
    g0 = np.clip(g0, 0, lut_n_size - 1)
    r0 = np.clip(r0, 0, lut_n_size - 1)

    # Calculate next indices (ceil, but clamped to N-1)
    b1 = min(b0 + 1, lut_n_size - 1)
    g1 = min(g0 + 1, lut_n_size - 1)
    r1 = min(r0 + 1, lut_n_size - 1)

    # Calculate interpolation weights (fractional parts)
    b_lerp = blue_idx_f - b0
    g_lerp = green_idx_f - g0
    r_lerp = red_idx_f - r0
    
    # Fetch 8 corner colors from the LUT strip
    # LUT strip coordinates: x = b_slice_index * N + r_index, y = g_index
    try:
        c000 = lut_data_np[g0, b0 * lut_n_size + r0]
        c100 = lut_data_np[g0, b0 * lut_n_size + r1]
        c010 = lut_data_np[g1, b0 * lut_n_size + r0]
        c110 = lut_data_np[g1, b0 * lut_n_size + r1]

        c001 = lut_data_np[g0, b1 * lut_n_size + r0]
        c101 = lut_data_np[g0, b1 * lut_n_size + r1]
        c011 = lut_data_np[g1, b1 * lut_n_size + r0]
        c111 = lut_data_np[g1, b1 * lut_n_size + r1]
    except IndexError:
        # Fallback for edge cases or if LUT is malformed
        return np.array([r_in, g_in, b_in], dtype=np.uint8)


    # Interpolate along red axis
    c00 = c000 * (1 - r_lerp) + c100 * r_lerp
    c01 = c001 * (1 - r_lerp) + c101 * r_lerp
    c10 = c010 * (1 - r_lerp) + c110 * r_lerp
    c11 = c011 * (1 - r_lerp) + c111 * r_lerp

    # Interpolate along green axis
    c0 = c00 * (1 - g_lerp) + c10 * g_lerp
    c1 = c01 * (1 - g_lerp) + c11 * g_lerp

    # Interpolate along blue axis
    final_color_np = c0 * (1 - b_lerp) + c1 * b_lerp

    return np.clip(final_color_np, 0, 255).astype(np.uint8)


def process_image_chunk(image_chunk_np, lut_data_np, lut_n_size, start_row_index):
    """Processes a horizontal chunk (strip) of the image."""
    chunk_h, chunk_w, _ = image_chunk_np.shape
    output_chunk_np = np.zeros_like(image_chunk_np, dtype=np.uint8)

    for y_chunk in range(chunk_h):
        for x_chunk in range(chunk_w):
            r_in, g_in, b_in = image_chunk_np[y_chunk, x_chunk][:3] # Get RGB, ignore Alpha
            output_chunk_np[y_chunk, x_chunk] = _trilinear_interpolate_pixel(r_in, g_in, b_in, lut_data_np, lut_n_size)
    return start_row_index, output_chunk_np


def apply_lut_to_image_data_parallel(input_image_np, lut_image_pil, progress_queue):
    """
    Applies a LUT to an image using multiprocessing for parallel computation.
    """
    lut_width, lut_height = lut_image_pil.size
    lut_n_size = lut_height # For a N*N x N LUT strip, N is the height

    if lut_width != lut_n_size * lut_n_size:
        raise ValueError(f"LUT dimensions are incorrect. Expected width {lut_n_size*lut_n_size} for height {lut_n_size}, got {lut_width}.")

    lut_data_np = np.array(lut_image_pil.convert("RGB")) # Ensure RGB
    input_h, input_w, _ = input_image_np.shape
    output_image_np = np.zeros_like(input_image_np, dtype=np.uint8)

    num_processes = os.cpu_count() or 1 # Use available CPUs, default to 1
    rows_per_chunk = math.ceil(input_h / num_processes)
    # Ensure chunks are not excessively small, which can hurt performance due to overhead
    if rows_per_chunk < 10 and input_h > 10 * num_processes : 
        rows_per_chunk = 10
    
    num_chunks_actual = math.ceil(input_h / rows_per_chunk)


    futures = []
    total_processed_rows = 0
    # progress_queue.put(("total_chunks", num_chunks_actual)) # For progress bar max value if needed

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(0, input_h, rows_per_chunk):
            start_row = i
            end_row = min(i + rows_per_chunk, input_h)
            image_chunk = input_image_np[start_row:end_row, :, :]
            futures.append(executor.submit(process_image_chunk, image_chunk, lut_data_np, lut_n_size, start_row))

        for future in concurrent.futures.as_completed(futures):
            try:
                start_idx, processed_chunk = future.result()
                end_idx = start_idx + processed_chunk.shape[0]
                output_image_np[start_idx:end_idx, :, :] = processed_chunk
                total_processed_rows += processed_chunk.shape[0]
                progress_queue.put(("progress", total_processed_rows / input_h * 100))
            except Exception as e:
                progress_queue.put(("error", f"Error in worker process: {e}"))
                # Decide how to handle partial failures
                raise # Re-raise to be caught by the calling worker thread

    return output_image_np

# --- GUI Application ---

class OptimizedLutApplicatorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Optimized LUT Applicator Tool")
        self.root.geometry("900x700")

        self.input_image_pil = None
        self.lut_image_pil = None
        self.processed_image_pil = None

        self.input_image_path_var = tk.StringVar()
        self.lut_image_path_var = tk.StringVar()
        
        self.worker_queue = queue.Queue()

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(root_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_frame, padding="5")
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="Load Input Image...", command=self.load_input_image_action).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Entry(controls_frame, textvariable=self.input_image_path_var, state="readonly", width=40).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(controls_frame, text="Load LUT File (.png)...", command=self.load_lut_action).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Entry(controls_frame, textvariable=self.lut_image_path_var, state="readonly", width=40).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        controls_frame.columnconfigure(1, weight=1)

        action_buttons_frame = ttk.Frame(main_frame, padding="5")
        action_buttons_frame.pack(fill=tk.X, pady=10)
        self.apply_button = ttk.Button(action_buttons_frame, text="Apply LUT to Image", command=self.start_apply_lut_task, state=tk.DISABLED)
        self.apply_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.save_button = ttk.Button(action_buttons_frame, text="Save Processed Image...", command=self.save_processed_action, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        preview_frame = ttk.Frame(main_frame, padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        ttk.Label(preview_frame, text="Original Image", font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, pady=(0,5))
        self.original_canvas = tk.Canvas(preview_frame, bg="gray80", relief=tk.SUNKEN, borderwidth=1)
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.original_photo_preview = None
        ttk.Label(preview_frame, text="Processed Image (with LUT)", font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, pady=(0,5))
        self.processed_canvas = tk.Canvas(preview_frame, bg="gray80", relief=tk.SUNKEN, borderwidth=1)
        self.processed_canvas.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.processed_photo_preview = None

        progress_status_frame = ttk.Frame(main_frame)
        progress_status_frame.pack(fill=tk.X, pady=(5,0))
        self.progress_bar = ttk.Progressbar(progress_status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_label = ttk.Label(progress_status_frame, text="Load an input image and a LUT file to begin.", width=40, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.root.after(100, self.process_worker_queue)

    def process_worker_queue(self):
        try:
            while True:
                message_type, data = self.worker_queue.get_nowait()
                if message_type == "progress":
                    self.progress_bar["value"] = data
                    self.status_label.config(text=f"Processing... {data:.1f}%")
                elif message_type == "result":
                    self.processed_image_pil = data
                    self._update_canvas(self.processed_canvas, self.processed_image_pil, "processed_photo_preview")
                    self.save_button.config(state=tk.NORMAL)
                    self.status_label.config(text="LUT applied successfully.")
                    self.apply_button.config(state=tk.NORMAL)
                    self.progress_bar["value"] = 100 # Ensure it hits 100
                elif message_type == "error":
                    messagebox.showerror("Processing Error", str(data))
                    self.status_label.config(text="Error during processing.")
                    self.apply_button.config(state=tk.NORMAL)
                    self.progress_bar["value"] = 0
                elif message_type == "status_update":
                    self.status_label.config(text=str(data))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_worker_queue)

    def _update_canvas(self, canvas, pil_image, photo_ref_attr_name):
        if pil_image is None:
            canvas.delete("all")
            setattr(self, photo_ref_attr_name, None)
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, lambda: self._update_canvas(canvas, pil_image, photo_ref_attr_name))
            return

        img_copy = pil_image.copy()
        original_width, original_height = img_copy.size
        img_aspect = original_width / original_height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            new_width = canvas_width
            new_height = int(new_width / img_aspect)
        else:
            new_height = canvas_height
            new_width = int(new_height * img_aspect)
        
        if new_width > 0 and new_height > 0:
            img_copy = img_copy.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_copy)
            setattr(self, photo_ref_attr_name, photo)
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo)
        else:
            canvas.delete("all")
            setattr(self, photo_ref_attr_name, None)


    def _check_enable_apply_button(self):
        if self.input_image_pil and self.lut_image_pil:
            self.apply_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready to apply LUT.")
        else:
            self.apply_button.config(state=tk.DISABLED)

    def load_input_image_action(self):
        path = filedialog.askopenfilename(title="Select Input Image", filetypes=(("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")))
        if path:
            try:
                self.input_image_pil = Image.open(path).convert("RGB")
                self.input_image_path_var.set(os.path.basename(path))
                self._update_canvas(self.original_canvas, self.input_image_pil, "original_photo_preview")
                self.processed_image_pil = None
                self._update_canvas(self.processed_canvas, None, "processed_photo_preview")
                self.save_button.config(state=tk.DISABLED)
                self.status_label.config(text="Input image loaded.")
                self.progress_bar["value"] = 0
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not load input image:\n{e}")
                self.input_image_pil = None; self.input_image_path_var.set("")
            self._check_enable_apply_button()

    def load_lut_action(self):
        path = filedialog.askopenfilename(title="Select LUT File", filetypes=(("PNG LUT files", "*.png"), ("All files", "*.*")))
        if path:
            try:
                self.lut_image_pil = Image.open(path).convert("RGB")
                w, h = self.lut_image_pil.size
                if h not in [16, 32, 64] or w != h*h: # Basic validation
                     messagebox.showwarning("LUT Warning", f"LUT dimensions ({w}x{h}) seem unusual. Expected N*NxN (e.g., 256x16, 1024x32).")
                self.lut_image_path_var.set(os.path.basename(path))
                self.status_label.config(text="LUT file loaded.")
                self.progress_bar["value"] = 0
            except Exception as e:
                messagebox.showerror("LUT Error", f"Could not load LUT file:\n{e}")
                self.lut_image_pil = None; self.lut_image_path_var.set("")
            self._check_enable_apply_button()

    def _apply_lut_worker(self):
        if not self.input_image_pil or not self.lut_image_pil:
            self.worker_queue.put(("error", "Input image or LUT not loaded."))
            return

        self.worker_queue.put(("status_update", "Preparing to process..."))
        self.worker_queue.put(("progress", 0))
        
        try:
            input_np = np.array(self.input_image_pil)
            
            self.worker_queue.put(("status_update", "Processing with multiple cores..."))
            start_time = time.time()
            processed_np = apply_lut_to_image_data_parallel(input_np, self.lut_image_pil, self.worker_queue)
            end_time = time.time()
            self.worker_queue.put(("status_update", f"Processing took {end_time - start_time:.2f} seconds."))

            if processed_np is not None:
                result_image_pil = Image.fromarray(processed_np)
                self.worker_queue.put(("result", result_image_pil))
            # else: error would have been put on queue by apply_lut_to_image_data_parallel

        except ValueError as ve: # For LUT dimension errors primarily
            self.worker_queue.put(("error", str(ve)))
        except Exception as e: # Catch-all for unexpected errors in this worker thread
            import traceback
            tb_str = traceback.format_exc()
            self.worker_queue.put(("error", f"An unexpected error occurred in worker thread:\n{e}\n{tb_str}"))

    def start_apply_lut_task(self):
        if not self.input_image_pil or not self.lut_image_pil:
            messagebox.showerror("Error", "Please load both an input image and a LUT file.")
            return
        
        self.apply_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.processed_image_pil = None
        self._update_canvas(self.processed_canvas, None, "processed_photo_preview")
        self.progress_bar["value"] = 0
        self.status_label.config(text="Starting processing...")

        thread = threading.Thread(target=self._apply_lut_worker, daemon=True)
        thread.start()

    def save_processed_action(self):
        if not self.processed_image_pil:
            messagebox.showwarning("No Image", "No processed image to save.")
            return
        original_filename = os.path.splitext(self.input_image_path_var.get())[0]
        suggested_filename = f"{original_filename}_lutted.png" # Default to original name + suffix
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")), title="Save Processed Image As", initialfile=suggested_filename)
        if file_path:
            try:
                self.processed_image_pil.save(file_path)
                messagebox.showinfo("Saved", f"Processed image saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save processed image:\n{e}")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    
    root = tk.Tk()
    app = OptimizedLutApplicatorApp(root)
    # Initial canvas updates after GUI is a bit more settled
    root.update_idletasks() # Ensure canvas dimensions are known
    root.after(100, lambda: app._update_canvas(app.original_canvas, None, "original_photo_preview"))
    root.after(100, lambda: app._update_canvas(app.processed_canvas, None, "processed_photo_preview"))
    root.mainloop()