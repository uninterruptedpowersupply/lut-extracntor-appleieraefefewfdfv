import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

# Attempt to import scikit-image for histogram matching
try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image library not found. 'Create from Reference Image' feature will be disabled.")
    print("You can install it by running: pip install scikit-image")

# --- Core LUT Logic ---

def generate_neutral_lut_image(lut_size=32):
    """
    Generates a neutral LUT strip image.
    The image will be (lut_size * lut_size) pixels wide and lut_size pixels high.
    Common sizes for lut_size: 16, 32, 64.
    e.g., for lut_size=16, image is 256x16.
    e.g., for lut_size=32, image is 1024x32.
    """
    width = lut_size * lut_size
    height = lut_size
    lut_image = Image.new("RGB", (width, height))
    pixels = lut_image.load()

    for b_slice in range(lut_size):  # Blue varies with each "slice" or "depth"
        for g_row in range(lut_size):   # Green varies with Y (row in the slice)
            for r_col in range(lut_size): # Red varies with X (column in the slice)
                # Normalize values to 0-255 range
                r = int((r_col / (lut_size - 1)) * 255)
                g = int((g_row / (lut_size - 1)) * 255)
                b = int((b_slice / (lut_size - 1)) * 255)

                # Calculate pixel coordinates in the final LUT strip
                x = b_slice * lut_size + r_col
                y = g_row
                pixels[x, y] = (r, g, b)
    return lut_image

def create_lut_from_reference(reference_image_path, lut_size=32):
    """
    Generates a neutral LUT and then attempts to match its histogram
    to the reference image.
    Returns the graded LUT image or None on failure.
    """
    if not SKIMAGE_AVAILABLE:
        messagebox.showerror("Dependency Missing",
                             "The 'scikit-image' library is required for this feature but not found.\n"
                             "Please install it by running: pip install scikit-image")
        return None
    if not os.path.exists(reference_image_path):
        messagebox.showerror("File Error", f"Reference image not found:\n{reference_image_path}")
        return None

    try:
        reference_img = Image.open(reference_image_path).convert("RGB")
    except Exception as e:
        messagebox.showerror("Image Error", f"Could not open reference image:\n{e}")
        return None

    neutral_lut_img = generate_neutral_lut_image(lut_size)

    # Convert images to numpy arrays for scikit-image
    neutral_lut_np = np.array(neutral_lut_img)
    reference_np = np.array(reference_img)

    try:
        # Match histograms
        # `channel_axis=-1` for RGB images (last axis)
        matched_lut_np = match_histograms(neutral_lut_np, reference_np, channel_axis=-1)

        # Convert back to PIL Image, ensuring data type is uint8 for image saving
        matched_lut_img = Image.fromarray(matched_lut_np.astype(np.uint8))
        return matched_lut_img
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error during histogram matching:\n{e}")
        return None

# --- GUI Application ---

class LutCreatorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Unity LUT Creator")
        self.root.geometry("700x750")

        self.current_lut_image_pil = None
        # This is the correct Tkinter variable for LUT size, initialized to 32
        self.current_lut_size_var = tk.IntVar(value=32)
        self.reference_image_path_var = tk.StringVar()

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(root_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)

        self.tab_neutral = ttk.Frame(self.notebook, padding="10")
        self.tab_from_ref = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_neutral, text='  Generate Neutral LUT  ')
        self.notebook.add(self.tab_from_ref, text='  Create from Reference Image  ')

        self._create_neutral_lut_tab_content()
        self._create_from_reference_tab_content()

        preview_save_frame = ttk.LabelFrame(main_frame, text="LUT Preview & Save", padding="10")
        preview_save_frame.pack(pady=10, padx=5, fill=tk.X)

        self.preview_status_label = ttk.Label(preview_save_frame, text="Generate or load a LUT to see preview.")
        self.preview_status_label.pack(pady=(0,5))

        self.image_preview_canvas = tk.Canvas(preview_save_frame, width=512, height=64, bg="gray80", relief=tk.SUNKEN, borderwidth=1)
        self.image_preview_canvas.pack(pady=5)
        self.displayed_photo_preview = None

        save_button = ttk.Button(preview_save_frame, text="Save Current LUT As (.PNG)...", command=self.save_lut_action)
        save_button.pack(pady=(10,0))

        instructions_frame = ttk.LabelFrame(main_frame, text="How to use in Unity", padding="10")
        instructions_frame.pack(pady=10, padx=5, fill=tk.X)
        instructions_text = (
            "1. Save the generated LUT as a .PNG file.\n"
            "2. Import the .PNG into your Unity project.\n"
            "3. Select the imported texture. In the Inspector:\n"
            "   - Texture Type: Default (or Advanced for more control).\n"
            "   - sRGB (Color Texture): Uncheck if your LUT is authored in linear space (usually it's sRGB).\n"
            "   - Read/Write Enabled: Not usually needed for Post-Processing Stack v2 LUTs.\n"
            "     (Enable if sampling manually in a custom shader needing CPU access).\n"
            "   - Generate Mip Maps: Disable.\n"
            "   - Wrap Mode: Clamp.\n"
            "   - Filter Mode: Bilinear (recommended for LUTs).\n"
            "   - Compression: None (or a lossless format like RGBA32).\n"
            "4. Apply using Unity's Post-Processing (URP/HDRP or built-in):\n"
            "   - Add a 'Color Lookup' (URP/HDRP) or 'Color Grading' (Built-in PPv2) effect.\n"
            "   - Mode (if applicable for Color Grading): External / Lookup Texture.\n"
            "   - Lookup Texture: Drag your imported LUT texture here.\n"
            "   - Adjust 'Contribution' (or Intensity/Blend) as needed."
        )
        ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT, wraplength=650).pack(anchor="w")


    def _create_neutral_lut_tab_content(self):
        frame = self.tab_neutral

        ttk.Label(frame, text="LUT Size (N):", font=('Segoe UI', 10)).grid(row=0, column=0, padx=5, pady=10, sticky="w")
        size_options = [16, 32, 64]
        # Combobox uses self.current_lut_size_var which is already set to 32 by default
        self.neutral_lut_size_combo = ttk.Combobox(frame, textvariable=self.current_lut_size_var, values=size_options, state="readonly", width=5, font=('Segoe UI', 10))
        # The line self.neutral_lut_size_var.set(32) was here and caused the error; it's removed as it's redundant.
        self.neutral_lut_size_combo.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        
        # Initialize label text based on the default value of current_lut_size_var
        initial_size = self.current_lut_size_var.get()
        self.neutral_lut_size_label = ttk.Label(frame, text=f"(Creates {initial_size*initial_size}x{initial_size} LUT)", font=('Segoe UI', 9))
        self.neutral_lut_size_label.grid(row=0, column=2, padx=5, pady=10, sticky="w")
        self.neutral_lut_size_combo.bind("<<ComboboxSelected>>", self._update_neutral_lut_size_label_text)


        generate_neutral_button = ttk.Button(frame, text="Generate Neutral LUT", command=self.generate_neutral_lut_action)
        generate_neutral_button.grid(row=1, column=0, columnspan=3, pady=15, sticky="ew")

        info_text = (
            "This generates a 'neutral' (identity) LUT strip. \n"
            "You typically open this neutral LUT in an image editor (Photoshop, GIMP, Krita), \n"
            "apply your desired color grading adjustments to IT, and then save the modified LUT.\n"
            "This modified LUT is what you then use in Unity."
        )
        ttk.Label(frame, text=info_text, justify=tk.LEFT, wraplength=600, font=('Segoe UI', 9)).grid(row=2, column=0, columnspan=3, pady=10, sticky="w")

    def _update_neutral_lut_size_label_text(self, event=None): # Renamed for clarity
        try:
            size = self.current_lut_size_var.get() # Use the correct var
            self.neutral_lut_size_label.config(text=f"(Creates {size*size}x{size} LUT)")
        except tk.TclError: # Catch error if var is empty or invalid during update
            self.neutral_lut_size_label.config(text="(Invalid size)")


    def _create_from_reference_tab_content(self):
        frame = self.tab_from_ref

        ref_img_frame = ttk.Frame(frame)
        ref_img_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")
        ttk.Label(ref_img_frame, text="Reference Image:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0,5))
        self.ref_image_entry = ttk.Entry(ref_img_frame, textvariable=self.reference_image_path_var, width=50, font=('Segoe UI', 9))
        self.ref_image_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        browse_button = ttk.Button(ref_img_frame, text="Browse...", command=self.browse_reference_image_action)
        browse_button.pack(side=tk.LEFT, padx=5)

        ttk.Label(frame, text="Output LUT Size (N):", font=('Segoe UI', 10)).grid(row=1, column=0, padx=5, pady=10, sticky="w")
        size_options = [16, 32, 64]
        self.ref_lut_size_combo = ttk.Combobox(frame, values=size_options, state="readonly", width=5, font=('Segoe UI', 10))
        self.ref_lut_size_combo.set(32)
        self.ref_lut_size_combo.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        
        initial_ref_size = int(self.ref_lut_size_combo.get()) # Get initial value for label
        self.ref_lut_size_label = ttk.Label(frame, text=f"(Creates {initial_ref_size*initial_ref_size}x{initial_ref_size} LUT)", font=('Segoe UI', 9))
        self.ref_lut_size_label.grid(row=1, column=2, padx=5, pady=10, sticky="w")
        self.ref_lut_size_combo.bind("<<ComboboxSelected>>", self._update_ref_lut_size_label_text)


        generate_from_ref_button = ttk.Button(frame, text="Generate LUT from Reference", command=self.generate_from_reference_action)
        generate_from_ref_button.grid(row=2, column=0, columnspan=3, pady=15, sticky="ew")

        info_text = (
            "This attempts to create a LUT by matching the color histogram of a neutral LUT to your chosen reference image. "
            "This is an automated approach and might not perfectly capture complex artistic grades. "
            "It works best for overall color shifts and tonal adjustments. \n"
            "Requires 'scikit-image' library (pip install scikit-image)."
        )
        ttk.Label(frame, text=info_text, justify=tk.LEFT, wraplength=600, font=('Segoe UI', 9)).grid(row=3, column=0, columnspan=3, pady=10, sticky="w")
        if not SKIMAGE_AVAILABLE:
            ttk.Label(frame, text="WARNING: scikit-image not found. This feature is disabled.", foreground="red", font=('Segoe UI', 9, 'bold')).grid(row=4, column=0, columnspan=3, pady=5, sticky="w")

    def _update_ref_lut_size_label_text(self, event=None): # Renamed for clarity
        try:
            size_str = self.ref_lut_size_combo.get()
            if size_str:
                size = int(size_str)
                self.ref_lut_size_label.config(text=f"(Creates {size*size}x{size} LUT)")
            else:
                self.ref_lut_size_label.config(text="(Select size)")
        except ValueError:
            self.ref_lut_size_label.config(text="(Invalid size)")


    def update_preview(self):
        if self.current_lut_image_pil:
            img_copy = self.current_lut_image_pil.copy()
            
            original_width, original_height = img_copy.size
            canvas_width = self.image_preview_canvas.winfo_width()
            canvas_height = self.image_preview_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <=1:
                canvas_width = 512
                canvas_height = 64

            img_aspect = original_width / original_height
            canvas_aspect = canvas_width / canvas_height

            if img_aspect > canvas_aspect:
                new_width = canvas_width
                new_height = int(new_width / img_aspect)
            else:
                new_height = canvas_height
                new_width = int(new_height * img_aspect)
            
            if new_width > 0 and new_height > 0: # Ensure dimensions are positive
                img_copy = img_copy.resize((new_width, new_height), Image.LANCZOS)
                self.displayed_photo_preview = ImageTk.PhotoImage(img_copy)
                self.image_preview_canvas.delete("all")
                self.image_preview_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.displayed_photo_preview)
                self.preview_status_label.config(text=f"Current LUT ({original_width}x{original_height}):")
            else: # Fallback if calculated dimensions are zero or negative
                self.image_preview_canvas.delete("all")
                self.preview_status_label.config(text="Error displaying preview (invalid size).")
                self.displayed_photo_preview = None

        else:
            self.image_preview_canvas.delete("all")
            self.preview_status_label.config(text="Generate or load a LUT to see preview.")
            self.displayed_photo_preview = None


    def generate_neutral_lut_action(self):
        try:
            lut_n_size = self.current_lut_size_var.get() # Use the correct var
        except tk.TclError:
            messagebox.showerror("Error", "Please select a valid LUT size.")
            return
            
        self.current_lut_image_pil = generate_neutral_lut_image(lut_n_size)
        self.root.after(100, self.update_preview)
        messagebox.showinfo("Success", f"Neutral {lut_n_size*lut_n_size}x{lut_n_size} LUT generated. You can now save it.")

    def browse_reference_image_action(self):
        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=(("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*"))
        )
        if path:
            self.reference_image_path_var.set(path)

    def generate_from_reference_action(self):
        if not SKIMAGE_AVAILABLE:
            messagebox.showerror("Dependency Missing",
                                 "The 'scikit-image' library is required for this feature but not found.\n"
                                 "Please install it by running: pip install scikit-image")
            return

        ref_path = self.reference_image_path_var.get()
        if not ref_path:
            messagebox.showerror("Input Error", "Please select a reference image.")
            return

        try:
            lut_n_size_str = self.ref_lut_size_combo.get()
            if not lut_n_size_str:
                 messagebox.showerror("Input Error", "Please select an output LUT size.")
                 return
            lut_n_size = int(lut_n_size_str)
        except ValueError:
            messagebox.showerror("Input Error", "Invalid LUT size selected.")
            return

        self.current_lut_image_pil = create_lut_from_reference(ref_path, lut_n_size)

        if self.current_lut_image_pil:
            self.root.after(100, self.update_preview)
            messagebox.showinfo("Success", f"LUT generated from reference image. You can now save it.")

    def save_lut_action(self):
        if not self.current_lut_image_pil:
            messagebox.showwarning("No LUT", "No LUT has been generated yet.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("All files", "*.*")),
            title="Save LUT As",
            initialfile="MyLUT.png"
        )
        if file_path:
            try:
                self.current_lut_image_pil.save(file_path, "PNG")
                messagebox.showinfo("Saved", f"LUT saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save LUT:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LutCreatorApp(root)
    root.mainloop()