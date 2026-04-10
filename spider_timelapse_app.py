import os
import shutil
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
 
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
 
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
 
# ── palette (classic gray desktop) ───────────────────────────────────────────
BG, PANEL, ACCENT = "#ececec", "#f3f3f3", "#dfdfdf"
GREEN, BLUE, ORANGE, PURPLE = "#d9d9d9", "#d9d9d9", "#d9d9d9", "#d9d9d9"
FG, MUTED, STEP_HDR = "#1f1f1f", "#5f5f5f", "#e6e6e6"
ENTRY_BG, LOG_BG, LOG_FG = "#ffffff", "#000000", "#FFD04D"
BTN_BG, BTN_ACTIVE, BTN_FG = "#dadada", "#cfcfcf", "#111111"
 
PREVIEW_MAX_W, PREVIEW_MAX_H = 1280, 720   # preview window cap
 
 
def _make_writer(path, fps, w, h):
    """
    Create a writer with Windows-safe fallbacks.
    Returns (writer, actual_output_path, fourcc_str) or (None, None, None).
    """
    base, _ = os.path.splitext(path)
    candidates = [
        (path, "mp4v"),          # safest MP4 codec for many OpenCV Windows builds
        (f"{base}.avi", "MJPG"), # fallback when MP4 codecs are unavailable
    ]

    for out_path, fourcc_str in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, out_path, fourcc_str
    return None, None, None
 
 
class SpiderTimelapse(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spider Timelapse Detector")
        self.geometry("800x780")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._icon_img = None
 
        # shared state
        self.s1_src_folder  = tk.StringVar()
        self.date_prefix    = tk.StringVar(value="9.21.25")
        self.s2_img_folder  = tk.StringVar()
        self.fps            = tk.IntVar(value=5)
        self.model_path     = tk.StringVar(value="runs/detect/train/weights/best.pt")
        self.s3_video_path  = tk.StringVar()
 
        self.output_folder  = None
        self.timelapse_path = None
        self.labeled_path   = None
 
        self._build()
        self._set_app_icon()
        self._check_deps()
 
    # ─────────────────────────────────────────────────────────────────────────
    # UI helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=STEP_HDR, pady=1)
        outer.pack(fill="x", pady=5)
        tk.Label(outer, text=title, font=("Courier New", 11, "bold"),
                 bg=STEP_HDR, fg=FG).pack(anchor="w", padx=10, pady=4)
        inner = tk.Frame(outer, bg=PANEL, padx=10, pady=8)
        inner.pack(fill="x", padx=1, pady=(0, 1))
        return inner
 
    def _row(self, p):
        r = tk.Frame(p, bg=PANEL)
        r.pack(fill="x", pady=2)
        return r
 
    def _btn(self, p, txt, cmd, color=ACCENT, state="normal"):
        b = tk.Button(p, text=txt, command=cmd, bg=BTN_BG, fg=BTN_FG,
                      relief="raised", bd=1, font=("Segoe UI", 9),
                      activebackground=BTN_ACTIVE, activeforeground=BTN_FG,
                      padx=10, pady=4, state=state)
        b.pack(side="left", padx=4, pady=4)
        return b
 
    def _entry(self, p, var, width=36):
        e = tk.Entry(p, textvariable=var, width=width,
                     bg=ENTRY_BG, fg=FG, insertbackground=FG,
                     relief="flat", font=("Courier New", 9))
        e.pack(side="left", padx=4)
        return e
 
    def _lbl(self, p, txt):
        tk.Label(p, text=txt, bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
 
    def _log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.update_idletasks()
 
    def _unlock(self, *btns):
        for b in btns:
            b.config(state="normal")

    def _validate_video_output(self, video_path):
        """
        Ensure the output file is non-empty and decodable.
        """
        if not os.path.exists(video_path):
            return False, "Output file was not created."
        if os.path.getsize(video_path) <= 0:
            return False, "Output file is empty."

        cap = cv2.VideoCapture(video_path)
        ok, _ = cap.read()
        cap.release()
        if not ok:
            return False, "Output file was created but could not be decoded."
        return True, ""
 
    def _check_deps(self):
        if not CV2_OK:
            self._log("⚠  opencv-python not found.  Run: pip install opencv-python")
        if not YOLO_OK:
            self._log("⚠  ultralytics not found.    Run: pip install ultralytics")

    def _set_app_icon(self):
        """
        Use a spider PNG icon if available. Keep a reference to prevent GC.
        """
        icon_candidates = [
            os.path.join(os.path.dirname(__file__), "icon-spider.png"),
            r"C:\Users\ChichiPepe\.cursor\projects\c-Users-ChichiPepe-Desktop-Collab3D-ZooMiamiTDspider\assets\c__Users_ChichiPepe_AppData_Roaming_Cursor_User_workspaceStorage_eda72ce8b0bbaafb462b43aa7c2eaaca_images_image-e10f9c60-6389-4070-9019-e3de8f4d9206.png",
        ]
        for icon_path in icon_candidates:
            if os.path.exists(icon_path):
                try:
                    self._icon_img = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, self._icon_img)
                    return
                except Exception:
                    continue
 
    # ─────────────────────────────────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        hdr = tk.Frame(self, bg=ACCENT, height=56)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Spider Timelapse Detector",
                 font=("Segoe UI", 15, "bold"),
                 bg=ACCENT, fg=FG).pack(side="left", padx=18, pady=10)
 
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=8)
 
        self._step1_ui(body)
        self._step2_ui(body)
        self._step3_ui(body)
        self._step4_ui(body)
 
        tk.Label(body, text="LOG", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.log_box = tk.Text(body, height=13, bg=LOG_BG, fg=LOG_FG,
                               font=("Courier New", 8), relief="flat",
                               state="disabled", wrap="word")
        self.log_box.pack(fill="x")
 
    # ── Step 1 ────────────────────────────────────────────────────────────────
    def _step1_ui(self, p):
        c = self._card(p, "  STEP 1  Rename Images by Date / Time")
 
        r1 = self._row(c)
        self._lbl(r1, "Image Folder:")
        self._entry(r1, self.s1_src_folder)
        self._btn(r1, "Browse", lambda: self._browse_dir(self.s1_src_folder), BLUE)
 
        r2 = self._row(c)
        self._lbl(r2, "Date Prefix (e.g. 9.21.25):")
        self._entry(r2, self.date_prefix, width=12)
 
        r3 = self._row(c)
        self._btn(r3, "Rename Images", self._rename_images, GREEN)
 
    # ── Step 2 ────────────────────────────────────────────────────────────────
    def _step2_ui(self, p):
        c = self._card(p, "  STEP 2  Create Timelapse Video")
 
        r1 = self._row(c)
        self._lbl(r1, "Image Folder:")
        self._entry(r1, self.s2_img_folder)
        self._btn(r1, "Browse", lambda: self._browse_dir(self.s2_img_folder), BLUE)
 
        r2 = self._row(c)
        self._lbl(r2, "Frame Rate (FPS):")
        tk.Spinbox(r2, from_=1, to=30, textvariable=self.fps, width=5,
                   bg=ENTRY_BG, fg=FG, relief="flat",
                   font=("Courier New", 9)).pack(side="left", padx=4)
 
        r3 = self._row(c)
        self._btn(r3, "Create Timelapse", self._create_timelapse, BLUE)
 
    # ── Step 3 ────────────────────────────────────────────────────────────────
    def _step3_ui(self, p):
        c = self._card(p, "  STEP 3  Run Spider Detection on Video")
 
        r1 = self._row(c)
        self._lbl(r1, "Model (.pt):")
        self._entry(r1, self.model_path)
        self._btn(r1, "Browse", self._browse_model, BLUE)
 
        r2 = self._row(c)
        self._lbl(r2, "Video (.mp4):")
        self._entry(r2, self.s3_video_path)
        self._btn(r2, "Browse", self._browse_video, BLUE)
 
        r3 = self._row(c)
        self.s3_btn = self._btn(r3, "Run Detection", self._run_detection, ORANGE)
 
    # ── Step 4 ────────────────────────────────────────────────────────────────
    def _step4_ui(self, p):
        c = self._card(p, "  STEP 4  Preview Labeled Video")
 
        r1 = self._row(c)
        self.prev_btn = self._btn(r1, "Preview Labeled Video",
                                  self._preview_video, PURPLE, state="disabled")
        tk.Label(r1, text="(press Q inside the window to close)",
                 bg=PANEL, fg=MUTED, font=("Courier New", 8)).pack(side="left", padx=6)
 
    # ─────────────────────────────────────────────────────────────────────────
    # BROWSE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _browse_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)
 
    def _browse_model(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt")])
        if p:
            self.model_path.set(p)
 
    def _browse_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("MP4 video", "*.mp4"), ("All video", "*.mp4 *.avi *.mov")])
        if p:
            self.s3_video_path.set(p)
 
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 - RENAME
    # ─────────────────────────────────────────────────────────────────────────
    def _rename_images(self):
        src    = self.s1_src_folder.get().strip()
        prefix = self.date_prefix.get().strip()
 
        if not src or not os.path.isdir(src):
            messagebox.showerror("Error", "Please select a valid image folder."); return
        if not prefix:
            messagebox.showerror("Error", "Please enter a date prefix."); return
 
        # folder named after the prefix
        self.output_folder = os.path.join(src, prefix)
        os.makedirs(self.output_folder, exist_ok=True)
 
        exts  = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        files = []
        for root, _, fnames in os.walk(src):
            if os.path.abspath(root) == os.path.abspath(self.output_folder):
                continue
            for fn in fnames:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(root, fn))
 
        if not files:
            messagebox.showerror("Error", "No image files found."); return
 
        files.sort(key=lambda p: os.path.getmtime(p))
 
        self._log(f"Found {len(files)} images. Renaming into '{prefix}' folder...")
        for i, fpath in enumerate(files, 1):
            suffix   = Path(fpath).suffix.lower()
            new_name = f"{prefix}_{i:05d}{suffix}"
            new_path = os.path.join(self.output_folder, new_name)
            shutil.copy2(fpath, new_path)
            if i % 50 == 0 or i == len(files):
                self._log(f"  Copied {i}/{len(files)}: {new_name}")
 
        self._log(f"Done! Output folder: {self.output_folder}")
        messagebox.showinfo("Step 1 Complete",
                            f"{len(files)} images renamed.\nFolder: {self.output_folder}")
 
        # auto-fill Step 2 folder
        self.s2_img_folder.set(self.output_folder)
 
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 - TIMELAPSE
    # ─────────────────────────────────────────────────────────────────────────
    def _create_timelapse(self):
        if not CV2_OK:
            messagebox.showerror("Error", "opencv-python is not installed."); return
 
        img_dir = self.s2_img_folder.get().strip()
        prefix  = self.date_prefix.get().strip()
 
        if not img_dir or not os.path.isdir(img_dir):
            messagebox.showerror("Error", "Please select a valid image folder."); return
        if not prefix:
            messagebox.showerror("Error", "Please set a date prefix in Step 1."); return
 
        exts   = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([f for f in os.listdir(img_dir)
                         if f.lower().endswith(exts)])
 
        if not images:
            messagebox.showerror("Error", "No images found in the selected folder."); return
 
        first = cv2.imread(os.path.join(img_dir, images[0]))
        if first is None:
            messagebox.showerror("Error", "Could not read the first image."); return
 
        h, w = first.shape[:2]
 
        # video name: "{prefix} Spider Cam.mp4"
        video_name          = f"{prefix} Spider Cam.mp4"
        self.timelapse_path = os.path.join(img_dir, video_name)
 
        writer, actual_path, used_codec = _make_writer(self.timelapse_path, self.fps.get(), w, h)
        if writer is None:
            messagebox.showerror("Error", "Could not open a video writer."); return
        self.timelapse_path = actual_path
 
        self._log(f"Building '{video_name}' from {len(images)} frames at {self.fps.get()} FPS...")
        written = 0
        for idx, name in enumerate(images, 1):
            frame = cv2.imread(os.path.join(img_dir, name))
            if frame is not None:
                writer.write(frame)
                written += 1
            if idx % 50 == 0 or idx == len(images):
                self._log(f"  Written {idx}/{len(images)} frames")
 
        writer.release()
        ok, err = self._validate_video_output(self.timelapse_path)
        if not ok:
            messagebox.showerror("Error", f"Video creation failed: {err}"); return

        self._log(f"Codec used: {used_codec} | Frames written: {written}")
        self._log(f"Done! Video saved: {self.timelapse_path}")
        messagebox.showinfo("Step 2 Complete",
                            f"Timelapse created!\n{self.timelapse_path}")
 
        # auto-fill Step 3 video path
        self.s3_video_path.set(self.timelapse_path)
 
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 - DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    def _run_detection(self):
        if not CV2_OK or not YOLO_OK:
            messagebox.showerror("Error", "opencv-python and ultralytics must be installed.")
            return
 
        video_in   = self.s3_video_path.get().strip()
        model_file = self.model_path.get().strip()
        prefix     = self.date_prefix.get().strip()
 
        if not video_in or not os.path.exists(video_in):
            messagebox.showerror("Error", "Please select a valid input video."); return
        if not os.path.exists(model_file):
            messagebox.showerror("Error", "Model file not found. Check the path."); return
        if not prefix:
            messagebox.showerror("Error", "Please set a date prefix in Step 1."); return
 
        # output saved next to the input video
        out_dir           = os.path.dirname(video_in)
        out_name          = f"{prefix} Spider Cam AI.mp4"
        self.labeled_path = os.path.join(out_dir, out_name)
 
        self.s3_btn.config(state="disabled")
        self._log(f"Loading model...  output: {out_name}")
 
        def _work():
            try:
                model = YOLO(model_file)
                cap   = cv2.VideoCapture(video_in)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_v = cap.get(cv2.CAP_PROP_FPS) or self.fps.get()
 
                writer, actual_path, used_codec = _make_writer(self.labeled_path, fps_v, w, h)
                if writer is None:
                    self.after(0, messagebox.showerror, "Error",
                               "Could not open video writer.")
                    self.after(0, self._unlock, self.s3_btn)
                    return
                self.labeled_path = actual_path
 
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results   = model(frame, verbose=False)
                    annotated = results[0].plot()
                    writer.write(annotated)
                    count += 1
                    if count % 10 == 0 or count == total:
                        self.after(0, self._log,
                                   f"  Detected frame {count}/{total}")
 
                cap.release()
                writer.release()
                ok, err = self._validate_video_output(self.labeled_path)
                if not ok:
                    self.after(0, messagebox.showerror, "Error",
                               f"Detection video failed: {err}")
                    self.after(0, self._unlock, self.s3_btn)
                    return

                self.after(0, self._log, f"Codec used: {used_codec}")
                self.after(0, self._log, f"Done! Saved: {self.labeled_path}")
                self.after(0, messagebox.showinfo, "Step 3 Complete",
                           f"Detection finished!\nSaved as:\n{self.labeled_path}")
                self.after(0, self._unlock, self.prev_btn, self.s3_btn)
 
            except Exception as exc:
                self.after(0, messagebox.showerror, "Detection Error", str(exc))
                self.after(0, self._unlock, self.s3_btn)
 
        threading.Thread(target=_work, daemon=True).start()
 
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 - PREVIEW
    # ─────────────────────────────────────────────────────────────────────────
    def _preview_video(self):
        if not self.labeled_path or not os.path.exists(self.labeled_path):
            messagebox.showerror("Error", "Complete Step 3 first."); return
 
        self._log("Opening preview... press Q to close.")
 
        def _play():
            cap   = cv2.VideoCapture(self.labeled_path)
            fps_v = cap.get(cv2.CAP_PROP_FPS) or self.fps.get()
            delay = max(1, int(1000 / fps_v))
 
            win = "Labeled Timelapse  (press Q to quit)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
 
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
 
                # scale down to fit inside PREVIEW_MAX_W x PREVIEW_MAX_H
                fh, fw = frame.shape[:2]
                scale  = min(PREVIEW_MAX_W / fw, PREVIEW_MAX_H / fh, 1.0)
                if scale < 1.0:
                    frame = cv2.resize(frame,
                                       (int(fw * scale), int(fh * scale)),
                                       interpolation=cv2.INTER_AREA)
 
                cv2.imshow(win, frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
 
            cap.release()
            cv2.destroyAllWindows()
 
        threading.Thread(target=_play, daemon=True).start()
 
 
if __name__ == "__main__":
    app = SpiderTimelapse()
    app.mainloop()