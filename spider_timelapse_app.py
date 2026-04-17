import os
import shutil
import threading
import math
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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError:
    SEABORN_OK = False
 
# ── palette (classic gray desktop) ───────────────────────────────────────────
BG, PANEL, ACCENT = "#ececec", "#f3f3f3", "#dfdfdf"
GREEN, BLUE, ORANGE, PURPLE = "#d9d9d9", "#d9d9d9", "#d9d9d9", "#d9d9d9"
FG, MUTED, STEP_HDR = "#1f1f1f", "#5f5f5f", "#e6e6e6"
ENTRY_BG, LOG_BG, LOG_FG = "#ffffff", "#000000", "#FFD04D"
BTN_BG, BTN_ACTIVE, BTN_FG = "#dadada", "#cfcfcf", "#111111"
 
OUTPUT_W, OUTPUT_H = 1280, 720
GRAPH_TYPES = ["Line Graph", "Bar Chart", "Scatter Plot"]
TARGET_CLASSES = [
    "NG Flatworm",
    "TD Maintenance",
    "TD Peeking",
    "TD Silking",
    "TD Spider",
]
 
if CV2_OK:
    # Prefer OpenCV optimized code paths where available.
    cv2.setUseOptimized(True)

 
def _make_writer(path, fps, w, h):
    """
    Create an MP4 writer with conservative codec fallbacks.
    Returns (writer, actual_output_path, fourcc_str) or (None, None, None).
    """
    base, _ = os.path.splitext(path)
    out_path = f"{base}.mp4"
    safe_fps = max(1.0, float(fps or 1.0))
    candidates = ["mp4v", "avc1", "H264", "XVID"]

    for fourcc_str in candidates:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, safe_fps, (w, h), True)
        if writer.isOpened():
            return writer, out_path, fourcc_str
        writer.release()
    return None, None, None
 
 
class SpiderTimelapse(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TrapdoorSpider: AI Timelapse Detector")
        self.geometry("860x820")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._icon_img = None
 
        # shared state
        self.s1_src_folder  = tk.StringVar()
        self.date_prefix    = tk.StringVar(value="")
        self.s2_img_folder  = tk.StringVar()
        self.fps            = tk.IntVar(value=5)
        self.model_path     = tk.StringVar(value="runs/detect/train/weights/best.pt")
        self.s3_video_path  = tk.StringVar()
        self.s4_image_path  = tk.StringVar()
        self.graph_type_var = tk.StringVar(value=GRAPH_TYPES[0])
        self.graph_name_var = tk.StringVar(value="")
 
        self.output_folder  = None
        self.timelapse_path = None
        self.labeled_path   = None
        self.labeled_image_path = None
        self.labeled_image = None
        self.graph_path     = None
        self.class_counts   = {name: 0 for name in TARGET_CLASSES}
        self.per_frame_counts = None
        self.graph_data = None
        self.graph_type_combo = None
        self.graph_name_entry = None
        self.selected_graph_type = GRAPH_TYPES[0]
        self.selected_graph_name = ""

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

    def _resize_if_needed(self, frame, w, h):
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        if fw == w and fh == h:
            return frame
        # INTER_LINEAR is noticeably faster than INTER_AREA for this workflow.
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def _clear_all(self):
        self.s1_src_folder.set("")
        self.date_prefix.set("")
        self.s2_img_folder.set("")
        self.fps.set(5)
        self.model_path.set("runs/detect/train/weights/best.pt")
        self.s3_video_path.set("")
        self.s4_image_path.set("")
        self.graph_type_var.set(GRAPH_TYPES[0])
        self.graph_name_var.set("")
        self.selected_graph_type = GRAPH_TYPES[0]
        self.selected_graph_name = ""

        self.output_folder = None
        self.timelapse_path = None
        self.labeled_path = None
        self.labeled_image_path = None
        self.labeled_image = None
        self.graph_path = None
        self.class_counts = {name: 0 for name in TARGET_CLASSES}
        self.per_frame_counts = None
        self.graph_data = None

        if self.graph_type_combo is not None:
            self.graph_type_combo.set(GRAPH_TYPES[0])
        if self.graph_name_entry is not None:
            self.graph_name_entry.delete(0, "end")

        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self._update_count_display()

        if hasattr(self, "prev_btn"):
            self.prev_btn.config(state="disabled")
        if hasattr(self, "s4_detect_btn"):
            self.s4_detect_btn.config(state="normal")
        if hasattr(self, "s4_preview_btn"):
            self.s4_preview_btn.config(state="disabled")
        if hasattr(self, "s4_download_btn"):
            self.s4_download_btn.config(state="disabled")
        if hasattr(self, "graph_btn"):
            self.graph_btn.config(state="disabled")
        if hasattr(self, "s3_btn"):
            self.s3_btn.config(state="normal")

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
        if not MATPLOTLIB_OK:
            self._log("⚠  matplotlib not found.    Run: pip install matplotlib")

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
        tk.Label(hdr, text="TrapdoorSpider: AI Timelapse Detector",
                 font=("Segoe UI", 15, "bold"),
                 bg=ACCENT, fg=FG).pack(side="left", padx=18, pady=10)
        tk.Button(
            hdr,
            text="Clear",
            command=self._clear_all,
            bg=BTN_BG,
            fg=BTN_FG,
            relief="raised",
            bd=1,
            font=("Segoe UI", 9, "bold"),
            activebackground=BTN_ACTIVE,
            activeforeground=BTN_FG,
            padx=10,
            pady=4,
        ).pack(side="right", padx=12, pady=10)
 
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        left_canvas = tk.Canvas(left, bg=BG, highlightthickness=0, bd=0)
        left_scroll = tk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        left_content = tk.Frame(left_canvas, bg=BG)
        left_window = left_canvas.create_window((0, 0), window=left_content, anchor="nw")

        def _update_scroll_region(_event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _resize_left_content(_event):
            left_canvas.itemconfigure(left_window, width=_event.width)

        def _on_mousewheel(_event):
            # Use Windows wheel deltas; negative means scroll down.
            left_canvas.yview_scroll(int(-1 * (_event.delta / 120)), "units")

        left_content.bind("<Configure>", _update_scroll_region)
        left_canvas.bind("<Configure>", _resize_left_content)
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        right = tk.Frame(main, bg=BG, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._step1_ui(left_content)
        self._step2_ui(left_content)
        self._step3_ui(left_content)
        self._step4_ui(left_content)
        self._step5_ui(left_content)

        tk.Label(right, text="OUTPUT LOG", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.log_box = tk.Text(right, height=11, bg=LOG_BG, fg=LOG_FG,
                               font=("Courier New", 9), relief="flat",
                               state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)
 
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
        c = self._card(p, "  STEP 3  AI Run Spider Detection on Video")
 
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
        c = self._card(p, "  STEP 4  Preview Labeled Video & Image")
 
        r1 = self._row(c)
        self.prev_btn = self._btn(r1, "Preview Labeled Video",
                                  self._preview_video, PURPLE, state="disabled")
        tk.Label(r1, text="(press Q inside the window to close)",
                 bg=PANEL, fg=MUTED, font=("Courier New", 8)).pack(side="left", padx=6)

        r2 = self._row(c)
        self._lbl(r2, "Image:")
        self._entry(r2, self.s4_image_path)
        self._btn(r2, "Browse", self._browse_image, BLUE)

        r3 = self._row(c)
        self.s4_detect_btn = self._btn(r3, "Detect On Image", self._detect_image, ORANGE)
        self.s4_preview_btn = self._btn(
            r3, "Preview Labeled Image", self._preview_labeled_image, PURPLE, state="disabled"
        )
        self.s4_download_btn = self._btn(
            r3, "Download Labeled Image", self._download_labeled_image, GREEN, state="disabled"
        )

        r4 = self._row(c)
        tk.Label(r4, text="(image preview opens in a new window)",
                 bg=PANEL, fg=MUTED, font=("Courier New", 8)).pack(side="left", padx=6)

        r5 = self._row(c)
        self._lbl(r5, "Detection Classification Counts:")
        self.count_box = tk.Text(
            c,
            height=7,
            bg=LOG_BG,
            fg=LOG_FG,
            font=("Courier New", 9),
            relief="flat",
            state="disabled",
            wrap="word",
        )
        self.count_box.pack(fill="x", padx=4, pady=(2, 0))

    # ── Step 5 ────────────────────────────────────────────────────────────────
    def _step5_ui(self, p):
        c = self._card(p, "  STEP 5  Classification Graph")
        r1 = self._row(c)
        self._lbl(r1, "Graph Type:")
        self.graph_type_combo = ttk.Combobox(
            r1,
            textvariable=self.graph_type_var,
            values=GRAPH_TYPES,
            width=16,
            state="readonly",
        )
        self.graph_type_combo.pack(side="left", padx=4)
        self.graph_type_combo.bind("<<ComboboxSelected>>", self._on_graph_type_changed)

        r2 = self._row(c)
        self._lbl(r2, "Graph Name (required):")
        self.graph_name_entry = self._entry(r2, self.graph_name_var, width=24)
        self.graph_name_entry.bind("<KeyRelease>", self._on_graph_name_changed)
        self.graph_name_entry.bind("<FocusOut>", self._on_graph_name_changed)

        r3 = self._row(c)
        self.graph_btn = self._btn(
            r3, "Download Graph", self._download_graph, PURPLE, state="disabled"
        )
        tk.Label(
            r3,
            text="(enabled after detection + graph generation)",
            bg=PANEL,
            fg=MUTED,
            font=("Courier New", 8),
        ).pack(side="left", padx=6)

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

    def _browse_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if p:
            self.s4_image_path.set(p)

    def _to_label(self, model, cls_id):
        names = getattr(model, "names", {})
        name = str(cls_id)
        if isinstance(names, dict):
            name = str(names.get(cls_id, cls_id))
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            name = str(names[cls_id])
        return name

    def _match_target_class(self, label):
        """Map model class string to a TARGET_CLASSES entry (handles case/spacing mismatches)."""
        raw = str(label or "").strip()
        if raw in TARGET_CLASSES:
            return raw
        rl = raw.lower()
        for t in TARGET_CLASSES:
            if t.lower() == rl:
                return t
        return None

    def _safe_name(self, text):
        raw = str(text or "").strip()
        if not raw:
            return ""
        bad_chars = '<>:"/\\|?*'
        cleaned = "".join("_" if ch in bad_chars else ch for ch in raw).strip()
        cleaned = cleaned.rstrip(". ")
        return cleaned or ""

    def _on_graph_type_changed(self, _event=None):
        if self.graph_type_combo is not None:
            selected = self.graph_type_combo.get().strip()
            if selected in GRAPH_TYPES:
                self.graph_type_var.set(selected)
                self.selected_graph_type = selected

    def _on_graph_name_changed(self, _event=None):
        typed = ""
        if self.graph_name_entry is not None:
            typed = self.graph_name_entry.get()
        self.selected_graph_name = self._safe_name(typed)

    def _graph_title_for_export(self):
        typed = ""
        if self.graph_name_entry is not None:
            typed = self.graph_name_entry.get()
        if not str(typed or "").strip():
            typed = self.graph_name_var.get()
        safe = self._safe_name(typed)
        return safe if safe else None

    def _get_graph_type(self):
        selected = ""
        if self.graph_type_combo is not None:
            selected = self.graph_type_combo.get().strip()
        if not selected:
            selected = self.graph_type_var.get().strip()
        if selected in GRAPH_TYPES:
            self.selected_graph_type = selected
        return self.selected_graph_type

    def _update_count_display(self):
        lines = [f"{label}: {self.class_counts[label]}" for label in TARGET_CLASSES]
        self.count_box.config(state="normal")
        self.count_box.delete("1.0", "end")
        self.count_box.insert("end", "\n".join(lines))
        self.count_box.config(state="disabled")

    def _generate_frequency_graph(self, out_dir, graph_data, save_path=None):
        if not MATPLOTLIB_OK:
            return None
        graph_type = self._get_graph_type()
        user_name = self._graph_title_for_export()
        if not user_name:
            return None
        graph_name = f"{user_name} {graph_type}.png"
        graph_path = save_path or os.path.join(out_dir, graph_name)

        if not graph_data:
            return None
        per_frame_presence = graph_data.get("per_frame_presence", {})
        per_frame_avg_conf = graph_data.get("per_frame_avg_conf", {})
        per_frame_counts = graph_data.get("per_frame_counts", {})
        total_entries = graph_data.get("total_entries") or {}
        detection_points = graph_data.get("detection_points", [])

        x_vals = list(range(1, len(per_frame_presence.get(TARGET_CLASSES[0], [])) + 1))
        if not x_vals:
            return None

        if SEABORN_OK:
            sns.set_style("whitegrid")
        plt.figure(figsize=(11, 5))
        if graph_type == "Line Graph":
            for label in TARGET_CLASSES:
                series = per_frame_avg_conf.get(label, [])
                plt.plot(x_vals, series, label=label, linewidth=1.5)
            plt.ylabel("Average Confidence per Frame")
            plt.xlabel("Frame Index")
        elif graph_type == "Bar Chart":
            # Match the "Detection Classification Counts" box exactly.
            totals = [
                int(total_entries.get(label, self.class_counts.get(label, 0)))
                for label in TARGET_CLASSES
            ]
            bars = plt.bar(TARGET_CLASSES, totals, color="#6baed6", edgecolor="#333333", linewidth=0.5)
            for bar, t in zip(bars, totals):
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h,
                    str(int(t)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.ylabel("Detection classification count")
            plt.xlabel("Class")
            plt.xticks(rotation=20, ha="right")
        elif graph_type == "Scatter Plot":
            for label in TARGET_CLASSES:
                pts = [p for p in detection_points if p["label"] == label]
                if not pts:
                    continue
                xs = [p["frame"] for p in pts]
                ys = [p["confidence"] for p in pts]
                sizes = [max(10.0, min(300.0, p["bbox_area"] / 200.0)) for p in pts]
                plt.scatter(xs, ys, s=sizes, alpha=0.45, label=label)
            plt.xlabel("Frame Index")
            plt.ylabel("Confidence")
        else:
            for label in TARGET_CLASSES:
                series = per_frame_avg_conf.get(label, [])
                plt.plot(x_vals, series, label=label, linewidth=1.5)
            plt.ylabel("Average Confidence per Frame")
            plt.xlabel("Frame Index")
        plt.title(user_name)
        if graph_type != "Bar Chart":
            plt.legend(loc="upper right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150)
        plt.close()
        return graph_path

    def _download_graph(self):
        if not self.graph_data:
            messagebox.showerror("Error", "No graph data found. Run detection first."); return
        graph_title = self._graph_title_for_export()
        if not graph_title:
            messagebox.showerror("Error", "Please enter a graph name in Step 5.")
            return
        self.update_idletasks()
        graph_type = self._get_graph_type()
        dst = filedialog.asksaveasfilename(
            title="Save Classification Graph",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{graph_title} {graph_type}.png",
        )
        if not dst:
            return
        out_dir = os.path.dirname(dst) or os.getcwd()
        saved_graph = self._generate_frequency_graph(out_dir, self.graph_data, save_path=dst)
        if not saved_graph or not os.path.exists(saved_graph):
            messagebox.showerror("Error", "Failed to generate the graph for download."); return
        self.graph_path = saved_graph
        messagebox.showinfo("Saved", f"Graph saved to:\n{dst}")
 
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
 
        # video name: "{prefix} Spider Cam.mp4"
        video_name          = f"{prefix} Spider Cam.mp4"
        self.timelapse_path = os.path.join(img_dir, video_name)
 
        fps_out = max(1, int(self.fps.get()))
        writer, actual_path, used_codec = _make_writer(
            self.timelapse_path, fps_out, OUTPUT_W, OUTPUT_H
        )
        if writer is None:
            messagebox.showerror("Error", "Could not open a video writer."); return
        self.timelapse_path = actual_path
 
        self._log(f"Building '{video_name}' from {len(images)} frames at {fps_out} FPS...")
        written = 0
        for idx, name in enumerate(images, 1):
            frame = cv2.imread(os.path.join(img_dir, name))
            if frame is not None:
                frame = self._resize_if_needed(frame, OUTPUT_W, OUTPUT_H)
                writer.write(frame)
                written += 1
            if idx % 200 == 0 or idx == len(images):
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
    # STEP 3 - AI DETECTION
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
        self.graph_path   = None
        self.class_counts = {name: 0 for name in TARGET_CLASSES}
        self.per_frame_counts = None
        self.graph_data = None
        self.graph_btn.config(state="disabled")
        self._update_count_display()
 
        self.s3_btn.config(state="disabled")
        self._log(f"Loading model...  output: {out_name}")
 
        def _work():
            try:
                model = YOLO(model_file)
                cap   = cv2.VideoCapture(video_in)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v = cap.get(cv2.CAP_PROP_FPS)
                if not isinstance(fps_v, (int, float)) or not math.isfinite(fps_v) or fps_v <= 0:
                    fps_v = max(1, int(self.fps.get()))
 
                writer, actual_path, used_codec = _make_writer(
                    self.labeled_path, fps_v, OUTPUT_W, OUTPUT_H
                )
                if writer is None:
                    self.after(0, messagebox.showerror, "Error",
                               "Could not open video writer.")
                    self.after(0, self._unlock, self.s3_btn)
                    return
                self.labeled_path = actual_path
 
                count = 0
                per_frame_counts = {name: [] for name in TARGET_CLASSES}
                per_frame_presence = {name: [] for name in TARGET_CLASSES}
                per_frame_avg_conf = {name: [] for name in TARGET_CLASSES}
                conf_sum_by_class = {name: 0.0 for name in TARGET_CLASSES}
                conf_n_by_class = {name: 0 for name in TARGET_CLASSES}
                detection_points = []
                prev_present = {name: False for name in TARGET_CLASSES}
                co_detect_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_in = self._resize_if_needed(frame, OUTPUT_W, OUTPUT_H)
                    results = model(frame_in, verbose=False, imgsz=640)
                    annotated = results[0].plot()
                    writer.write(annotated)
                    count += 1

                    frame_counts = {name: 0 for name in TARGET_CLASSES}
                    frame_conf_sum = {name: 0.0 for name in TARGET_CLASSES}
                    frame_conf_n = {name: 0 for name in TARGET_CLASSES}
                    boxes = results[0].boxes
                    if boxes is not None and boxes.cls is not None:
                        cls_ids = boxes.cls.tolist()
                        conf_vals = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(cls_ids)
                        xyxy_vals = boxes.xyxy.tolist() if boxes.xyxy is not None else []
                        for idx, cls_raw in enumerate(cls_ids):
                            cls_id = int(cls_raw)
                            raw_label = self._to_label(model, cls_id)
                            label = self._match_target_class(raw_label)
                            if label is not None:
                                frame_counts[label] += 1
                                conf = float(conf_vals[idx]) if idx < len(conf_vals) else 0.0
                                frame_conf_sum[label] += conf
                                frame_conf_n[label] += 1
                                conf_sum_by_class[label] += conf
                                conf_n_by_class[label] += 1
                                if idx < len(xyxy_vals):
                                    x1, y1, x2, y2 = xyxy_vals[idx]
                                    area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
                                else:
                                    area = 0.0
                                detection_points.append(
                                    {
                                        "frame": count,
                                        "label": label,
                                        "confidence": conf,
                                        "bbox_area": area,
                                    }
                                )

                    # Entry count: only count when class transitions
                    # from absent in previous frame to present now.
                    for label in TARGET_CLASSES:
                        is_present = frame_counts[label] > 0
                        if is_present and not prev_present[label]:
                            self.class_counts[label] += 1
                        prev_present[label] = is_present
                        per_frame_presence[label].append(1 if is_present else 0)
                        if frame_conf_n[label] > 0:
                            per_frame_avg_conf[label].append(frame_conf_sum[label] / frame_conf_n[label])
                        else:
                            per_frame_avg_conf[label].append(0.0)

                    for label in TARGET_CLASSES:
                        per_frame_counts[label].append(frame_counts[label])

                    if frame_counts["TD Spider"] > 0 and frame_counts["NG Flatworm"] > 0:
                        co_detect_frames.append(count)
                        self.after(
                            0,
                            self._log,
                            f"ALERT: TD Spider + NG Flatworm together at frame {count}",
                        )

                    if count % 60 == 0 or count == total:
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
                self.after(0, self._log, "Classification totals:")
                for label in TARGET_CLASSES:
                    self.after(0, self._log, f"  {label}: {self.class_counts[label]}")
                self.after(0, self._update_count_display)

                self.per_frame_counts = per_frame_counts
                avg_conf_by_class = {}
                for label in TARGET_CLASSES:
                    if conf_n_by_class[label] > 0:
                        avg_conf_by_class[label] = conf_sum_by_class[label] / conf_n_by_class[label]
                    else:
                        avg_conf_by_class[label] = 0.0
                total_detections_by_class = {
                    label: int(sum(per_frame_counts[label])) for label in TARGET_CLASSES
                }
                self.graph_data = {
                    "per_frame_counts": per_frame_counts,
                    "per_frame_presence": per_frame_presence,
                    "per_frame_avg_conf": per_frame_avg_conf,
                    "detection_points": detection_points,
                    "avg_conf_by_class": avg_conf_by_class,
                    "total_entries": dict(self.class_counts),
                    "total_detections_by_class": total_detections_by_class,
                }
                if MATPLOTLIB_OK:
                    self.after(0, self._unlock, self.graph_btn)
                    self.after(0, self._log, "Graph data is ready. Click 'Download Classification Graph' to save it.")
                else:
                    self.after(0, self._log, "Graph download is unavailable: matplotlib is not installed.")

                if co_detect_frames:
                    self.after(
                        0,
                        messagebox.showwarning,
                        "Co-Detection Alert",
                        "Detected TD Spider and NG Flatworm together "
                        f"in {len(co_detect_frames)} frame(s).",
                    )

                self.after(0, self._log, f"Done! Saved: {self.labeled_path}")
                self.after(0, messagebox.showinfo, "Step 3 Complete",
                           "Detection finished!\n"
                           f"Saved video:\n{self.labeled_path}\n\n"
                           + "\n".join([f"{k}: {self.class_counts[k]}" for k in TARGET_CLASSES]))
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
            cv2.resizeWindow(win, OUTPUT_W, OUTPUT_H)
 
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
 
                frame = cv2.resize(frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)
 
                cv2.imshow(win, frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
 
            cap.release()
            cv2.destroyAllWindows()
 
        threading.Thread(target=_play, daemon=True).start()

    def _detect_image(self):
        if not CV2_OK or not YOLO_OK:
            messagebox.showerror("Error", "opencv-python and ultralytics must be installed.")
            return

        model_file = self.model_path.get().strip()
        image_path = self.s4_image_path.get().strip()
        if not os.path.exists(model_file):
            messagebox.showerror("Error", "Model file not found. Check the path."); return
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please choose a valid image."); return

        self.s4_detect_btn.config(state="disabled")
        self.s4_preview_btn.config(state="disabled")
        self.s4_download_btn.config(state="disabled")
        self.labeled_image = None
        self.labeled_image_path = None
        self._log(f"Running image detection: {os.path.basename(image_path)}")

        def _work():
            try:
                model = YOLO(model_file)
                frame = cv2.imread(image_path)
                if frame is None:
                    self.after(0, messagebox.showerror, "Error", "Could not read selected image.")
                    self.after(0, self._unlock, self.s4_detect_btn)
                    return

                results = model(frame, verbose=False, imgsz=640)
                annotated = results[0].plot()
                self.labeled_image = annotated
                self.labeled_image_path = image_path
                self.after(0, self._log, "Image detection complete.")
                self.after(0, self._unlock, self.s4_preview_btn, self.s4_download_btn, self.s4_detect_btn)
            except Exception as exc:
                self.after(0, messagebox.showerror, "Detection Error", str(exc))
                self.after(0, self._unlock, self.s4_detect_btn)

        threading.Thread(target=_work, daemon=True).start()

    def _preview_labeled_image(self):
        if self.labeled_image is None:
            messagebox.showerror("Error", "Run image detection first."); return

        self._log("Opening labeled image preview...")
        win = "Labeled Image Preview  (press any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, self.labeled_image)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    def _download_labeled_image(self):
        if self.labeled_image is None:
            messagebox.showerror("Error", "Run image detection first."); return

        src_name = "labeled_image.png"
        if self.labeled_image_path:
            src_name = f"{Path(self.labeled_image_path).stem}_AI.png"
        dst = filedialog.asksaveasfilename(
            title="Save Labeled Image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg *.jpeg"), ("Bitmap", "*.bmp")],
            initialfile=src_name,
        )
        if not dst:
            return
        ok = cv2.imwrite(dst, self.labeled_image)
        if not ok:
            messagebox.showerror("Error", "Failed to save labeled image."); return
        self._log(f"Labeled image saved: {dst}")
        messagebox.showinfo("Saved", f"Labeled image saved to:\n{dst}")
 
 
if __name__ == "__main__":
    app = SpiderTimelapse()
    app.mainloop()