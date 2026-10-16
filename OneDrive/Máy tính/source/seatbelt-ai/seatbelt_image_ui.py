from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR/ "models"/ "seatbelt-seg-v2"/ "weights"/ "best_roboflow.pt")

CONFIDENCE = 0.20
CONF_CHAC = 0.50
PANEL_MAX_WIDTH = 650
PANEL_MAX_HEIGHT = 650


class SeatbeltImageUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seatbelt AI - So sanh anh")
        self.root.geometry("1450x850")
        self.root.minsize(900, 600)

        self.image_path = None
        self.original_pil = None
        self.result_pil = None
        self.original_photo = None
        self.result_photo = None
        self.model = None
        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg="#17212d")

        header = tk.Frame(self.root, bg="#17212d")
        header.pack(fill="x", padx=18, pady=(16, 10))

        tk.Label(
            header,
            text="Seatbelt AI",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#17212d",
        ).pack()

        tk.Label(
            header,
            text="Chon anh de xem mask phan doan day an toan",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#17212d",
        ).pack(pady=(3, 0))

        controls = tk.Frame(self.root, bg="#17212d")
        controls.pack(fill="x", padx=18, pady=8)

        tk.Button(
            controls,
            text="Chon anh",
            command=self.choose_image,
            font=("Segoe UI", 11, "bold"),
            bg="#2478ed",
            fg="white",
            activebackground="#1765d1",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=9,
            cursor="hand2",
        ).pack(side="left")

        self.path_var = tk.StringVar(value="Chua chon anh")
        tk.Label(
            controls,
            textvariable=self.path_var,
            anchor="w",
            font=("Segoe UI", 10),
            fg="#e2e8f0",
            bg="#243244",
            padx=10,
            pady=10,
        ).pack(side="left", fill="x", expand=True, padx=10)

        tk.Button(
            controls,
            text="Phan tich",
            command=self.analyze_image,
            font=("Segoe UI", 11, "bold"),
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=9,
            cursor="hand2",
        ).pack(side="right")

        panels = tk.Frame(self.root, bg="#17212d")
        panels.pack(fill="both", expand=True, padx=18, pady=(8, 10))
        panels.columnconfigure(0, weight=1, uniform="panel")
        panels.columnconfigure(1, weight=1, uniform="panel")
        panels.rowconfigure(1, weight=1)

        tk.Label(
            panels,
            text="Anh goc",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#243244",
            padx=10,
            pady=8,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        tk.Label(
            panels,
            text="Anh sau khi to mask seatbelt",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#243244",
            padx=10,
            pady=8,
            anchor="w",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.original_panel = tk.Label(
            panels,
            text="Chon mot anh de bat dau",
            font=("Segoe UI", 12),
            fg="#94a3b8",
            bg="#0f1720",
            relief="flat",
        )
        self.original_panel.grid(
            row=1, column=0, sticky="nsew", padx=(0, 8), pady=(6, 0)
        )

        self.result_panel = tk.Label(
            panels,
            text="Anh ket qua se hien thi tai day",
            font=("Segoe UI", 12),
            fg="#94a3b8",
            bg="#0f1720",
            relief="flat",
        )
        self.result_panel.grid(
            row=1, column=1, sticky="nsew", padx=(8, 0), pady=(6, 0)
        )

        self.status_var = tk.StringVar(value="San sang.")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#17212d",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 12))

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chon anh can kiem tra",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Loi", "Khong doc duoc anh nay.")
            return

        self.image_path = Path(path)
        self.path_var.set(str(self.image_path))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_pil = Image.fromarray(rgb)
        self.result_pil = self.original_pil.copy()

        self._show_images()
        self.status_var.set("Da chon anh. Bam 'Phan tich' de chay model.")

    def _load_model(self):
        if self.model is not None:
            return True

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Khong tim thay model",
                "Khong tim thay file model:\n"
                f"{MODEL_PATH}\n\n"
                "Hay dat file UI nay trong thu muc goc seatbelt-ai.",
            )
            return False

        self.status_var.set("Dang tai model YOLO...")
        self.root.update_idletasks()
        try:
            self.model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            messagebox.showerror("Loi tai model", str(exc))
            self.status_var.set("Tai model that bai.")
            return False

        return True

    def analyze_image(self):
        if self.image_path is None:
            messagebox.showinfo("Chua co anh", "Hay bam 'Chon anh' truoc.")
            return

        if not self._load_model():
            return

        image_bgr = cv2.imread(str(self.image_path))
        if image_bgr is None:
            messagebox.showerror("Loi", "Khong doc lai duoc anh da chon.")
            return

        self.status_var.set("Dang phan tich anh...")
        self.root.update_idletasks()

        try:
            predictions = self.model.predict(
                source=image_bgr,
                conf=CONFIDENCE,
                retina_masks=True,
                verbose=False,
            )
            prediction = predictions[0]
            output = image_bgr.copy()
            detected_count = 0

            if prediction.masks is not None and prediction.boxes is not None:
                masks = prediction.masks.data.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy()
                confidences = prediction.boxes.conf.cpu().numpy()

                for i, class_id in enumerate(classes):
                    class_name = self.model.names[int(class_id)]
                    if class_name != "seatbelt":
                        continue

                    mask = masks[i]
                    if mask.shape != image_bgr.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (image_bgr.shape[1], image_bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    binary_mask = mask > 0.5
                    if not np.any(binary_mask):
                        continue

                    # To mau xanh ban trong tren vung mask
                    green = np.zeros_like(output)
                    green[:] = (0, 255, 0)  # BGR
                    blended = cv2.addWeighted(output, 0.55, green, 0.45, 0)
                    output[binary_mask] = blended[binary_mask]

                    # Ve vien mask de de quan sat
                    contours, _ = cv2.findContours(
                        binary_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
                    detected_count += 1

                    print(
                        f"[IMAGE DEBUG] candidate={i}, "
                        f"confidence={float(confidences[i]):.3f}, "
                        f"class={class_name}"
                    )

            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            self.result_pil = Image.fromarray(output_rgb)
            self._show_images()

            if detected_count:
                self.status_var.set(
                    f"Model tra ve {detected_count} mask seatbelt. "
                    "Hay kiem tra mask co bam dung day that khong."
                )
            else:
                self.status_var.set(
                    "Khong co mask seatbelt nao vuot nguong confidence "
                    f"{CONFIDENCE:.2f}."
                )

        except Exception as exc:
            messagebox.showerror("Loi phan tich", str(exc))
            self.status_var.set("Phan tich that bai.")

    def _fit_image(self, pil_image, max_width, max_height):
        image = pil_image.copy()
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return image

    def _show_images(self):
        if self.original_pil is None:
            return

        self.root.update_idletasks()
        panel_width = max(300, self.original_panel.winfo_width() - 20)
        panel_height = max(250, self.original_panel.winfo_height() - 20)

        original_display = self._fit_image(
            self.original_pil,
            min(PANEL_MAX_WIDTH, panel_width),
            min(PANEL_MAX_HEIGHT, panel_height),
        )
        result_source = self.result_pil or self.original_pil
        result_display = self._fit_image(
            result_source,
            min(PANEL_MAX_WIDTH, panel_width),
            min(PANEL_MAX_HEIGHT, panel_height),
        )

        self.original_photo = ImageTk.PhotoImage(original_display)
        self.result_photo = ImageTk.PhotoImage(result_display)

        self.original_panel.configure(image=self.original_photo, text="")
        self.result_panel.configure(image=self.result_photo, text="")


def main():
    root = tk.Tk()
    SeatbeltImageUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
