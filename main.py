import os
import cv2
import json
import numpy as np
import tkinter as tk
import face_recognition
from tkinter import messagebox, simpledialog, filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

# Direktori utama
REGIST_DIR = "faceRegist"
USERS_FILE = "users.json"
SOAL_FILE = "soal.json"
HISTORY_FILE = "history.json"

# Threshold untuk Euclidean distance
THRESHOLD = 0.4

# Warna dan style
BG_COLOR = "#f0f0f0"
SIDEBAR_COLOR = "#2c3e50"
BUTTON_COLOR = "#3498db"
BUTTON_TEXT_COLOR = "white"
TITLE_FONT = ("Helvetica", 16, "bold")
BUTTON_FONT = ("Helvetica", 12)
TEXT_FONT = ("Helvetica", 11)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cek & buat folder utama
os.makedirs(REGIST_DIR, exist_ok=True)
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"users": []}, f)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump({"history": []}, f)

class GAN:
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        
        # Build dan compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                 metrics=['accuracy'])
        
        # Build generator
        self.generator = self.build_generator()
        
        # Generator mengambil noise sebagai input dan menghasilkan gambar
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # Untuk model kombinasi, kita hanya melatih generator
        self.discriminator.trainable = False
        
        # Discriminator mengambil gambar yang dihasilkan sebagai input dan menentukan validitas
        validity = self.discriminator(img)
        
        # Model kombinasi (stacked generator dan discriminator)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                            optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    def build_generator(self):
        model = tf.keras.Sequential()
        
        model.add(layers.Dense(256, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(layers.Reshape(self.img_shape))
        
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential()
        
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def train(self, X_train, epochs, batch_size=128, sample_interval=50):
        # Normalisasi gambar
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        
        # Label valid dan tidak valid
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Pilih batch gambar secara acak
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Generate batch gambar baru
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train generator
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    
    def generate_images(self, n_samples=1):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        return (gen_imgs * 127.5 + 127.5).astype(np.uint8)

class ExamSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EXAMPRO")
        self.root.geometry("1200x700")
        self.root.configure(bg=BG_COLOR)
        
        # Variabel untuk menyimpan state
        self.current_user = None
        self.is_verified = False
        self.verification_hist = None
        self.gan = GAN()  # Inisialisasi GAN
        
        self.setup_gui()
        
    def setup_gui(self):
        # Frame utama
        self.main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)
        
        # Sidebar
        self.sidebar = tk.Frame(self.main_frame, bg=SIDEBAR_COLOR, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo/Title di sidebar
        tk.Label(self.sidebar, text="EXAMPRO", font=TITLE_FONT, 
                bg=SIDEBAR_COLOR, fg="white").pack(pady=20)
        
        # Tombol-tombol menu
        self.create_menu_button("Registrasi", self.register_user)
        self.create_menu_button("Registrasi Ulang", self.reregister_user)
        self.create_menu_button("Verifikasi", self.verify_user)
        self.create_menu_button("Ujian", self.show_exam)
        self.create_menu_button("Histogram", self.show_histogram)
        self.create_menu_button("Riwayat", self.show_history)
        
        # Content area
        self.content_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.content_frame.pack(side="right", fill="both", expand=True)
        
        # Welcome message
        self.welcome_label = tk.Label(self.content_frame, 
                                    text="Selamat Datang di EXAMPRO\nSilakan verifikasi wajah untuk memulai ujian",
                                    font=TITLE_FONT, bg=BG_COLOR)
        self.welcome_label.pack(pady=50)
        
    def create_menu_button(self, text, command):
        btn = tk.Button(self.sidebar, text=text, command=command,
                       bg=SIDEBAR_COLOR, fg="white", font=BUTTON_FONT,
                       width=20, height=2, bd=0,
                       activebackground="#34495e")
        btn.pack(pady=5)
        
    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def load_users(self):
        try:
            with open(USERS_FILE) as f:
                return json.load(f)["users"]
        except:
            return []
            
    def save_user(self, data):
        users = self.load_users()
        users.append(data)
        with open(USERS_FILE, 'w') as f:
            json.dump({"users": users}, f, indent=2)
            
    def enhance_face_image(self, face_img):
        try:
            # Konversi ke grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Resize ke ukuran yang sesuai dengan GAN
            resized = cv2.resize(gray, (64, 64))
            
            # Normalisasi dan reshape untuk input GAN
            normalized = (resized.astype(np.float32) - 127.5) / 127.5
            normalized = normalized.reshape(1, 64, 64, 1)
            
            # Generate gambar yang ditingkatkan
            enhanced = self.gan.generator.predict(normalized)
            
            # Denormalisasi dan konversi ke uint8
            enhanced = (enhanced * 127.5 + 127.5).astype(np.uint8)
            enhanced = enhanced.reshape(64, 64)
            
            # Resize kembali ke ukuran asli
            enhanced = cv2.resize(enhanced, (face_img.shape[1], face_img.shape[0]))
            
            # Konversi kembali ke BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
        except Exception as e:
            print(f"Error dalam peningkatan gambar: {str(e)}")
            return face_img

    def detect_and_crop_face(self, frame, user_name=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        # Ambil wajah terbesar
        max_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = max_face
        
        # Crop wajah dengan ukuran yang sama dengan bounding box
        face_img = gray[y:y+h, x:x+w]
        
        # Resize ke ukuran standar (misalnya 100x100)
        face_img = cv2.resize(face_img, (200, 200))
        
        # Konversi kembali ke BGR untuk ditampilkan
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
        # Gambar bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Tambahkan label nama jika ada
        if user_name:
            cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 255, 0), -1)
            cv2.putText(frame, user_name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return face_img_bgr, frame
            
    def get_histogram(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except:
            return None
            
    def capture_face(self, show_preview=True):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Tidak dapat mengakses kamera!")
            return None
            
        while True:
            ret, frame = cam.read()
            if not ret:
                messagebox.showerror("Error", "Gagal mengambil gambar dari kamera!")
                cam.release()
                cv2.destroyAllWindows()
                return None
                
            # Deteksi dan crop wajah
            face_img, frame_with_box = self.detect_and_crop_face(frame)
            
            if face_img is not None:
                if show_preview:
                    cv2.imshow("Tekan 's' untuk simpan", frame_with_box)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return face_img
                elif key == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return None
            else:
                if show_preview:
                    cv2.imshow("Tekan 's' untuk simpan", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return None
                    
    def register_user(self):
        self.clear_content()
        
        # Form registrasi
        form_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Registrasi Pengguna", font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        
        # Input fields
        fields = {}
        for label in ["Nama", "NIM", "Kelas"]:
            frame = tk.Frame(form_frame, bg=BG_COLOR)
            frame.pack(pady=5)
            tk.Label(frame, text=f"{label}:", width=10, anchor="w", bg=BG_COLOR).pack(side="left")
            entry = tk.Entry(frame, width=30)
            entry.pack(side="left")
            fields[label.lower()] = entry
            
        def submit_registration():
            data = {k: v.get() for k, v in fields.items()}
            if not all(data.values()):
                messagebox.showerror("Error", "Semua field harus diisi!")
                return
                
            folder_name = f"{data['nama']}_{data['nim']}_{data['kelas']}"
            folder_path = os.path.join(REGIST_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            messagebox.showinfo("Instruksi", "Ambil wajah, tekan 's' untuk simpan atau 'q' untuk batal")
            frame = self.capture_face()
            
            if frame is None:
                return
                
            save_path = os.path.join(folder_path, f"{folder_name}_1.jpg")
            cv2.imwrite(save_path, frame)
            
            self.save_user({"nama": data['nama'], "nim": data['nim'], 
                          "kelas": data['kelas'], "folder": folder_name})
            messagebox.showinfo("Sukses", f"User {data['nama']} berhasil didaftarkan!")
            self.clear_content()
            self.welcome_label.pack(pady=50)
            
        tk.Button(form_frame, text="Daftar", command=submit_registration,
                 bg=BUTTON_COLOR, fg="white", font=BUTTON_FONT).pack(pady=20)
                 
    def reregister_user(self):
        if not self.is_verified:
            messagebox.showwarning("Peringatan", "Anda harus verifikasi terlebih dahulu!")
            return
            
        self.clear_content()
        
        # Form registrasi ulang
        form_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Registrasi Ulang Data Diri", font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        
        # Input fields
        fields = {}
        for label in ["Nama", "NIM", "Kelas"]:
            frame = tk.Frame(form_frame, bg=BG_COLOR)
            frame.pack(pady=5)
            tk.Label(frame, text=f"{label}:", width=10, anchor="w", bg=BG_COLOR).pack(side="left")
            entry = tk.Entry(frame, width=30)
            entry.pack(side="left")
            # Isi field dengan data user saat ini
            if label.lower() in self.current_user:
                entry.insert(0, self.current_user[label.lower()])
            fields[label.lower()] = entry
            
        def submit_reregistration():
            data = {k: v.get() for k, v in fields.items()}
            if not all(data.values()):
                messagebox.showerror("Error", "Semua field harus diisi!")
                return
                
            # Konfirmasi perubahan
            if not messagebox.askyesno("Konfirmasi", 
                "Apakah Anda yakin ingin mengubah data diri?\nData lama akan dihapus."):
                return
                
            # Hapus folder lama
            old_folder = os.path.join(REGIST_DIR, self.current_user['folder'])
            if os.path.exists(old_folder):
                for file in os.listdir(old_folder):
                    os.remove(os.path.join(old_folder, file))
                os.rmdir(old_folder)
            
            # Buat folder baru
            folder_name = f"{data['nama']}_{data['nim']}_{data['kelas']}"
            folder_path = os.path.join(REGIST_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Ambil foto baru
            messagebox.showinfo("Instruksi", "Ambil wajah baru, tekan 's' untuk simpan atau 'q' untuk batal")
            frame = self.capture_face()
            
            if frame is None:
                return
                
            # Simpan foto baru
            save_path = os.path.join(folder_path, f"{folder_name}_1.jpg")
            cv2.imwrite(save_path, frame)
            
            # Update data user
            users = self.load_users()
            for i, user in enumerate(users):
                if user['nama'] == self.current_user['nama']:
                    users[i] = {
                        "nama": data['nama'],
                        "nim": data['nim'],
                        "kelas": data['kelas'],
                        "folder": folder_name
                    }
                    break
            
            # Simpan perubahan
            with open(USERS_FILE, 'w') as f:
                json.dump({"users": users}, f, indent=2)
            
            # Update current user
            self.current_user = users[i]
            
            messagebox.showinfo("Sukses", f"Data {data['nama']} berhasil diperbarui!")
            self.clear_content()
            self.welcome_label.config(text=f"Selamat Datang, {data['nama']}!\nSilakan pilih menu Ujian untuk memulai ujian")
            self.welcome_label.pack(pady=50)
            
        tk.Button(form_frame, text="Update Data", command=submit_reregistration,
                 bg=BUTTON_COLOR, fg="white", font=BUTTON_FONT).pack(pady=20)

    def verify_user(self):
        self.clear_content()
        
        # Frame untuk verifikasi
        verify_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        verify_frame.pack(pady=20)
        
        tk.Label(verify_frame, text="Verifikasi Wajah", font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        tk.Label(verify_frame, text="Arahkan wajah ke kamera\nTekan 's' untuk verifikasi atau 'q' untuk batal",
                font=TEXT_FONT, bg=BG_COLOR).pack(pady=10)
                
        def start_verification():
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                messagebox.showerror("Error", "Tidak dapat mengakses kamera!")
                return
                
            while True:
                ret, frame = cam.read()
                if not ret:
                    messagebox.showerror("Error", "Gagal mengambil gambar dari kamera!")
                    cam.release()
                    cv2.destroyAllWindows()
                    return
                    
                # Deteksi dan crop wajah
                face_img, frame_with_box = self.detect_and_crop_face(frame)
                
                if face_img is not None:
                    # Coba verifikasi wajah
                    input_hist = self.get_histogram(face_img)
                    if input_hist is not None:
                        users = self.load_users()
                        best_match = None
                        best_dist = float('inf')
                        
                        for user in users:
                            folder = os.path.join(REGIST_DIR, user['folder'])
                            if os.path.exists(folder):
                                for file in os.listdir(folder):
                                    if file.endswith(('.jpg', '.jpeg', '.png')):
                                        path = os.path.join(folder, file)
                                        ref_img = cv2.imread(path)
                                        if ref_img is not None:
                                            ref_hist = self.get_histogram(ref_img)
                                            if ref_hist is not None:
                                                dist = np.linalg.norm(input_hist - ref_hist)
                                                if dist < best_dist:
                                                    best_dist = dist
                                                    best_match = user
                        
                        # Tampilkan hasil verifikasi
                        if best_match and best_dist < THRESHOLD:
                            face_img, frame_with_box = self.detect_and_crop_face(frame, f"{best_match['nama']} ({best_dist:.4f})")
                        else:
                            face_img, frame_with_box = self.detect_and_crop_face(frame, "Tidak Dikenali")
                
                cv2.imshow("Tekan 's' untuk simpan", frame_with_box if face_img is not None else frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if face_img is not None and best_match and best_dist < THRESHOLD:
                        self.verification_hist = input_hist
                        self.current_user = best_match
                        self.is_verified = True
                        messagebox.showinfo("Sukses", f"Selamat datang, {best_match['nama']}!")
                        self.clear_content()
                        self.welcome_label.config(text=f"Selamat Datang, {best_match['nama']}!\nSilakan pilih menu Ujian untuk memulai ujian")
                        self.welcome_label.pack(pady=50)
                    else:
                        messagebox.showerror("Gagal", "Verifikasi gagal! Wajah tidak dikenali.")
                    cam.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    return
                    
        tk.Button(verify_frame, text="Mulai Verifikasi", command=start_verification,
                 bg=BUTTON_COLOR, fg="white", font=BUTTON_FONT).pack(pady=20)
                 
    def show_exam(self):
        if not self.is_verified:
            messagebox.showwarning("Peringatan", "Anda harus verifikasi terlebih dahulu!")
            return
            
        self.clear_content()
        
        try:
            with open(SOAL_FILE) as f:
                soal = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat soal ujian: {str(e)}")
            return
            
        # Frame untuk ujian
        exam_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        exam_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Canvas dan Scrollbar
        canvas = tk.Canvas(exam_frame, bg=BG_COLOR)
        scrollbar = ttk.Scrollbar(exam_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Judul
        tk.Label(scrollable_frame, text="UJIAN ONLINE", font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        
        # Soal Pilihan Ganda
        tk.Label(scrollable_frame, text="Soal Pilihan Ganda:", 
                font=("Helvetica", 12, "bold"), bg=BG_COLOR).pack(anchor='w', pady=(0,10))
                
        jawaban_pg = []
        for i, s in enumerate(soal["pilihan_ganda"]):
            frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
            frame.pack(fill="x", pady=5)
            
            tk.Label(frame, text=f"{i+1}. {s['pertanyaan']}", 
                    font=TEXT_FONT, bg=BG_COLOR, wraplength=700).pack(anchor='w')
            var = tk.StringVar()
            jawaban_pg.append(var)
            for p in s['pilihan']:
                tk.Radiobutton(frame, text=p, variable=var, value=p, 
                             bg=BG_COLOR, font=TEXT_FONT).pack(anchor='w', padx=20)
                             
        def submit_exam():
            # Hitung nilai
            nilai_pg = 0
            total_bobot_pg = 0
            jawaban_benar = 0
            total_soal = len(soal["pilihan_ganda"])
            
            for i, (var, soal_item) in enumerate(zip(jawaban_pg, soal["pilihan_ganda"])):
                if var.get() == soal_item["kunci_jawaban"]:
                    nilai_pg += soal_item["bobot"]
                    jawaban_benar += 1
                total_bobot_pg += soal_item["bobot"]
                
            nilai_akhir = (nilai_pg / total_bobot_pg) * 100
            
            # Simpan ke history
            try:
                # Baca file history yang ada
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE) as f:
                        history = json.load(f)
                else:
                    history = {"history": []}
                    
                # Tambahkan data baru
                history["history"].append({
                    "nama": self.current_user["nama"],
                    "nim": self.current_user["nim"],
                    "kelas": self.current_user["kelas"],
                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nilai": nilai_akhir,
                    "jawaban_benar": jawaban_benar,
                    "total_soal": total_soal
                })
                
                # Simpan kembali ke file
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(history, f, indent=2)
                    
                # Tampilkan pesan sukses
                messagebox.showinfo("Selesai", 
                                  f"Jawaban telah dikumpulkan!\n\n"
                                  f"Nilai Anda: {nilai_akhir:.2f}\n"
                                  f"Jawaban Benar: {jawaban_benar} dari {total_soal} soal")
                
                # Kembali ke halaman utama
                self.clear_content()
                self.welcome_label.config(text=f"Selamat Datang {self.current_user['nama']}!\nSilakan pilih menu di sidebar")
                self.welcome_label.pack(pady=50)
                
            except Exception as e:
                # Jika terjadi error saat menyimpan, tetap tampilkan hasil
                messagebox.showinfo("Selesai", 
                                  f"Jawaban telah dikumpulkan!\n\n"
                                  f"Nilai Anda: {nilai_akhir:.2f}\n"
                                  f"Jawaban Benar: {jawaban_benar} dari {total_soal} soal\n\n"
                                  f"Catatan: Cek riwayat untuk melihat hasil ujian.")
                
                # Kembali ke halaman utama
                self.clear_content()
                self.welcome_label.config(text=f"Selamat Datang {self.current_user['nama']}!\nSilakan pilih menu di sidebar")
                self.welcome_label.pack(pady=50)
                
        # Tombol Kumpulkan
        submit_btn = tk.Button(scrollable_frame, text="Kumpulkan Jawaban", 
                             command=submit_exam,
                             bg=BUTTON_COLOR, fg="white", font=BUTTON_FONT,
                             padx=20, pady=10)
        submit_btn.pack(pady=20)
        
        # Pack canvas dan scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_pdf_report(self, registered_hist, verification_hist, euclidean_dist, failed_users):
        try:
            # Buat nama file PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"histogram_report_{timestamp}.pdf"
            
            # Buat dokumen PDF
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Judul
            title = Paragraph("Laporan Perhitungan Histogram Wajah", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))
            
            # Informasi User
            user_info = Paragraph(f"User: {self.current_user['nama']}", styles['Heading2'])
            elements.append(user_info)
            elements.append(Spacer(1, 10))
            
            # Jarak Euclidean
            dist_info = Paragraph(f"Jarak Euclidean: {euclidean_dist:.4f}", styles['Normal'])
            elements.append(dist_info)
            elements.append(Spacer(1, 10))

            # Threshold
            dist_info = Paragraph(f"Threshold (toleransi): {THRESHOLD}", styles['Normal'])
            elements.append(dist_info)
            elements.append(Spacer(1, 10))
            
            # Status Verifikasi
            status = "Verifikasi Berhasil" if euclidean_dist < THRESHOLD else "Verifikasi Gagal"
            status_info = Paragraph(f"Status: {status}", styles['Normal'])
            elements.append(status_info)
            elements.append(Spacer(1, 20))
            
            # Tambahkan gambar histogram
            if os.path.exists('histogram.png'):
                from reportlab.platypus import Image as RLImage
                img = RLImage('histogram.png', width=500, height=200)
                elements.append(img)
                elements.append(Spacer(1, 20))

            # Tambahkan tabel perhitungan manual jarak Euclidean
            elements.append(Paragraph("Perhitungan Manual Jarak Euclidean:", styles['Heading3']))
            elements.append(Spacer(1, 10))

            # Buat data untuk tabel
            table_data = [["No.", "x (Matriks A)", "y (Matriks B)", "x-y", "(x-y)²"]]
            sum_squared_diff = 0

            # Tampilkan semua 256 nilai
            for i in range(256):
                x = registered_hist[i]
                y = verification_hist[i]
                diff = x - y
                squared_diff = diff * diff
                sum_squared_diff += squared_diff
                
                table_data.append([
                    str(i+1),
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{diff:.4f}",
                    f"{squared_diff:.4f}"
                ])

            # Tambahkan baris total
            table_data.append(["Total", "", "", "", f"{sum_squared_diff:.4f}"])
            
            # Tambahkan baris hasil akar
            manual_euclidean = np.sqrt(sum_squared_diff)
            table_data.append(["√Total", "", "", "", f"{manual_euclidean:.4f}"])

            # Buat tabel dengan style yang sesuai
            table = Table(table_data, colWidths=[40, 100, 100, 100, 100])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, -2), (-1, -1), colors.lightgrey),
                ('FONTNAME', (0, -2), (-1, -1), 'Helvetica-Bold'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey]),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 20))
            
            # Tambahkan catatan verifikasi
            note = Paragraph(f"Catatan: Perhitungan manual menghasilkan jarak Euclidean = {manual_euclidean:.4f}, sedangkan perhitungan sistem = {euclidean_dist:.4f}. {'Kedua nilai sama' if abs(manual_euclidean - euclidean_dist) < 0.0001 else 'Terdapat perbedaan nilai'}", styles['Normal'])
            elements.append(note)
            elements.append(Spacer(1, 20))
            
            # Matriks Intensitas Pixel Wajah Terdaftar
            elements.append(Paragraph("Matriks Intensitas Pixel (Wajah Terdaftar):", styles['Heading3']))
            registered_matrix = []
            for i in range(0, 256, 16):
                row = [f"{registered_hist[j]:.4f}" for j in range(i, min(i+16, 256))]
                registered_matrix.append(" ".join(row))

            registered_text = "\n".join(registered_matrix)
            elements.append(Paragraph(registered_text, styles['Normal']))
            elements.append(Spacer(1, 20))

            # Matriks Intensitas Pixel Wajah Verifikasi
            elements.append(Paragraph("Matriks Intensitas Pixel (Wajah Verifikasi):", styles['Heading3']))
            verification_matrix = []
            for i in range(0, 256, 16):
                row = [f"{verification_hist[j]:.4f}" for j in range(i, min(i+16, 256))]
                verification_matrix.append(" ".join(row))

            verification_text = "\n".join(verification_matrix)
            elements.append(Paragraph(verification_text, styles['Normal']))
            elements.append(Spacer(1, 20))

            # Perbandingan dengan User Lain
            elements.append(Paragraph("Perbandingan dengan User Lain:", styles['Heading3']))
            other_users_text = []
            for name, _, dist in failed_users:
                other_users_text.append(f"{name}: {dist:.4f}")

            elements.append(Paragraph("\n".join(other_users_text), styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            messagebox.showinfo("Sukses", f"Laporan berhasil disimpan sebagai {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat PDF: {str(e)}")
            return None

    def create_history_pdf(self):
        try:
            # Buat nama file PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"history_report_{timestamp}.pdf"
            
            # Buat dokumen PDF
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Judul
            title = Paragraph("Laporan Riwayat Ujian", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))
            
            # Baca data history
            with open(HISTORY_FILE) as f:
                history = json.load(f)
            
            if not history["history"]:
                elements.append(Paragraph("Belum ada riwayat ujian", styles['Normal']))
            else:
                # Buat tabel history
                history_data = [["Nama", "NIM", "Kelas", "Tanggal", "Nilai", "Jawaban Benar"]]
                for h in history["history"]:
                    history_data.append([
                        h["nama"],
                        h["nim"],
                        h["kelas"],
                        h["tanggal"],
                        f"{h['nilai']:.2f}",
                        f"{h['jawaban_benar']}/{h['total_soal']}"
                    ])
                
                history_table = Table(history_data, colWidths=[100, 80, 80, 120, 60, 80])
                history_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(history_table)
            
            # Build PDF
            doc.build(elements)
            messagebox.showinfo("Sukses", f"Laporan riwayat berhasil disimpan sebagai {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat PDF riwayat: {str(e)}")
            return None

    def create_euclidean_csv(self, registered_hist, verification_hist, euclidean_dist):
        try:
            # Buat nama file CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"euclidean_calculation_{timestamp}.csv"
            
            # Buat data untuk CSV
            sum_squared_diff = 0
            rows = [["No.", "x (Matriks A)", "y (Matriks B)", "x-y", "(x-y)²"]]
            
            # Tampilkan semua 256 nilai
            for i in range(256):
                x = registered_hist[i]
                y = verification_hist[i]
                diff = x - y
                squared_diff = diff * diff
                sum_squared_diff += squared_diff
                
                rows.append([
                    str(i+1),
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{diff:.4f}",
                    f"{squared_diff:.4f}"
                ])
            
            # Tambahkan baris total dan akar
            manual_euclidean = np.sqrt(sum_squared_diff)
            rows.append(["Total", "", "", "", f"{sum_squared_diff:.4f}"])
            rows.append(["√Total", "", "", "", f"{manual_euclidean:.4f}"])
            
            # Tulis ke file CSV
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerows(rows)
            
            messagebox.showinfo("Sukses", f"File CSV berhasil disimpan sebagai {csv_filename}")
            return csv_filename
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat file CSV: {str(e)}")
            return None

    def show_histogram(self):
        if not self.is_verified:
            messagebox.showwarning("Peringatan", "Anda harus verifikasi terlebih dahulu!")
            return
            
        self.clear_content()
        
        # Frame utama untuk histogram
        main_hist_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        main_hist_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Canvas dan Scrollbar vertikal dan horizontal
        canvas = tk.Canvas(main_hist_frame, bg=BG_COLOR)
        v_scrollbar = ttk.Scrollbar(main_hist_frame, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_hist_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Judul
        tk.Label(scrollable_frame, text="Histogram Encoding Wajah", 
                font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        
        try:
            # Buat figure dengan 3 subplot
            plt.figure(figsize=(15, 5), dpi=100)
            
            # Plot 1: Histogram wajah terdaftar
            plt.subplot(1, 3, 1)
            registered_hist = None
            if self.current_user:
                folder = os.path.join(REGIST_DIR, self.current_user['folder'])
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            path = os.path.join(folder, file)
                            ref_img = cv2.imread(path)
                            if ref_img is not None:
                                ref_hist = self.get_histogram(ref_img)
                                if ref_hist is not None:
                                    registered_hist = ref_hist
                                    plt.plot(ref_hist, label=f'Wajah {self.current_user["nama"]}')
                                    break
            
            plt.title(f'Histogram Wajah Terdaftar\n({self.current_user["nama"]})', pad=10)
            plt.xlabel('Intensitas Pixel')
            plt.ylabel('Frekuensi')
            plt.legend(fontsize='small')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Histogram verifikasi
            plt.subplot(1, 3, 2)
            if self.verification_hist is not None:
                plt.plot(self.verification_hist, label=f'Wajah Verifikasi ({self.current_user["nama"]})', color='red')
            
            plt.title(f'Histogram Wajah Verifikasi\n({self.current_user["nama"]})', pad=10)
            plt.xlabel('Intensitas Pixel')
            plt.ylabel('Frekuensi')
            plt.legend(fontsize='small')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 3: Histogram user lain yang gagal
            plt.subplot(1, 3, 3)
            failed_users = []
            shown_users = set()  # Set untuk melacak user yang sudah ditampilkan
            if self.verification_hist is not None:
                users = self.load_users()
                for user in users:
                    if user["nama"] != self.current_user["nama"] and user["nama"] not in shown_users:
                        folder = os.path.join(REGIST_DIR, user['folder'])
                        if os.path.exists(folder):
                            for file in os.listdir(folder):
                                if file.endswith(('.jpg', '.jpeg', '.png')):
                                    path = os.path.join(folder, file)
                                    ref_img = cv2.imread(path)
                                    if ref_img is not None:
                                        ref_hist = self.get_histogram(ref_img)
                                        if ref_hist is not None:
                                            dist = np.linalg.norm(self.verification_hist - ref_hist)
                                            failed_users.append((user["nama"], ref_hist, dist))
                                            shown_users.add(user["nama"])  # Tambahkan user ke set
                                            break
            
            # Urutkan berdasarkan jarak Euclidean
            failed_users.sort(key=lambda x: x[2])
            
            # Tampilkan semua user yang gagal
            colors = plt.cm.rainbow(np.linspace(0, 1, len(failed_users)))
            for i, (name, hist, dist) in enumerate(failed_users):
                plt.plot(hist, label=f'{name} (dist: {dist:.4f})', 
                        color=colors[i], linestyle='--')
            
            plt.title('Histogram User Lain\n(Gagal Verifikasi)', pad=10)
            plt.xlabel('Intensitas Pixel')
            plt.ylabel('Frekuensi')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Simpan plot
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25, right=0.85)
            plt.savefig('histogram.png', dpi=100, bbox_inches='tight', pad_inches=0.5)
            plt.close()
            
            # Frame untuk gambar histogram
            img_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
            img_frame.pack(pady=20)
            
            # Tampilkan gambar
            img = Image.open('histogram.png')
            # Hitung rasio aspek
            width, height = img.size
            aspect_ratio = width / height
            new_height = 400  # Ukuran gambar diperkecil
            new_width = int(new_height * aspect_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(img_frame, image=photo, bg=BG_COLOR)
            label.image = photo
            label.pack()
            
            # Frame untuk informasi dan matriks
            info_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
            info_frame.pack(fill="x", pady=10)
            
            # Hitung dan tampilkan jarak Euclidean
            if registered_hist is not None and self.verification_hist is not None:
                euclidean_dist = np.linalg.norm(registered_hist - self.verification_hist)
                
                # Frame untuk informasi jarak
                dist_frame = tk.Frame(info_frame, bg=BG_COLOR)
                dist_frame.pack(side="left", padx=20)
                
                tk.Label(dist_frame, 
                        text=f"Jarak Euclidean: {euclidean_dist:.4f}\n"
                             f"Threshold: {THRESHOLD}\n"
                             f"Status: {'Verifikasi Berhasil' if euclidean_dist < THRESHOLD else 'Verifikasi Gagal'}",
                        font=TEXT_FONT, bg=BG_COLOR, justify="left").pack(anchor="w")
                
                # Frame untuk matriks intensitas
                matrix_frame = tk.Frame(info_frame, bg=BG_COLOR)
                matrix_frame.pack(side="left", padx=20)
                
                # Frame untuk matriks wajah terdaftar
                registered_matrix_frame = tk.Frame(matrix_frame, bg=BG_COLOR)
                registered_matrix_frame.pack(pady=5)
                
                tk.Label(registered_matrix_frame, text="Matriks Intensitas Pixel (Wajah Terdaftar):", 
                        font=TEXT_FONT, bg=BG_COLOR).pack(anchor="w")
                
                registered_matrix_text = tk.Text(registered_matrix_frame, height=10, width=50, font=("Courier", 10))
                registered_matrix_text.pack(pady=5)
                
                # Format matriks intensitas wajah terdaftar
                registered_matrix_data = []
                for i in range(0, 256, 16):
                    row = [f"{registered_hist[j]:.4f}" for j in range(i, min(i+16, 256))]
                    registered_matrix_data.append(" ".join(row))
                
                registered_matrix_text.insert("1.0", "\n".join(registered_matrix_data))
                registered_matrix_text.config(state="disabled")
                
                # Frame untuk matriks wajah verifikasi
                verification_matrix_frame = tk.Frame(matrix_frame, bg=BG_COLOR)
                verification_matrix_frame.pack(pady=5)
                
                tk.Label(verification_matrix_frame, text="Matriks Intensitas Pixel (Wajah Verifikasi):", 
                        font=TEXT_FONT, bg=BG_COLOR).pack(anchor="w")
                
                verification_matrix_text = tk.Text(verification_matrix_frame, height=10, width=50, font=("Courier", 10))
                verification_matrix_text.pack(pady=5)
                
                # Format matriks intensitas wajah verifikasi
                verification_matrix_data = []
                for i in range(0, 256, 16):
                    row = [f"{self.verification_hist[j]:.4f}" for j in range(i, min(i+16, 256))]
                    verification_matrix_data.append(" ".join(row))
                
                verification_matrix_text.insert("1.0", "\n".join(verification_matrix_data))
                verification_matrix_text.config(state="disabled")
                
                # Frame untuk tombol
                button_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
                button_frame.pack(pady=10)
                
                # Tombol untuk cetak PDF dan CSV
                tk.Button(button_frame, text="Cetak Laporan PDF", 
                         command=lambda: self.create_pdf_report(registered_hist, self.verification_hist, 
                                                              euclidean_dist, failed_users),
                         bg=BUTTON_COLOR, fg="white", 
                         font=BUTTON_FONT).pack(side="left", padx=5)
                         
                tk.Button(button_frame, text="Cetak Perhitungan CSV", 
                         command=lambda: self.create_euclidean_csv(registered_hist, self.verification_hist, 
                                                                 euclidean_dist),
                         bg=BUTTON_COLOR, fg="white", 
                         font=BUTTON_FONT).pack(side="left", padx=5)
            
            # Pack canvas dan scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            v_scrollbar.pack(side="right", fill="y")
            h_scrollbar.pack(side="bottom", fill="x")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan histogram: {str(e)}")
            
    def show_history(self):
        if not self.is_verified:
            messagebox.showwarning("Peringatan", "Anda harus verifikasi terlebih dahulu!")
            return
            
        self.clear_content()
        
        # Frame untuk riwayat
        history_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        history_frame.pack(pady=20)
        
        tk.Label(history_frame, text="Riwayat Ujian", font=TITLE_FONT, bg=BG_COLOR).pack(pady=10)
        
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)
                
            if not history["history"]:
                tk.Label(history_frame, text="Belum ada riwayat ujian", 
                        font=TEXT_FONT, bg=BG_COLOR).pack(pady=20)
                return
                
            # Buat tabel
            columns = ("Nama", "NIM", "Kelas", "Tanggal", "Nilai", "Jawaban Benar")
            tree = ttk.Treeview(history_frame, columns=columns, show="headings")
            
            # Set header
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150)
                
            # Isi data
            for h in history["history"]:
                tree.insert("", "end", values=(
                    h["nama"],
                    h["nim"],
                    h["kelas"],
                    h["tanggal"],
                    f"{h['nilai']:.2f}",
                    f"{h['jawaban_benar']}/{h['total_soal']}"
                ))
                
            tree.pack(pady=20)
            
            # Tombol cetak PDF
            tk.Button(history_frame, text="Cetak Laporan PDF", 
                     command=self.create_history_pdf,
                     bg=BUTTON_COLOR, fg="white", 
                     font=BUTTON_FONT).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan riwayat: {str(e)}")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ExamSystem()
    app.run()