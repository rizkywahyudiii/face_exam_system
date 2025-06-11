# Format matriks intensitas wajah verifikasi
                verification_matrix_data = []
                for i in range(0, 256, 16):
                    row = [f"{self.verification_hist[j]:.4f}" for j in range(i, min(i+16, 256))]
                    verification_matrix_data.append(" ".join(row))
                
                verification_matrix_text.insert("1.0", "\n".join(verification_matrix_data))
                verification_matrix_text.config(state="disabled")