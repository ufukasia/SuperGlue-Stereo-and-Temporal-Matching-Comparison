import cv2
import numpy as np
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class StereoMatcher:
    def __init__(self, base_path="", sequence="MH_01_easy"):
        """StereoMatcher sınıfını başlat"""
        self.base_path = Path(base_path)
        self.sequence = sequence
        self.results = []
        # Temporal takip için önceki frame verileri
        self.prev_frame = {
            'kp': None,
            'desc': None,
            'points3d': None,
            'image': None,
            'filename0': None  # Önceki cam0 dosya adı
        }

        # ORB dedektörü oluştur
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )

        # BFMatcher'ları oluştur
        self.stereo_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        self.temporal_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        print("Kalibrasyon parametreleri yükleniyor...")
        self.load_calibration()
        print("Görüntü zaman damgaları yükleniyor...")
        self.load_timestamps()

    def load_calibration(self):
        """Kamera kalibrasyon parametrelerini yükle"""
        try:
            # Kalibrasyon YAML dosyalarını oku
            cam0_yaml = self.base_path / self.sequence / "mav0/cam0/sensor.yaml"
            cam1_yaml = self.base_path / self.sequence / "mav0/cam1/sensor.yaml"

            with open(cam0_yaml, 'r') as f:
                self.cam0_params = yaml.safe_load(f)
            with open(cam1_yaml, 'r') as f:
                self.cam1_params = yaml.safe_load(f)

            # Kamera içsel parametreleri
            intrinsics1 = self.cam0_params['intrinsics']
            intrinsics2 = self.cam1_params['intrinsics']

            self.K1 = np.array([
                [intrinsics1[0], 0, intrinsics1[2]],
                [0, intrinsics1[1], intrinsics1[3]],
                [0, 0, 1]
            ])

            self.K2 = np.array([
                [intrinsics2[0], 0, intrinsics2[2]],
                [0, intrinsics2[1], intrinsics2[3]],
                [0, 0, 1]
            ])

            # Bozulma katsayıları
            self.D1 = np.array(self.cam0_params['distortion_coefficients'])
            self.D2 = np.array(self.cam1_params['distortion_coefficients'])

            # Stereo dışsal parametreleri (cam0 -> cam1 dönüşüm matrisi)
            baseline_matrix = np.array([
                [ 0.99999609,  0.0023145 , -0.00136141, -0.110073  ],
                [-0.00231508,  0.99999693, -0.00042232, -0.000399  ],
                [ 0.00136043,  0.00042547,  0.99999868, -0.000284  ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]
            ])
            # Dönüşüm matrisinden R ve T'yi al
            self.R = baseline_matrix[:3, :3]
            self.T = baseline_matrix[:3, 3]

            # Baseline norm bilgisini yazdır
            baseline_norm = np.linalg.norm(self.T)
            print("Baseline (cam0 to cam1) matris:\n", baseline_matrix)
            print(f"Baseline norm: {baseline_norm:.9f} [m]")

            print("Rektifikasyon haritaları hesaplanıyor...")
            self.compute_rectification()

        except Exception as e:
            print(f"Kalibrasyon yükleme hatası: {e}")
            raise

    def compute_rectification(self):
        """Rektifikasyon dönüşümlerini hesapla"""
        img_size = (752, 480)  # Kamera çözünürlüğü

        # Stereo rektifikasyon parametrelerini hesapla
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            img_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Rektifikasyon haritalarını hesapla
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1,
            img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2,
            img_size, cv2.CV_32FC1
        )

    def load_timestamps(self):
        """Görüntü zaman damgalarını yükle"""
        try:
            # CSV dosyalarını oku
            cam0_csv = self.base_path / self.sequence / "mav0/cam0/data.csv"
            cam1_csv = self.base_path / self.sequence / "mav0/cam1/data.csv"

            self.cam0_df = pd.read_csv(cam0_csv)
            self.cam1_df = pd.read_csv(cam1_csv)

            # Zaman damgası sütununu yeniden adlandır ve string yap
            self.cam0_df = self.cam0_df.rename(columns={'#timestamp [ns]': 'timestamp'})
            self.cam1_df = self.cam1_df.rename(columns={'#timestamp [ns]': 'timestamp'})
            self.cam0_df['timestamp'] = self.cam0_df['timestamp'].astype(str)
            self.cam1_df['timestamp'] = self.cam1_df['timestamp'].astype(str)

            print(f"Kamera 0'dan {len(self.cam0_df)} zaman damgası yüklendi")
            print(f"Kamera 1'den {len(self.cam1_df)} zaman damgası yüklendi")

        except Exception as e:
            print(f"CSV dosya yükleme hatası: {e}")
            raise

    def rectify_images(self, img1, img2):
        """Stereo görüntüleri rektifiye et"""
        rect1 = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect1, rect2

    def detect_features(self, img):
        """Görüntüde özellik noktalarını tespit et"""
        try:
            if img is None:
                print("Uyarı: Özellik tespiti için görüntü boş")
                return None, None

            kp, desc = self.orb.detectAndCompute(img, None)

            if kp is None or len(kp) == 0:
                print("Uyarı: Hiç özellik noktası tespit edilemedi")
                return None, None

            return kp, desc

        except Exception as e:
            print(f"Özellik tespiti hatası: {e}")
            return None, None

    def match_stereo_features(self, desc1, desc2):
        """Stereo görüntüler arasında özellik eşleştirme"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            # kNN eşleştirme yap
            matches = self.stereo_matcher.knnMatch(desc1, desc2, k=2)

            # Lowe oranı testi uygula
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            return good_matches

        except Exception as e:
            print(f"Stereo eşleştirme hatası: {e}")
            return []

    def filter_matches_with_fundamental(self, kp1, kp2, matches, threshold=1.0):
        """Temel matris kısıtlaması ile eşleştirmeleri filtrele"""
        if len(matches) < 8:
            return [], None

        # Eşleşen noktaları al
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # RANSAC ile temel matrisi hesapla
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold)

        if F is None or mask is None:
            return [], None

        # İyi eşleştirmeleri seç
        good_matches = []
        filtered_pts1 = []
        filtered_pts2 = []

        for i, (m, msk) in enumerate(zip(matches, mask.ravel())):
            if msk:
                good_matches.append(m)
                filtered_pts1.append(pts1[i])
                filtered_pts2.append(pts2[i])

        return good_matches, (np.array(filtered_pts1), np.array(filtered_pts2))

    def triangulate_points(self, pts1, pts2):
        """3D noktaları üçgenle"""
        # Noktaları normalize et
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2),
            self.K1,
            self.D1,
            R=self.R1,
            P=self.P1
        )
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2),
            self.K2,
            self.D2,
            R=self.R2,
            P=self.P2
        )

        # Projeksiyon matrisleri
        P1 = self.P1
        P2 = self.P2

        # Noktaları üçgenle
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def process_frame_pair(self, timestamp):
        """Stereo kare çiftini işle"""
        try:
            # Görüntü dosya adlarını al
            filename0 = self.cam0_df.loc[self.cam0_df['timestamp'] == timestamp, 'filename'].values
            filename1 = self.cam1_df.loc[self.cam1_df['timestamp'] == timestamp, 'filename'].values

            if len(filename0) == 0 or len(filename1) == 0:
                print(f"Uyarı: {timestamp} için dosya adları bulunamadı")
                return None, None

            filename0 = filename0[0]
            filename1 = filename1[0]

            # Görüntüleri yükle
            img1_path = self.base_path / self.sequence / "mav0/cam0/data" / filename0
            img2_path = self.base_path / self.sequence / "mav0/cam1/data" / filename1

            img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"Uyarı: {timestamp} için görüntüler yüklenemedi")
                return None, None

            # Görüntüleri rektifiye et
            rect1, rect2 = self.rectify_images(img1, img2)

            # Özellik noktalarını bul
            kp1, desc1 = self.detect_features(rect1)
            kp2, desc2 = self.detect_features(rect2)

            if kp1 is None or kp2 is None or desc1 is None or desc2 is None:
                print(f"Uyarı: {timestamp} için özellik veya descriptor bulunamadı")
                return None, None

            # İlk stereo eşleştirme
            initial_matches = self.match_stereo_features(desc1, desc2)

            # Temel matris kısıtlaması ile filtrele
            good_matches, points = self.filter_matches_with_fundamental(kp1, kp2, initial_matches)

            if not good_matches or points is None:
                print(f"Uyarı: {timestamp} için geçerli eşleştirme bulunamadı")
                return None, None

            pts1, pts2 = points

            # 3D noktaları hesapla
            points_3d = self.triangulate_points(pts1, pts2)

            # Eşleşen keypoint ve descriptor'ları topla
            matched_kp = [kp1[m.queryIdx] for m in good_matches]
            matched_desc = np.array([desc1[m.queryIdx] for m in good_matches])

            # Temporal takip - stereo eşleştirmeden gelen noktaları kullan
            prev_pts, curr_pts, motion_vectors = self.track_temporal_matches(
                matched_kp, matched_desc, points_3d, rect1
            )

            # Sonuçları kaydet
            result = {
                'timestamp': timestamp,
                'kp1_count': len(kp1),
                'kp2_count': len(kp2),
                'stereo_matches': len(initial_matches),
                'triangulated_points': len(good_matches),
                'temporal_matches': len(prev_pts) if prev_pts is not None else 0
            }

            # Görselleştirme verilerini hazırla
            vis_data = {
                'stereo': (rect1, rect2, kp1, kp2, good_matches),
                'temporal': (self.prev_frame['image'], rect1, prev_pts, curr_pts, motion_vectors)
                            if prev_pts is not None else None
            }

            # Dosya adlarını hazırla
            filenames_stereo = (filename0, filename1)
            filenames_temporal = (self.prev_frame['filename0'], filename0)

            print(f"İşleme başarılı - Toplam/İyi Eşleştirme: {len(initial_matches)}/{len(good_matches)}")
            print(f"Temporal eşleştirme sayısı: {len(prev_pts) if prev_pts is not None else 0}")

            # Son kareyi güncelle (filename0'u önceki olarak kaydet)
            self.prev_frame = {
                'kp': matched_kp,
                'desc': matched_desc,
                'points3d': points_3d,
                'image': rect1.copy() if rect1 is not None else None,
                'filename0': filename0  # Şu anki filename0'u önceki olarak kaydet
            }

            return (result, vis_data, filenames_stereo, filenames_temporal)

        except Exception as e:
            print(f"Kare çifti işleme hatası: {e}")
            return None, None, None, None

    def track_temporal_matches(self, curr_kp, curr_desc, curr_points3d, curr_img):
        """Ardışık kareler arasında 3D noktaları takip et"""
        if self.prev_frame['kp'] is None:
            # İlk kare için önceki veri yok
            return None, None, None

        try:
            if curr_desc is None or len(curr_desc) == 0 or self.prev_frame['desc'] is None or len(self.prev_frame['desc']) == 0:
                print("Uyarı: Descriptor'lar eksik!")
                return None, None, None

            print(f"Önceki keypoint sayısı: {len(self.prev_frame['kp'])}")
            print(f"Geçerli keypoint sayısı: {len(curr_kp)}")
            print(f"Önceki descriptor sayısı: {len(self.prev_frame['desc'])}")
            print(f"Geçerli descriptor sayısı: {len(curr_desc)}")
            print(f"3D noktalar sayısı: {len(curr_points3d)}")

            # Descriptor'ları eşleştir
            matches = self.temporal_matcher.knnMatch(self.prev_frame['desc'], curr_desc, k=2)

            # Lowe oranı testi uygula
            good_matches = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    # İndeks kontrolü ekle
                    if (m.queryIdx < len(self.prev_frame['points3d']) 
                        and m.trainIdx < len(curr_points3d)):
                        good_matches.append(m)

            if len(good_matches) < 8:
                print(f"Uyarı: Yetersiz eşleşme sayısı ({len(good_matches)})")
                return None, None, None

            # Eşleşen noktaları al
            prev_pts = []
            curr_pts = []
            prev_3d = []
            curr_3d = []

            for m in good_matches:
                if (m.queryIdx < len(self.prev_frame['kp']) and 
                    m.trainIdx < len(curr_kp) and 
                    m.queryIdx < len(self.prev_frame['points3d']) and 
                    m.trainIdx < len(curr_points3d)):
                    
                    prev_pts.append(self.prev_frame['kp'][m.queryIdx].pt)
                    curr_pts.append(curr_kp[m.trainIdx].pt)
                    prev_3d.append(self.prev_frame['points3d'][m.queryIdx])
                    curr_3d.append(curr_points3d[m.trainIdx])

            if not prev_pts:  # Eğer geçerli nokta bulunamadıysa
                return None, None, None

            prev_pts = np.float32(prev_pts)
            curr_pts = np.float32(curr_pts)
            prev_3d = np.array(prev_3d)
            curr_3d = np.array(curr_3d)

            # Hareket vektörlerini hesapla
            motion_vectors = curr_3d - prev_3d

            print(f"Geçerli temporal eşleşme sayısı: {len(prev_pts)}")

            return prev_pts, curr_pts, motion_vectors

        except Exception as e:
            print(f"Temporal takip hatası: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def visualize_combined_tracking(self, stereo_data, temporal_data, filenames_stereo, filenames_temporal, save_path=None):
        """Stereo ve temporal eşleştirmeleri tek bir plotta göster (1 satır, 4 sütun)"""
        if stereo_data is None:
            return

        try:
            rect1, rect2, kp1, kp2, stereo_matches = stereo_data

            # Ana görüntü alanını oluştur
            h, w = rect1.shape[:2]
            combined_h = h  # Tek satır
            combined_w = 4 * w  # Dört sütun
            vis_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

            # Stereo bölümü (ilk iki sütun)
            stereo_vis = cv2.cvtColor(rect1, cv2.COLOR_GRAY2BGR)
            stereo_vis2 = cv2.cvtColor(rect2, cv2.COLOR_GRAY2BGR)

            # Görüntülerin üzerine etiket ekle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 255)  # Sarı tonları
            thickness = 4

            # Tüm özellik noktalarını kırmızı çiz
            for kp in kp1:
                pt = tuple(map(int, kp.pt))
                cv2.circle(stereo_vis, pt, 10, (0, 0, 255), -1)

            for kp in kp2:
                pt = tuple(map(int, kp.pt))
                cv2.circle(stereo_vis2, pt, 10, (0, 0, 255), -1)

            # Eşleşen noktaları yeşil çiz
            for m in stereo_matches:
                pt1 = tuple(map(int, kp1[m.queryIdx].pt))
                pt2 = tuple(map(int, kp2[m.trainIdx].pt))

                cv2.circle(stereo_vis, pt1, 10, (0, 255, 0), -1)
                cv2.circle(stereo_vis2, pt2, 10, (0, 255, 0), -1)
            
            # Cam0 için dosya adı ekle
            cv2.putText(stereo_vis, f'Cam0 Curr {filenames_stereo[0]}', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(stereo_vis, f'ORB', (10, 450), font, font_scale, color, thickness, cv2.LINE_AA)
            # Cam1 için dosya adı ekle
            cv2.putText(stereo_vis2, f'Cam1 Curr  {filenames_stereo[1]}', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

            # Stereo görüntüleri yerleştir
            vis_img[:, :w] = stereo_vis
            vis_img[:, w:2*w] = stereo_vis2

            # Temporal bölümü (son iki sütun)
            if temporal_data is not None:
                prev_img, curr_img, prev_pts, curr_pts, motion_vectors = temporal_data
                filename_prev, filename_curr = filenames_temporal

                if prev_pts is not None and curr_pts is not None and len(prev_pts) > 0:
                    # Görüntüleri hazırla
                    temp_vis1 = cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    temp_vis2 = cv2.cvtColor(curr_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    # Her nokta için
                    for i, ((x1, y1), (x2, y2)) in enumerate(zip(prev_pts, curr_pts)):
                        # Hareket büyüklüğüne göre renk
                        if motion_vectors is not None and i < len(motion_vectors):
                            motion_magnitude = np.linalg.norm(motion_vectors[i])
                            color_mapped = plt.cm.RdYlGn(1 - min(motion_magnitude / 0.1, 1.0))[:3]
                            color_mapped = tuple(int(c * 255) for c in color_mapped)
                        else:
                            color_mapped = (0, 255, 0)

                        # Noktaları çiz
                        cv2.circle(temp_vis1, (int(x1), int(y1)), 10, color_mapped, -1)
                        cv2.circle(temp_vis2, (int(x2), int(y2)), 10, color_mapped, -1)
                    
                    # Görüntülerin üzerine etiket ekle
                    cv2.putText(temp_vis1, f'Prev: {filename_prev}', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                    cv2.putText(temp_vis2, f'Curr: {filename_curr}', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

                    # Temporal görüntüleri yerleştir
                    vis_img[:, 2*w:3*w] = temp_vis1
                    vis_img[:, 3*w:4*w] = temp_vis2

            # Kaydet veya göster
            if save_path:
                cv2.imwrite(save_path, vis_img)
            else:
                plt.figure(figsize=(20, 5))
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
            print(f"Satır: {e.__traceback__.tb_lineno}")

    def process_sequence(self, max_frames=None, visualize=False, viz_interval=1):
        """Tüm sekansı işle"""
        try:
            timestamps = sorted(set(self.cam0_df['timestamp'].astype(str)))

            if max_frames:
                timestamps = timestamps[:max_frames]

            if visualize:
                os.makedirs("visualizations_orb", exist_ok=True)

            total_frames = len(timestamps)
            print(f"{total_frames} kare işleniyor...")

            for i, timestamp in enumerate(timestamps):
                print(f"Kare işleniyor {i+1}/{total_frames}")

                # Frame pair ve dosya adlarını işle
                result, vis_data, filenames_stereo, filenames_temporal = self.process_frame_pair(timestamp)

                if result:
                    self.results.append(result)

                    if visualize and i % viz_interval == 0 and vis_data is not None:
                        # Stereo ve temporal eşleştirmeleri tek plotta göster
                        save_path = f"visualizations_orb/combined_tracking_{timestamp}.png"
                        self.visualize_combined_tracking(
                            vis_data['stereo'],
                            vis_data['temporal'],
                            filenames_stereo,
                            filenames_temporal,
                            save_path=save_path
                        )

            print("\nİşlem tamamlandı!")
            self.save_results()

        except Exception as e:
            print(f"Sekans işleme hatası: {e}")
            raise

    def save_results(self):
        """Sonuçları CSV dosyasına kaydet"""
        if not self.results:
            print("Kaydedilecek sonuç yok!")
            return

        os.makedirs("results_orb", exist_ok=True)

        df = pd.DataFrame(self.results)
        output_file = f"results_orb/stereo_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Sonuçlar kaydedildi: {output_file}")

    def compute_statistics(self):
        """İstatistikleri hesapla ve yazdır"""
        if not self.results:
            print("İstatistik hesaplanacak sonuç yok!")
            return None

        df = pd.DataFrame(self.results)

        stats = {
            'avg_kp1_count': df['kp1_count'].mean(),
            'std_kp1_count': df['kp1_count'].std(),
            'avg_kp2_count': df['kp2_count'].mean(),
            'std_kp2_count': df['kp2_count'].std(),
            'avg_stereo_matches': df['stereo_matches'].mean(),
            'std_stereo_matches': df['stereo_matches'].std(),
            'avg_triangulated': df['triangulated_points'].mean(),
            'std_triangulated': df['triangulated_points'].std(),
            'avg_temporal_matches': df['temporal_matches'].mean(),
            'std_temporal_matches': df['temporal_matches'].std(),
            'total_frames': len(df)
        }

        print("\nEşleştirme İstatistikleri:")
        print("-" * 50)
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        print("-" * 50)

        return stats

def main():
    try:
        # Matcher'ı başlat
        matcher = StereoMatcher("")

        # Sekansı işle
        matcher.process_sequence(
            max_frames=None,     # Tüm kareleri işle
            visualize=True,      # Görselleştirmeleri oluştur
            viz_interval=100     # Her 100 karede bir görselleştir
        )

        # İstatistikleri hesapla ve göster
        stats = matcher.compute_statistics()

    except Exception as e:
        print(f"Ana program hatası: {e}")
        raise

if __name__ == "__main__":
    main()
