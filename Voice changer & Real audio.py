import sys
import os
import numpy as np
import wavio
import sounddevice as sd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.fft import fft, fftfreq
import math
import requests
from scipy.io import wavfile
import time
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Fungsi untuk menghitung desibel dari tekanan suara
def hitung_desibel(tekanan_suara, tekanan_referensi=20e-6):
    if tekanan_suara <= 0:
        return 0  # Menghindari nilai negatif
    db = 20 * math.log10(tekanan_suara / tekanan_referensi)
    return db


class EdgeImpulseUploader:
    def __init__(self, api_key="ei_3a5eb7348a8c2dcfcd9930491dda8417896c31b46d5e7b451673b798d25db075",
                 api_url="https://ingestion.edgeimpulse.com/api/training/files"):
        self.api_key = api_key
        self.api_url = api_url
        self.label = "Rantai"

    def upload_audio_to_edge_impulse(self, audio_filename):
        try:
            with open(audio_filename, "rb") as f:
                response = requests.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "x-label": self.label,
                    },
                    files={"data": (os.path.basename(audio_filename), f, "audio/wav")},
                    timeout=30
                )
                if response.status_code == 200:
                    return True, "Uploaded successfully!"
                else:
                    return False, f"Failed with status code: {response.status_code}, response: {response.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Request failed: {e}"


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Layout setup
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        # Title Label
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setText("<h3 align='center'>Voice Changer and Real Audio Detector</h3>")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 1)

        # Group Box for Parameters
        self.groupBox = QtWidgets.QGroupBox("Parameters", self.centralwidget)
        self.gridLayout_parameters = QtWidgets.QGridLayout(self.groupBox)

        # Input Fields
        self.label_sampling_rate = QtWidgets.QLabel("Sampling Rate:")
        self.lineEdit_sampling_rate = QtWidgets.QLineEdit("16000")
        self.label_update_interval = QtWidgets.QLabel("Update Interval (ms):")
        self.lineEdit_update_interval = QtWidgets.QLineEdit("50")
        self.label_label = QtWidgets.QLabel("Label:")
        self.label_db = QtWidgets.QLabel("Nilai Desibel: 0")
        font = self.label_db.font()
        font.setPointSize(13)
        self.label_db.setFont(font)
        self.label_db.setAlignment(QtCore.Qt.AlignLeft)
        self.lineEdit_label = QtWidgets.QLineEdit("recording")

        # Buttons
        self.pushButton_record = QtWidgets.QPushButton("Start Recording")
        self.pushButton_replay = QtWidgets.QPushButton("Replay Audio")
        self.pushButton_upload = QtWidgets.QPushButton("Upload to Edge Impulse")
        self.pushButton_reset = QtWidgets.QPushButton("Reset")
        self.pushButton_replay.setEnabled(False)
        self.pushButton_upload.setEnabled(False)
        self.pushButton_reset.setEnabled(False) 


        # Layout for Parameters
        self.gridLayout_parameters.addWidget(self.label_sampling_rate, 0, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_sampling_rate, 0, 1)
        self.gridLayout_parameters.addWidget(self.label_update_interval, 1, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_update_interval, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_label, 2, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_label, 2, 1)
        self.gridLayout_parameters.addWidget(self.pushButton_record, 3, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.pushButton_replay, 4, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.pushButton_upload, 5, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.pushButton_reset, 6, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.label_db, 7, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)

        # Plot Widgets for Time and Frequency Domain
        self.plot_widget_time = pg.PlotWidget(self.centralwidget)
        self.plot_widget_time.setBackground('k')
        self.plot_widget_time.setTitle("Time Domain Signal")
        self.plot_widget_time.showGrid(x=True, y=True)
        self.gridLayout.addWidget(self.plot_widget_time, 2, 0, 1, 1)

        self.plot_widget_freq = pg.PlotWidget(self.centralwidget)
        self.plot_widget_freq.setBackground('k')
        self.plot_widget_freq.setTitle("Frequency Domain (FFT)")
        self.plot_widget_freq.showGrid(x=True, y=True)
        self.gridLayout.addWidget(self.plot_widget_freq, 3, 0, 1, 1)

        # Initialize plot data
        self.plot_data_time = self.plot_widget_time.plot(pen=pg.mkPen(color='lime', width=2))  # New color for Time Domain
        self.plot_data_freq = self.plot_widget_freq.plot(pen=pg.mkPen(color='red', width=2))    # New color for FFT


        MainWindow.setCentralWidget(self.centralwidget)

        # Initialize parameters
        self.is_recording = False
        self.audio_data = []
        self.audio_file_path = "gear2.wav"
        self.uploader = EdgeImpulseUploader(api_key="ei_09122e8ba66ab67d4a8cf4ac8cccd44e407c29c04971254fe6b6108e8130a4c4")

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Connect buttons to functions
        self.pushButton_record.clicked.connect(self.toggle_recording)
        self.pushButton_replay.clicked.connect(self.replay_audio)
        self.pushButton_upload.clicked.connect(self.upload_to_edge_impulse)
        self.pushButton_reset.clicked.connect(self.reset_audio)
        pass

    def reset_audio(self):
        # Reset data audio
        self.audio_data = []
        # Hapus file audio jika ada
        if os.path.exists(self.audio_file_path):
            os.remove(self.audio_file_path)
        # Reset grafik
        self.plot_data_time.clear()
        self.plot_data_freq.clear()
        # Reset tampilan desibel
        self.label_db.setText("dB: 0")
        # Disable tombol replay dan upload
        self.pushButton_replay.setEnabled(False)
        self.pushButton_upload.setEnabled(False)
        self.pushButton_reset.setEnabled(False)
        QtWidgets.QMessageBox.information(None, "Reset", "Audio and plots have been reset.")

    def validate_inputs(self):
        try:
            sampling_rate = int(self.lineEdit_sampling_rate.text())
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be a positive integer.")
            update_interval = int(self.lineEdit_update_interval.text())
            if update_interval <= 0:
                raise ValueError("Update interval must be a positive integer.")
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(None, "Input Error", f"Invalid input: {str(e)}")
            return False

    def toggle_recording(self):
        if not self.is_recording:
            if not self.validate_inputs():
                return
            self.is_recording = True
            self.pushButton_record.setText("Stop Recording")
            self.start_recording()
        else:
            self.is_recording = False
            self.pushButton_record.setText("Start Recording")
            self.stop_recording()

    def start_recording(self):
        self.record_start_time = time.time()  # Waktu mulai rekaman
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        self.audio_data = []
        try:
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sampling_rate)
            self.stream.start()
            self.timer.start(int(self.lineEdit_update_interval.text()))
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Error", f"Failed to start recording: {e}")
            self.is_recording = False
            self.pushButton_record.setText("Start Recording")
    
    def stop_recording(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        self.timer.stop()
        self.save_audio()

        # Menampilkan grafik untuk sinyal yang telah direkam
        self.show_recorded_audio_on_plot()

        self.pushButton_replay.setEnabled(True)
        self.pushButton_upload.setEnabled(True)
        self.pushButton_reset.setEnabled(True)

    def show_recorded_audio_on_plot(self):
        # Ambil data yang sudah direkam dan update grafik waktu dan frekuensi
        if self.audio_data:
            audio_data_np = np.concatenate(self.audio_data)
            audio_array = audio_data_np[:, 0]  # Ambil data channel pertama

        # Update grafik waktu (time domain)
        num_samples = len(audio_array)
        elapsed_time = time.time() - self.record_start_time  # Waktu sejak mulai
        time_array = np.linspace(elapsed_time - num_samples / self.sampling_rate, elapsed_time, num_samples)
        self.plot_data_time.setData(time_array, audio_array)

        # Update grafik frekuensi (FFT)
        fft_data = np.abs(fft(audio_array))
        freq_array = fftfreq(num_samples, 1 / self.sampling_rate)
        mask = (freq_array >= 0) & (freq_array <= 3000)
        self.plot_data_freq.setData(freq_array[mask], fft_data[mask])

    def save_audio(self):
        if self.audio_data:
            audio_data_np = np.concatenate(self.audio_data)
            wavio.write(self.audio_file_path, audio_data_np, self.sampling_rate, sampwidth=2)

    def replay_audio(self):
        if os.path.exists(self.audio_file_path):
            _, data = wavfile.read(self.audio_file_path)
            sd.play(data, self.sampling_rate)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        lowcut = 20.0  # Frekuensi bawah (sesuaikan dengan suara rantai)
        highcut = 1000.0  # Frekuensi atas (sesuaikan dengan suara rantai)
        filtered_data = bandpass_filter(indata[:, 0], lowcut, highcut, self.sampling_rate)

        # Tambahkan data yang telah difilter ke audio_data
        self.audio_data.append(filtered_data[:, np.newaxis])  # Tambahkan dimensi untuk channel
        self.update_db_value(filtered_data)

    def update_db_value(self, indata):
        rms_value = np.sqrt(np.mean(indata ** 2))
        if rms_value > 0:
            db_value = hitung_desibel(rms_value)
            self.label_db.setText(f"dB: {db_value:.2f}")

    def update_plot(self):
        if self.audio_data:
            # Ambil data terbaru
            window_size = self.sampling_rate  # 1 detik data
            audio_array = np.concatenate(self.audio_data)[:, 0]
            if len(audio_array) > window_size:
                audio_array = audio_array[-window_size:]

            # Time domain plot
            num_samples = len(audio_array)
            elapsed_time = time.time() - self.record_start_time
            time_array = np.linspace(elapsed_time - num_samples / self.sampling_rate, elapsed_time, num_samples)
            self.plot_data_time.setData(time_array, audio_array)

            # Frequency domain (FFT) plot
            fft_data = np.abs(fft(audio_array))
            freq_array = fftfreq(num_samples, 1 / self.sampling_rate)
            mask = (freq_array >= 100) & (freq_array <= 3000)
            self.plot_data_freq.setData(freq_array[mask], fft_data[mask])

    def upload_to_edge_impulse(self):
        label = self.lineEdit_label.text()
        if not label:
            QtWidgets.QMessageBox.warning(None, "Error", "Label cannot be empty!")
            return
        self.uploader.label = label
        success, message = self.uploader.upload_audio_to_edge_impulse(self.audio_file_path)
        if success:
            QtWidgets.QMessageBox.information(None, "Success", message)
        else:
            QtWidgets.QMessageBox.warning(None, "Error", message)


# Run the application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
