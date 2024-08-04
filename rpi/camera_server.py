import socket
import time
import threading
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, FfmpegOutput, CircularOutput

# Features:
# * Stream to file
#   -> Split into manageable 1-min chunks
#   -> Predict recording time remaining
#   -> Auto-select ideal quality/compression/resolution based on mission time
#   -> Auto-detect feature - only record interesting data
# * Stream to localhost udp address
#   -> OpenCV, or tensorflow applications can run to determine submarine actions
# * Take high quality photos between video streams
#   -> Auto-detect feature - only take pictures with interesting data

class CameraServer:
    def __init__(self):
        self.picam2 = Picamera2()
        self.encoder = H264Encoder(repeat=True)  # Interframe occurs every 60 seconds by default
        self.circular_file_output = CircularOutput()
        self.outputs = [self.circular_file_output]
        self.encoder.output = self.outputs
        #camera_config = self.picam2.create_preview_configuration()
        self.camera_config = self.picam2.create_video_configuration()

        self.encoder_lock = threading.Lock()

        self.connection_handler = threading.Thread(target=self.accept_connections)
        self.connection_handler.setDaemon(True)
        self.connection_handler.start()

    def __del__(self):
        self.picam2.stop_encoder()
        self.picam2.stop()

    # Configure
    def configure_camera(self, config):
        self.camera_config = config
        self.picam2.configure(config)

    def accept_connections(self):
        connection_events = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 12345))
            s.listen()
            while accepted := s.accept():
                print("Connected")
                conn, addr = accepted
                stream = conn.makefile("wb")
                filestream = FileOutput(stream)
                filestream.start()
                with self.encoder_lock:
                    if isinstance(self.encoder.output, list):
                        self.encoder.output += [filestream]
                    else:
                        self.encoder.output = [self.encoder.output, filestream]

                def remove_output_object():
                    if isinstance(self.encoder.output, list):
                        self.encoder.output.remove(filestream)
                    else:
                        self.encoder.output = []
                filestream.connectiondead = lambda _: print(f"Connection has been closed: {_}") or remove_output_object() and self.event.set()


    # Start streaming
    def start(self):
        # picam2.start_preview(Preview.DRM)
        self.picam2.start()
        self.picam2.start_encoder(self.encoder)

    def stop(self):
        self.picam2.stop_encoder()
        self.picam2.stop()

    def start_save(self, filename=None):
        if filename is None:
            self.circular_file_output.fileoutput = f"{int(time.time())}.h264"
        else:
            self.circular_file_output.fileoutput = filename
        self.circular_file_output.start()

    def stop_save(self):
        self.circular_file_output.stop()


if __name__ == "__main__":
    cs = CameraServer()
    cs.start()
    while True:
        pass
    #for i in range(10):
    #    time.sleep(1)
    #    print(i)
    #cs.start_save()
    #for i in range(10):
    #    time.sleep(1)
    #    print(i)
    #cs.stop_save()

