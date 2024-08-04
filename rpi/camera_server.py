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
        #MAX_CONNECTIONS = 9999 -> May need to be used if a large amount of connects/disconnects happen, don't want to exceed integer limit for conn_id_counter
        conn_id_counter = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 12345))
            s.listen()
            while accepted := s.accept():

                conn, addr = accepted
                stream = conn.makefile("wb")
                filestream = FileOutput(stream)
                conn_id_counter += 1
                setattr(filestream, "id", conn_id_counter)
                filestream.start()
                with self.encoder_lock:
                    if isinstance(self.encoder.output, list):
                        self.encoder.output += [filestream]
                    else:
                        self.encoder.output = [self.encoder.output, filestream]
                print(f"Connected with id ({conn_id_counter})")

                def remove_output_object(id_to_remove):
                    print(f"Requesting to remove: {id_to_remove}")
                    for i in self.encoder.output:
                        print(f"{i} - {i.id if hasattr(i, 'id') else None}")
                    if isinstance(self.encoder.output, list):
                        # find the correct list item
                        for i in self.encoder.output:
                            if hasattr(i, "id") and i.id == id_to_remove:
                                print(f"Removed with id: {id_to_remove}")
                                self.encoder.output.remove(i)
                    else:
                        self.encoder.output = []
                def connectiondead_wrapper(conn_id):
                    return lambda _: print(f"Connection has been closed for id ({conn_id}): {_}") or remove_output_object(conn_id) and self.event.set()
                filestream.connectiondead = connectiondead_wrapper(conn_id_counter) 

    # Start streaming
    def start(self):
        #self.picam2.start_preview(Preview.DRM)
        self.picam2.start()
        # Note, encoder appears to add ~1 second delay
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

