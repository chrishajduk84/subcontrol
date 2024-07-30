#!/bin/bash
cd /tmp

# Replace camera autodetect
sed 's/camera_auto_detect=1/camera_auto_detect=0/' /boot/firmware/config.txt

# Add dtoverlay for camera
echo -e "[all]\r\ndtoverlay=imx708\r\n" >> /boot/firmware/config.txt

wget -O install_pivariety_pkgs.sh https://github.com/AduCAM/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera
./install_pivariety_pkgs.sh -p libcamera_apps
./install_pivariety_pkgs.sh -p imx708_b0311_kernel_driver

reboot