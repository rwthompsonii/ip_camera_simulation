"""
NeurodataLab LLC 25.12.2019
Created by Andrey Belyaev
"""
import cv2
import gi
import numpy as np
import asyncio
import itertools
import signal
import sys
import traceback
import aiohttp
import requests
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict


gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# Factory class
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    pass

# Server class
class GstServer(GstRtspServer.RTSPServer):
    pass


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)  # Init super class
        self.cap = cv2.VideoCapture(2)  # Initialize webcam. You may have to change 0 to your webcam number
        self.frame_number = 0  # Current frame number
        self.fps = 30  # output streaming fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=1280,height=720,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)

        self.christmas_image = cv2.imread('christmas.png', cv2.IMREAD_COLOR)
        self.christmas_image = cv2.resize(self.christmas_image, None, fx=0.1, fy=0.1)
        self.scale_factor = 0.5#TODO: this is the default, provide the ability to change it
        self.bodypix_url = "http://127.0.0.1:9000"#TODO: and here
        background = cv2.imread('background.jpg');#TODO: same here
        self.hologram = True #TODO: and more
        self.width = 1280
        self.height = 720
        
        #TODO: this came out of loadImages() it should go back there
        self.images: Dict[str, Any] = {}
        if background is not None:
                background = cv2.resize(background, (self.width, self.height))
                background = itertools.repeat(background)

        self.images["background"] = background
        self.session = requests.Session()
        

    def on_need_data(self, src, lenght):
        if self.cap.isOpened():  # Check webcam is opened
            ret, frame = self.cap.read()  # Read next frame
            frame = self.draw_on_frame(frame)  # Draw something on frame frame
            if ret:  # If read success
                data = frame.tostring()  # Reformat frame to string
                buf = Gst.Buffer.new_allocate(None, len(data), None)  # Allocate memory
                buf.fill(0, data)  # Put new data in memory
                buf.duration = self.duration  # Set data duration
                timestamp = self.frame_number * self.duration  # Current frame timestamp
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp  # Set frame timestamp
                self.frame_number += 1  # Increase current frame number
                retval = src.emit('push-buffer', buf)  # Push allocated memory to source container
                if retval != Gst.FlowReturn.OK:  # Check pushing process
                    print(retval)  # Print error message
    
    def _get_mask(self, frame, session):
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor,
                           fy=self.scale_factor)
        _, data = cv2.imencode(".png", frame)
        with session.post(
            url=self.bodypix_url, data=data.tostring(),
            headers={"Content-Type": "application/octet-stream"}
        ) as r:
            mask = np.frombuffer(r.content, dtype=np.uint8)
            mask = mask.reshape((frame.shape[0], frame.shape[1]))
            mask = cv2.resize(
                mask, (0, 0), fx=1 / self.scale_factor,
                fy=1 / self.scale_factor, interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
            mask = cv2.blur(mask.astype(float), (30, 30))
            return mask

    def shift_image(self, img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        return img

    def hologram_effect(self, img):
        # add a blue tint
        holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        # add a halftone effect
        bandLength, bandGap = 2, 3
        for y in range(holo.shape[0]):
            if y % (bandLength+bandGap) < bandLength:
                holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, self.shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4, self.shift_image(holo.copy(), -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
        return out 

    def draw_on_frame(self, frame):
        #idxs = np.where(np.logical_and(self.christmas_image > 0, self.christmas_image < 255))
        #for i in range(3):
        #    for j in range(2):
        #        cur_idxs = (idxs[0] + i * self.christmas_image.shape[0],
        #                    idxs[1] + j * (frame.shape[1] - self.christmas_image.shape[1]),
        #                    idxs[2])
        #        frame[cur_idxs] = self.christmas_image[idxs]
        
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = self._get_mask(frame, self.session)
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")
                traceback.print_exc()
                
        if self.hologram: 
            frame = self.hologram_effect(frame)

        # composite the mask and background
        background = next(self.images["background"])
        for c in range(frame.shape[2]):
            frame[:, :, c] = frame[:, :, c] * mask + background[:, :, c] * (1 - mask)
            
        return frame

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)  # Launch gst plugin

    def do_configure(self, rtsp_media):
        self.frame_number = 0  # Set current frame number to zero
        appsrc = rtsp_media.get_element().get_child_by_name('source')  # get source from gstreamer
        appsrc.connect('need-data', self.on_need_data)  # set data provider


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)  # Init super class
        self.factory = SensorFactory()  # Create factory
        self.set_service("3002")  # Set service port
        self.factory.set_shared(True)  # Set shared to true
        self.get_mount_points().add_factory("/test", self.factory)  # Add routing to access factory
        self.attach(None)


if __name__ == '__main__':
    loop = GObject.MainLoop()  # Create infinite loop for gstreamer server
    GObject.threads_init()  # Initialize server threads for asynchronous requests
    Gst.init(None)  # Initialize GStreamer

    server = GstServer()  # Initialize server
    loop.run()  # Start infinite loop
