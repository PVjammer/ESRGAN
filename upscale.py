import os
import glob
import cv2
import numpy as np
import torch 
import esrgan.rrdbnet as arch
import time
import threading

from queue import PriorityQueue, Queue

MODEL_PATH = "models/RRDB_ESRGAN_x4.pth"
device = torch.device('cuda')


model = arch.load_default_model()
#model = arch.RRDBNet(3,3, 64, 23, gc=32)
#model.load_state_dict(torch.load(MODEL_PATH), strict=True)
model.eval()
model = model.to(device)

buf = PriorityQueue()
out_q = Queue()

def upscale(frame):
    img = frame * 1.0/255
    img = torch.from_numpy(np.transpose(img[:,:,[2,1,0]], (2,0,1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    return output.astype(np.uint8)

def worker():
    current_frame = 0
    while True:
        if buf.empty():
            time.sleep(1)
            continue
        _, frame_obj = buf.get()
        if frame_obj.get("frame_number") <= current_frame:
            time.sleep(1)
            continue
        out_q.put(upscale(frame_obj["frame"]))


def load_frames(cap):
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream unavailable")
            break
        buf.put((-1 * frame_num, {"frame_number": frame_num, "frame": frame}))
        frame_num += 1


def main(args):
    frame_list = []
    frame_num = 0
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(int(args.cam_id))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))    
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.shape)
        if args.video:
            frame = cv2.resize(frame, (int(args.width), int(args.height)))
        if not ret:
            print("Stream unavailable")
            return 0
        print(frame.dtype)
        cv2.imshow("Frame", frame)
        if args.upscale:
            img = frame * 1.0/255
            img = torch.from_numpy(np.transpose(img[:,:,[2,1,0]], (2,0,1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            frame_list.append(output.astype(np.uint8))
            cv2.imshow("Upscaled", output.astype(np.uint8))
            # cv2.imwrite("test.png", output)
            # cv2.imwrite("orig.png", frame)
            frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return frame_list

def render():
    while True:
        if out_q.empty():
            time.sleep(1)
            continue
        frame = out_q.get()
        print(frame.shape)
        cv2.imshow("Upscale", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def run(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(int(args.cam_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    if not cap.isOpened():
        print("Camera not available at {!s}".format(args.cam_id))
        return
    loadframes = threading.Thread(target=load_frames, daemon=True, kwargs={"cap":cap})
    worker1 = threading.Thread(target=worker, daemon=True)
    worker2 = threading.Thread(target=worker, daemon=True)
    rendering = threading.Thread(target=render)
    loadframes.start()
    while buf.empty():
        time.sleep(.5)
    worker1.start()
    worker2.start()
    rendering.start()
    rendering.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", default=0, help="Integer identifier of the camera to use")
    parser.add_argument("--video", default=None, help="Path of video file to upscale")
    parser.add_argument("--out", default=".", help="Path of the output video")
    parser.add_argument("--width", default=490, help="Width of the camera stream")
    parser.add_argument("--height", default=270, help="Height of the camera stream")
    parser.add_argument("--upscale", default=False, help="If true renders an upscaled version of the video", action="store_true")

    args = parser.parse_args()

    frame_list = main(args)
    print(len(frame_list))
    render(frame_list)
#    run(args)
cv2.destroyAllWindows()
