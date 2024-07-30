import argparse
import glob
import os
import pandas as pd
import cv2
from mtcnn import MTCNN
import numpy as np
import torch
from torchvision import transforms
import sys
sys.path.append("../..")
from inference.network_inf import builder_inf

THRES = 0.75

def preprocess_frame(img):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
        transforms.Resize((112, 112)),
    ])
    return trans(img).unsqueeze(0)

def process_video(model, vid_path, df_list, vid_dir):
    cap = cv2.VideoCapture(vid_path)  # 0 is the default camera
    detector = MTCNN()
    idx = 0
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return df_list
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 3)  # Process one frame per second
    while True:
        color = (255,0,0)
        ret, frame = cap.read()
        if not ret:
            break
        w, h = frame.shape[0], frame.shape[1]
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if frame_id % frame_interval == 0:
            detect_box = detector.detect_faces(frame)
            for b in detect_box:
                box = b['box']
                face = frame[ max(0, box[1]) : min(w-1, box[1]+box[3]), max(0, box[0]) : min(h-1, box[0]+box[2]) ]
                path  = f'../../img/{vid_dir}/capture_image_{idx}.jpg'
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                cv2.imwrite(path, face)
                face = preprocess_frame(face)
                img_taken = True
                with torch.no_grad():
                    img_embeddings = model(face).data.cpu().numpy()[0] # shape (512)    
                df_list.append([path, str(img_embeddings.tolist())])
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 4)
                idx += 1
        
        if cv2.waitKey(1) & 0xFF == 27:  # esc key
            break
        cv2.imshow('image', frame)
    cap.release()
    cv2.destroyAllWindows()
    return df_list

def run_cam(vid_paths, name):
    df_list = []   

    for vid_path in vid_paths:
        base = os.path.basename(vid_path)
        df_list = process_video(model, vid_path, df_list, base)
        print(f'done {base}')
    
    df = pd.DataFrame(df_list, columns=['path', 'vector'])
    os.makedirs('../../df', exist_ok=True)
    df.to_csv(f'../../df/{name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cam feature test')
    parser.add_argument('--arch', default='iresnet100', type=str,
                        help='backbone architechture')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='The embedding feature size')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cpu_mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
    parser.add_argument("--vid_dir", default='../../vid', type=str, help="Video directory")
    parser.add_argument("--name", default='p2v', type=str, help="csv name")
    args = parser.parse_args()

    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    if not args.cpu_mode:
        model = model.cuda()
    
    model.eval()

    vid_paths = glob.glob(os.path.join(args.vid_dir, '*'))

    run_cam(vid_paths, args.name)