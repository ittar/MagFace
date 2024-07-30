import argparse
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

def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / np.pi
    return dist

def run_cam(model):
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    detector = MTCNN()
    img_taken = False
    img_embeddings = []
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        color = (255,0,0)
        ret, frame = cap.read()
        if not ret:
            break
        w, h = frame.shape[0], frame.shape[1]
        detect_box = detector.detect_faces(frame)
        for b in detect_box:
            box = b['box']
            if cv2.waitKey(1) & 0xFF == ord('1'):
                face = frame[ max(0, box[1]) : min(w-1, box[1]+box[3]), max(0, box[0]) : min(h-1, box[0]+box[2]) ]
                cv2.imwrite('../../img/capture.jpg', face)
                face = preprocess_frame(face)
                img_taken = True
                with torch.no_grad():
                    img_embeddings = model(face).data.cpu().numpy() # shape (1, 512)
            if (img_taken):
                with torch.no_grad():
                    embeddings = model(face).detach().cpu().numpy() # shape (1, 512)
                sim = distance_(embeddings, img_embeddings)
                if sim > THRES : color = (0,255,0)
                else : color = (0,0,255)
                cv2.putText(frame, str(sim), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 4)
        
        if cv2.waitKey(1) & 0xFF == ord('2'):
            print('Release picture')
            img_taken = False
        
        if cv2.waitKey(1) & 0xFF == 27:  # esc key
            break
        cv2.imshow('image', frame)
    cap.release()
    cv2.destroyAllWindows()

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
    args = parser.parse_args()

    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    if not args.cpu_mode:
        model = model.cuda()
    
    model.eval()

    run_cam(model)