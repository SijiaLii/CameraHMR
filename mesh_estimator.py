import cv2
import os
import torch
import smplx
from progress.bar import Bar
import numpy as np
from collections import defaultdict
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.frame_id = None
        self.tracks = {}            # {track_id: last_bbox}
        self.next_track_id = 0
        self.iou_thresh = 0.3

        self.output = defaultdict(lambda: {
            'pose': [],      # list of (72,)
            'trans': [],     # list of (3,)
            'betas': None,   # (NUM_BETAS,)
            'verts': [],     # list of (V,3)
            'frame_ids': []  # list of int
        })

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector
    
    @staticmethod
    def _iou(boxA, boxB):
        # compute [x1,y1,x2,y2] bbox's IoU
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA);    interH = max(0, yB - yA)
        inter = interW * interH
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter)


    def _associate_tracks(self, boxes):
        """
        Associate detected boxes with existing tracks based on IoU.
        """
        new_tracks = {}
        track_ids = []
        for box in boxes:
            best_id, best_iou = None, 0
            for tid, prev_box in self.tracks.items():
                iou = self._iou(box, prev_box)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou is not None and best_iou >= self.iou_thresh:
                assigned_id = best_id
            else:
                assigned_id = self.next_track_id
                self.next_track_id += 1
            new_tracks[assigned_id] = box
            track_ids.append(assigned_id)
        self.tracks = new_tracks
        return track_ids
    
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int


    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)

    def process_image(self, frame):
        # Detect humans in the image
        det_out = self.detector(frame)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        track_ids = self._associate_tracks(boxes)

        # Get Camera intrinsics using HumanFoV Model
        cam_int = self.get_cam_intrinsics(frame)
        dataset = Dataset(frame, bbox_center, bbox_scale, cam_int, False, f'frame_{self.frame_id:06d}')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)

            verts_np = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            trans_np = output_cam_trans.cpu().numpy() 
            B = out_smpl_params['global_orient'].shape[0]
            glob_aa = batch_rot2aa(out_smpl_params['global_orient'].reshape(-1, 3, 3)).view(B, 3)           # -> [B,3]
            body_aa = batch_rot2aa(out_smpl_params['body_pose'].reshape(-1, 3, 3)).view(B, 23, 3)       # -> [B,23,3]
            pose_np = torch.cat([glob_aa, body_aa.view(B, -1)],dim=1).cpu().numpy()  # -> [B,72]
            betas_np = out_smpl_params['betas'].cpu().numpy()                  # (B,NUM_BETAS)

            for idx, (bbox, tid) in enumerate(zip(boxes, track_ids)):
                if tid is not None:
                    subj = self.output[tid]
                    subj['pose'].append(pose_np[idx])
                    subj['trans'].append(trans_np[idx])
                    subj['verts'].append(verts_np[idx])
                    subj['frame_ids'].append(self.frame_id)
                    subj['betas'] = betas_np[idx] if subj['betas'] is None else subj['betas']
            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = verts_np
            renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.smpl_model.faces, same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=frame.copy())
            renderer.delete()
            front_view = cv2.convertScaleAbs(front_view, alpha= (255.0 if front_view.max()<=1.0 else 1.0))
            if front_view.shape[2] == 3:
                front_view = cv2.cvtColor(front_view, cv2.COLOR_RGB2BGR)
            self.video_writer.write(front_view)



    def run_on_video(self, video_pth, out_folder, fps=25, fourcc_str='mp4v'):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        cap = cv2.VideoCapture(video_pth)
        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")
        h, w = frame0.shape[:2]

        out_video_pth = os.path.join(out_folder, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.video_writer = cv2.VideoWriter(out_video_pth, fourcc, fps, (w, h))

        self.frame_id = 0
        self.process_image(frame0)

        bar = Bar('Preprocess:', fill='#', max=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        bar.next()
        while True:
            flag, frame = cap.read()
            if not flag: break
            self.frame_id += 1
            self.process_image(frame)
            bar.next()
        bar.finish()
        cap.release()

        self.video_writer.release()
        print(f"Saved SMPL overlay video to {out_video_pth}")

        for tid, subj in self.output.items():
            subj['pose'] = np.stack(subj['pose'], axis=0)
            subj['trans'] = np.stack(subj['trans'], axis=0)
            subj['verts'] = np.stack(subj['verts'], axis=0)
            subj['frame_ids'] = np.array(subj['frame_ids'], dtype=int)
        
        return self.output
