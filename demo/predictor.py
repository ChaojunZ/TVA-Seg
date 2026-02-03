import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

import numpy as np
import matplotlib.pyplot as plt

import warnings
# from pytorch_grad_cam import GradCAM
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import numpy as np
import cv2
from PIL import Image
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import json


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # # ####
        # file = open('/home/liuchanghe/lmy/CAT-Seg/datasets/pc459.json','r',encoding='utf-8')
        # data = json.load(file)
        # category_to_id = {category: idx for idx, category in enumerate(data)}
        # # print(category_to_id["person"])

        # hot = predictions['sem_seg']
        # height_hot = hot.size(1)
        # width_hot = hot.size(2)
        # hot = hot.reshape(1,459,height_hot,width_hot)
        # # print(hot.shape)

        # probs = F.softmax(hot,dim=1)

        # # Step 2: 选择目标类别（例如类别 0）
        # target_class = category_to_id["aeroplane"]   #
        # class_prob = probs[0, target_class].cpu().numpy()  # 形状为 (256, 256)

        # # Step 3: 生成热力图
        # plt.imshow(class_prob, cmap='jet', alpha=0.5)  # 使用 'hot' 颜色映射
        # plt.colorbar()  # 显示颜色条

        # # Step 4: 叠加到原图
        # plt.imshow(image)  # 显示原始图像
        # plt.imshow(class_prob, cmap='jet', alpha=0.5, interpolation='bilinear')  # 叠加热力图
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('/data4/zhangchaojun/DeCLIP/keshihua/123/2007_006946_plane_hot.png')

        with open('/data4/zhangchaojun/catseg/datasets/order/pc59.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        category_to_id = {cat: idx for idx, cat in enumerate(data)}

        # 2. 假设 predictions['sem_seg'] 是模型原始输出 logits，shape=[C, H, W]
        logits: torch.Tensor = predictions['sem_seg']  # e.g. torch.Size([459, 256, 256])
        print("zzzzzzzzzzzzz",predictions['sem_seg'].shape)
        # 3. 取每个像素的最大概率类别，得到标签图 shape=[H, W]
        pred_map = logits.argmax(dim=0)               # torch.LongTensor, shape=[H, W]

        # 4. 获得目标类别 ID
        target_class = category_to_id["book"]
        print(f"目标类别 'aeroplane' 的全局 ID = {target_class}")

        # 5. 找到所有属于该类的像素坐标
        coords = torch.nonzero(pred_map == target_class, as_tuple=False)  
        # coords 形状 [N, 2]，每一行是一个 (y, x)

        # 6. 打印总数
        num_points = coords.size(0)
        print(f"分割图中属于 'aeroplane' 的像素点总数：{num_points}")

        # 7. 打印所有坐标
        #    警告：如果点很多，打印会很长。可以按需截断或存文件。
        coords_list = coords.cpu().tolist()  # 转成 Python 列表方便打印
        for idx, (y, x) in enumerate(coords_list):
            print(f"{idx:4d}: (y={y}, x={x})")
        with open("book.txt", "w") as out:
            for y, x in coords_list:
                out.write(f"{y}\t{x}\n")

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device),
                    alpha=0.4,
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
