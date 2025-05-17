import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import (
    AverageMeter,
    LoadImages,
    driving_area_mask,
    increment_path,
    lane_line_mask,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    select_device,
    show_seg_result,
    split_for_trace_model,
    time_synchronized,
    xyxy2xywh,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="data/weights/yolopv2.pt",
        help="model.pt path(s)",
    )
    # file/folder, 0 for webcam
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="source (0 for webcam, or path to video/image)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", type=int, default=3, help="thickness of lane lines"
    )
    parser.add_argument(
        "--height-limit",
        type=float,
        default=0.6,
        help="limit lane detection to bottom portion of image (0.5 = bottom half)",
    )
    parser.add_argument(
        "--center-offset",
        type=float,
        default=0.0,
        help="horizontal offset from image center (-0.5 to 0.5, where 0 is center)",
    )
    parser.add_argument(
        "--center-line-length",
        type=float,
        default=0.1,
        help="length of center reference line as a proportion of image height (0.0-1.0)",
    )
    parser.add_argument(
        "--center-line-position",
        type=float,
        default=0.7,
        help="vertical position of center reference line (0.0-1.0, where 0 is top, 1 is bottom)",
    )
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="show height limit and center reference lines",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="window size for displaying live camera feed [width height]",
    )
    return parser


def extract_ego_lanes(lane_mask, height_limit=0.6, line_thickness=3, center_offset=0.0):
    """
    Find the closest lane markings to the left and right of center,
    limiting detection to the bottom portion of the image.

    Args:
        lane_mask: Binary lane mask
        height_limit: Proportion of image height to consider (from bottom)
        line_thickness: Thickness of lane lines
        center_offset: Horizontal offset from image center (-0.5 to 0.5)

    Returns:
        Mask containing only the ego lanes
    """
    if isinstance(lane_mask, torch.Tensor):
        lane_mask = lane_mask.cpu().numpy()

    # Get mask shape
    if len(lane_mask.shape) == 4:  # Shape: [batch, channel, height, width]
        batch, channel, height, width = lane_mask.shape
        lane_mask = lane_mask[0, 0]  # Take the first image and channel
    else:
        height, width = lane_mask.shape[-2:]

    # Apply height limit mask
    height_threshold = int(height * height_limit)
    height_mask = np.zeros_like(lane_mask)
    height_mask[height_threshold:, :] = 1
    limited_lane_mask = lane_mask * height_mask

    # Binary threshold
    if lane_mask.dtype == np.float32 or lane_mask.dtype == np.float64:
        limited_lane_mask = (limited_lane_mask > 0.5).astype(np.uint8)
    else:
        limited_lane_mask = limited_lane_mask.astype(np.uint8)

    # Get connected components in the lane mask
    labeled_mask, num_components = ndimage.label(limited_lane_mask)
    if num_components == 0:
        return np.zeros_like(lane_mask)

    # Calculate the center point with offset
    # Center offset is a value between -0.5 and 0.5 of the image width
    center_x = width // 2 + int(width * center_offset)

    # Ensure center_x is within image bounds
    center_x = max(0, min(width - 1, center_x))

    # Find all components and their centroids
    components = []
    for i in range(1, num_components + 1):
        # Get the component mask
        component_mask = labeled_mask == i

        # Get coordinates of all points in this component
        component_points = np.where(component_mask)
        if len(component_points[0]) == 0:
            continue

        # Calculate centroid coordinates
        y_coords, x_coords = component_points
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Check if the component is in the bottom part of the image
        if centroid_y >= height_threshold:
            # Store component info
            components.append(
                {
                    "id": i,
                    "mask": component_mask,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "distance_from_center": abs(centroid_x - center_x),
                }
            )

    # Sort components by horizontal distance from center
    components.sort(key=lambda c: c["distance_from_center"])

    # Separate left and right components based on the (potentially offset) center
    left_components = [c for c in components if c["centroid_x"] < center_x]
    right_components = [c for c in components if c["centroid_x"] >= center_x]

    # Sort by x-position (left to right)
    left_components.sort(key=lambda c: c["centroid_x"], reverse=True)  # Rightmost first
    right_components.sort(key=lambda c: c["centroid_x"])  # Leftmost first

    # Get the ego lanes (closest to center on each side)
    ego_lanes_mask = np.zeros_like(lane_mask)

    if left_components:
        # Rightmost left component
        ego_lanes_mask[left_components[0]["mask"]] = 1

    if right_components:
        # Leftmost right component
        ego_lanes_mask[right_components[0]["mask"]] = 1

    # Apply dilation for thicker lines if requested
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        ego_lanes_mask = cv2.dilate(ego_lanes_mask.astype(np.uint8), kernel)

    # Convert back to tensor if the input was a tensor
    if isinstance(lane_mask, torch.Tensor):
        ego_lanes_mask = torch.from_numpy(ego_lanes_mask).to(lane_mask.device)

        # Match the original tensor dimensions
        if len(lane_mask.shape) == 4:
            ego_lanes_mask = ego_lanes_mask.unsqueeze(0).unsqueeze(0)

    return ego_lanes_mask


def detect():
    # setting and directories
    source, weights, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.save_txt,
        opt.img_size,
    )

    # Fix: Use the first element of weights list instead of the list itself
    weight_path = weights[0] if isinstance(weights, list) else weights

    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    is_webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weight_path)
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # For webcam
    if is_webcam:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.view_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.view_size[1])

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # Create window for display
        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Detection", opt.view_size[0], opt.view_size[1])

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert image to model input format
            img = torch.from_numpy(im0).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW

            # Inference
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = model(img)
            t2 = time_synchronized()

            # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
            tw1 = time_synchronized()
            pred = split_for_trace_model(pred, anchor_grid)
            tw2 = time_synchronized()

            # Apply NMS
            t3 = time_synchronized()
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )
            t4 = time_synchronized()

            # Extract lane lines
            ll_seg_mask = lane_line_mask(ll)

            # Extract only ego lanes using the hybrid nets logic with customizable center
            ego_lanes_mask = extract_ego_lanes(
                ll_seg_mask,
                height_limit=opt.height_limit,
                line_thickness=opt.line_thickness,
                center_offset=opt.center_offset,
            )

            # Use driving area mask for visualization background
            da_seg_mask = driving_area_mask(seg)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s = "%gx%g " % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img:  # Add bbox to image
                            plot_one_box(xyxy, im0, line_thickness=3)

                # Print time (inference)
                print(f"{s}Done. ({t2 - t1:.3f}s)")

            da_seg_mask = cv2.resize(
                da_seg_mask,
                (im0.shape[1], im0.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            ego_lanes_mask = cv2.resize(
                ego_lanes_mask,
                (im0.shape[1], im0.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # Show ego lanes with driving area
            show_seg_result(im0, (da_seg_mask, ego_lanes_mask), is_demo=True)

            # Draw reference markers if enabled
            if opt.show_markers:
                # Draw height limit line for visualization
                height_threshold = int(im0.shape[0] * opt.height_limit)
                cv2.line(
                    im0,
                    (0, height_threshold),
                    (im0.shape[1], height_threshold),
                    (255, 255, 255),  # White color
                    1,
                )

                # Draw center reference line for visualization with custom position
                center_x = im0.shape[1] // 2 + int(im0.shape[1] * opt.center_offset)
                center_x = max(0, min(im0.shape[1] - 1, center_x))

                # Calculate the vertical position of the center line
                # center_line_position is from 0.0 (top) to 1.0 (bottom)
                center_y = int(im0.shape[0] * opt.center_line_position)

                # Calculate the start and end points for the center line
                half_length = int(im0.shape[0] * opt.center_line_length / 2)
                # Ensure it doesn't go beyond top
                line_start_y = max(0, center_y - half_length)
                # Ensure it doesn't go beyond bottom
                line_end_y = min(im0.shape[0], center_y + half_length)

                # Draw the center line at the specified position
                cv2.line(
                    im0,
                    (center_x, line_start_y),
                    (center_x, line_end_y),
                    (255, 255, 0),  # Yellow color for center line
                    1,
                )

            # Display the resulting frame
            im0 = cv2.resize(
                im0,
                (opt.view_size[0], opt.view_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow("Lane Detection", im0)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Original video/image processing code
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = model(img)
            t2 = time_synchronized()

            # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
            tw1 = time_synchronized()
            pred = split_for_trace_model(pred, anchor_grid)
            tw2 = time_synchronized()

            # Apply NMS
            t3 = time_synchronized()
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )
            t4 = time_synchronized()

            # Extract lane lines
            ll_seg_mask = lane_line_mask(ll)

            # Extract only ego lanes using the hybrid nets logic with customizable center
            ego_lanes_mask = extract_ego_lanes(
                ll_seg_mask,
                height_limit=opt.height_limit,
                line_thickness=opt.line_thickness,
                center_offset=opt.center_offset,
            )

            # Use driving area mask for visualization background
            da_seg_mask = driving_area_mask(seg)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}"
                )  # img.txt
                s += "%gx%g " % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )  # normalized xywh
                            # label format
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + ".txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img:  # Add bbox to image
                            plot_one_box(xyxy, im0, line_thickness=3)

                # Print time (inference)
                print(f"{s}Done. ({t2 - t1:.3f}s)")

                # Show ego lanes with driving area
                show_seg_result(im0, (da_seg_mask, ego_lanes_mask), is_demo=True)

                # Draw reference markers if enabled
                if opt.show_markers:
                    # Draw height limit line for visualization
                    height_threshold = int(im0.shape[0] * opt.height_limit)
                    cv2.line(
                        im0,
                        (0, height_threshold),
                        (im0.shape[1], height_threshold),
                        (255, 255, 255),  # White color
                        1,
                    )

                    # Draw center reference line for visualization with custom position
                    center_x = im0.shape[1] // 2 + int(im0.shape[1] * opt.center_offset)
                    center_x = max(0, min(im0.shape[1] - 1, center_x))

                    # Calculate the vertical position of the center line
                    # center_line_position is from 0.0 (top) to 1.0 (bottom)
                    center_y = int(im0.shape[0] * opt.center_line_position)

                    # Calculate the start and end points for the center line
                    half_length = int(im0.shape[0] * opt.center_line_length / 2)
                    # Ensure it doesn't go beyond top
                    line_start_y = max(0, center_y - half_length)
                    # Ensure it doesn't go beyond bottom
                    line_end_y = min(im0.shape[0], center_y + half_length)

                    # Draw the center line at the specified position
                    cv2.line(
                        im0,
                        (center_x, line_start_y),
                        (center_x, line_end_y),
                        (255, 255, 0),  # Yellow color for center line
                        1,
                    )

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w, h = im0.shape[1], im0.shape[0]
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                            )
                        vid_writer.write(im0)

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
        print(
            "inf : (%.4fs/frame)   nms : (%.4fs/frame)" % (inf_time.avg, nms_time.avg)
        )
        print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
