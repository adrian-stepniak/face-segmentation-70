from typing import Tuple
import json


def intersection_over_union(
        first_box: Tuple,
        second_box: Tuple
) -> float:
    """ Calculate intersection over union between two bounding boxes.
    Based on: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param first_box: Tuple with first bounding box. e.g. (x1, y1),(x2, y2)
    :param second_box: tuple with second bounding box
    :return: Value of intersection over union
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    first_x = max(first_box[0][0], second_box[0][1])
    first_y = max(first_box[0][1], second_box[0][1])
    second_x = min(first_box[1][0], second_box[1][0])
    second_y = min(first_box[1][1], second_box[1][1])

    # compute the area of intersection rectangle
    inter_area = max(0, second_x - first_x + 1) * max(0, second_y - first_y + 1)

    # compute the area of both the prediction and ground-truth rectangles
    first_box_area = (first_box[1][0] - first_box[0][0] + 1) * (first_box[1][1] - first_box[0][1] + 1)
    second_box_area = (second_box[1][0] - second_box[0][0] + 1) * (second_box[1][1] - second_box[0][1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the interesection area
    iou = inter_area / float(first_box_area + second_box_area - inter_area)

    return iou


def get_bounding_box(label: dict) -> Tuple:
    """ Extract rectangle and path to image from label from json file
    :param label: label from json file
    :return: rectangle and path to image
    """
    path = label['path']
    min_box = label['annotations']['face'][0]['data']['min']
    max_box = label['annotations']['face'][0]['data']['max']

    image_width = label['metadata']['image']['width']
    image_height = label['metadata']['image']['height']

    x1 = image_width * min_box[0]
    y1 = image_height * min_box[1]
    x2 = image_width * max_box[0]
    y2 = image_height * max_box[1]

    return ((x1, y1), (x2, y2)), path


def process_json(json_path: str) -> dict:
    """ Convert json with labels to python dictionary
    :param json_path: path to the json file
    :return: Dict with path to image as key and rectangle as value
    """
    with open(json_path, 'r') as json_file:
        labels = json.load(json_file)
        labels = labels['labels']

    paths_and_rectanges = {}

    for label in labels:
        rectangle, path = get_bounding_box(label)
        paths_and_rectanges[path] = rectangle

    return paths_and_rectanges
