## imports

import collections
import time
import cv2
import numpy as np
from openvino import runtime as ov
import operator
from urllib import parse
import html_reader as reader
import tokens_bert as tokens
import threading

# models
detect_model_path = "project/detection_model/ssdlite_mobilenet_v2_fp16.xml"
wiki_model_path = "project/wiki_model/intel/bert-small-uncased-whole-word-masking-squad-int8-0002/FP16-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.xml"

## load the models

# Initialize OpenVINO Runtime.
ie_core = ov.Core()
# Read the network and corresponding weights from a file.
detect_model = ie_core.read_model(detect_model_path)
wiki_model = ie_core.read_model(wiki_model_path)
# Compile the model for CPU (you can choose manually CPU, GPU etc.)
# or let the engine choose the best available device (AUTO).
compiled_detect_model = ie_core.compile_model(model=detect_model, device_name="CPU")
compiled_wiki_model = ie_core.compile_model(model=wiki_model, device_name="CPU")

# Get the input and output nodes.
input_layer = compiled_detect_model.input(0)
output_layer = compiled_detect_model.output(0)

# Get input and output names of nodes. ++
input_keys = list(compiled_wiki_model.inputs)
output_keys = list(compiled_wiki_model.outputs)

# Get the input size.
height, width = list(input_layer.shape)[1:3]
input_layer.any_name, output_layer.any_name

# Get the network input size. ++
input_size = compiled_wiki_model.input(0).shape[1]
[i.any_name for i in input_keys], [o.any_name for o in output_keys]

## global variables

# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]

# Colors for the classes above (Rainbow Color Map).
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

answers = [''] * len(classes)
thread_list = [''] * len(classes)

## processing ######################## interactive question answering model

# The path to the vocabulary file.
vocab_file_path = "project/wiki_model/vocab.txt"

# Create a dictionary with words and their indices.
vocab = tokens.load_vocab_file(vocab_file_path)

# Define special tokens.
cls_token = vocab["[CLS]"]
pad_token = vocab["[PAD]"]
sep_token = vocab["[SEP]"]

# A function to load text from given urls.
def load_context(sources):
    input_urls = []
    paragraphs = []
    for source in sources:
        result = parse.urlparse(source)
        if all([result.scheme, result.netloc]):
            input_urls.append(source)
        else:
            paragraphs.append(source)

    paragraphs.extend(reader.get_paragraphs(input_urls))
    # Produce one big context string.
    return "\n".join(paragraphs)


## preprocessing

# A generator of a sequence of inputs.
def prepare_input(question_tokens, context_tokens):
    # A length of question in tokens.
    question_len = len(question_tokens)
    # The context part size.
    context_len = input_size - question_len - 3

    if context_len < 16:
        raise RuntimeError("Question is too long in comparison to input size. No space for context")

    # Take parts of the context with overlapping by 0.5.
    for start in range(0, max(1, len(context_tokens) - context_len), context_len // 2):
        # A part of the context.
        part_context_tokens = context_tokens[start:start + context_len]
        # The input: a question and the context separated by special tokens.
        input_ids = [cls_token] + question_tokens + [sep_token] + part_context_tokens + [sep_token]
        # 1 for any index if there is no padding token, 0 otherwise.
        attention_mask = [1] * len(input_ids)
        # 0 for question tokens, 1 for context part.
        token_type_ids = [0] * (question_len + 2) + [1] * (len(part_context_tokens) + 1)

        # Add padding at the end.
        (input_ids, attention_mask, token_type_ids), pad_number = pad(input_ids=input_ids,
                                                                      attention_mask=attention_mask,
                                                                      token_type_ids=token_type_ids)

        # Create an input to feed the model.
        input_dict = {
            "input_ids": np.array([input_ids], dtype=np.int32),
            "attention_mask": np.array([attention_mask], dtype=np.int32),
            "token_type_ids": np.array([token_type_ids], dtype=np.int32),
        }

        # Some models require additional position_ids.
        if "position_ids" in [i_key.any_name for i_key in input_keys]:
            position_ids = np.arange(len(input_ids))
            input_dict["position_ids"] = np.array([position_ids], dtype=np.int32)

        yield input_dict, pad_number, start


# A function to add padding.
def pad(input_ids, attention_mask, token_type_ids):
    # How many padding tokens.
    diff_input_size = input_size - len(input_ids)

    if diff_input_size > 0:
        # Add padding to all the inputs.
        input_ids = input_ids + [pad_token] * diff_input_size
        attention_mask = attention_mask + [0] * diff_input_size
        token_type_ids = token_type_ids + [0] * diff_input_size

    return (input_ids, attention_mask, token_type_ids), diff_input_size


## postprocessing

# Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L163
def postprocess(output_start, output_end, question_tokens, context_tokens_start_end, padding, start_idx):

    def get_score(logits):
        out = np.exp(logits)
        return out / out.sum(axis=-1)

    # Get start-end scores for the context.
    score_start = get_score(output_start)
    score_end = get_score(output_end)

    # An index of the first context token in a tensor.
    context_start_idx = len(question_tokens) + 2
    # An index of the last+1 context token in a tensor.
    context_end_idx = input_size - padding - 1

    # Find product of all start-end combinations to find the best one.
    max_score, max_start, max_end = find_best_answer_window(start_score=score_start,
                                                            end_score=score_end,
                                                            context_start_idx=context_start_idx,
                                                            context_end_idx=context_end_idx)

    # Convert to context text start-end index.
    max_start = context_tokens_start_end[max_start + start_idx][0]
    max_end = context_tokens_start_end[max_end + start_idx][1]

    return max_score, max_start, max_end


# Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L188
def find_best_answer_window(start_score, end_score, context_start_idx, context_end_idx):
    context_len = context_end_idx - context_start_idx
    score_mat = np.matmul(
        start_score[context_start_idx:context_end_idx].reshape((context_len, 1)),
        end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
    )
    # Reset candidates with end before start.
    score_mat = np.triu(score_mat)
    # Reset long candidates (>16 words).
    score_mat = np.tril(score_mat, 16)
    # Find the best start-end pair.
    max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
    max_score = score_mat[max_s, max_e]

    return max_score, max_s, max_e


def get_best_answer(question, context):
    # Convert the context string to tokens.
    context_tokens, context_tokens_start_end = tokens.text_to_tokens(text=context.lower(),
                                                                     vocab=vocab)
    # Convert the question string to tokens.
    question_tokens, _ = tokens.text_to_tokens(text=question.lower(), vocab=vocab)

    results = []
    # Iterate through different parts of the context.
    for network_input, padding, start_idx in prepare_input(question_tokens=question_tokens,
                                                           context_tokens=context_tokens):
        # Get output layers.
        output_start_key = compiled_wiki_model.output("output_s")
        output_end_key = compiled_wiki_model.output("output_e")

        # OpenVINO inference.
        result = compiled_wiki_model(network_input)
        # Postprocess the result, getting the score and context range for the answer.
        score_start_end = postprocess(output_start=result[output_start_key][0],
                                      output_end=result[output_end_key][0],
                                      question_tokens=question_tokens,
                                      context_tokens_start_end=context_tokens_start_end,
                                      padding=padding,
                                      start_idx=start_idx)
        results.append(score_start_end)

    # Find the highest score.
    answer = max(results, key=operator.itemgetter(0))
    # Return the part of the context, which is already an answer.
    return context[answer[1]:answer[2]], answer[0]


## main processing function

def run_question_answering(label, thread_list):
    keyword = classes[label]
    context = load_context(["https://en.wikipedia.org/wiki/" + keyword])

    if len(context) == 0:
        print("Error: Empty context or outside paragraphs")
        return

    if keyword is not None:
        answer, score = get_best_answer(question="definition of " + keyword, context=context)
        global answers
        answers[label] = answer
        return answer

    else:
        while True:
            question = input()
            # if no question - break
            if question == "":
                break
            # measure processing time
            answer, score = get_best_answer(question=question, context=context)
            return answer


## processing results ######################## live object detection model


def process_results(frame, results, thresh=0.6):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
    )

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Choose color for the label.
        color = tuple(map(int, colors[label]))
        # Draw a box.
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

        if answers[label] == '' and thread_list[label] == '':
            t1 = threading.Thread(target=run_question_answering,args=(label, thread_list))    
            thread_list[label] = t1
            t1.start()
            answer = answers[label]
        else:
            answer = answers[label]
        # Draw a info and label name inside the box.
        cv2.putText(
            img=frame,
            text = answer,
            org=(box[0] + 10, box[1] + 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1500,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame

## main processing function

# Main processing function to run object detection.
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    try:
        # Create a video player to play with target fps.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        w = 640#1280#1920
        h = 480#720#1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Start capturing.
        
        processing_times = collections.deque()

        while True:
            # Grab the frame.
            
            ret, frame = cap.read()
            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Resize the image and change dims to fit neural network input.
            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            # Create a batch of images (size = 1).
            input_img = input_img[np.newaxis, ...]

            # Measure processing time.

            start_time = time.time()
            # Get the results.
            results = compiled_detect_model([input_img])[output_layer]
            stop_time = time.time()
            # Get poses from network results.
            boxes = process_results(frame=frame, results=results)
            
            # Draw boxes on a frame.
            frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname="Press ESC to Exit", mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                cv2.imshow("camera", encoded_img)

    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            for th in thread_list:
                th.join()
            cap.release()
            cv2.destroyAllWindows()

run_object_detection(source=0, flip=False, use_popup=True)
