import argparse
import torch
import os
import sys
import json
from tqdm import tqdm
import shortuuid
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
sys.path.insert(1, '/root/LLaVA-anomaly/llava/train/nsa')
from nsa_loader_test import get_nsa_dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


prompt_base =  "The corresponding images are arranged in the form of four. The left three represent normal sample images and the rightmost one represents the test image. Compared to the three sample images, is there an anomaly in the test image? Answer 0 if not, and 1 if there is."


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    categories = get_nsa_dataset()
    categories = get_chunk(categories, args.num_chunks, args.chunk_idx)

    t = time.localtime()
    current_time = time.strftime("%H%M%S", t)

    answers_file = os.path.expanduser(args.answers_file + current_time + 'jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for ct in tqdm(categories):
        questions = [{"question_id": ct["test_image_id"][i],
                      "image": ct["image"][i]} 
                      for i in range(len(ct["label"]))]
        for line in tqdm(questions):
            idx = line["question_id"]
            image = line["image"]
            qs = prompt_base
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print("[jslee] prompt: ", prompt)

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            #print("[jslee] image: ", image)
            #print("[jslee] image_tensor: ", image_tensor)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            print("[jslee] logits: ", output_ids.logits)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("[jslee] input_ids: ", output_ids)
            print("[jslee] outputs: ", outputs)
            break
            
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": outputs,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
