python model_vqa.py \
    --model-path /data/anomaly/llava_ckpt/llava-v1.5-13b-task-lora \
    --model-base liuhaotian/llava-v1.5-13b \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /data/anomaly/llava_result/
