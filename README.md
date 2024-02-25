
#### 1. Download and unzip the dataset from official url of VQA: https://visualqa.org/download.html.

```bash
$ cd basic_vqa/utils
$ chmod +x download_and_unzip_datasets.csh
$ ./download_and_unzip_datasets.csh
```

#### 2. Preproccess input data for (images, questions and answers).

```bash
$ python resize_images.py --input_dir='../datasets/Images' --output_dir='../datasets/Resized_Images'  
$ python make_vacabs_for_questions_answers.py --input_dir='../datasets'
$ python build_vqa_inputs.py --input_dir='../datasets' --output_dir='../datasets'
```

#### 3. Train model for VQA task.

```bash
$ cd ..
$ python train.py
```

## Results

- Comparison Result

| Model | Metric | Dataset | Accuracy | Source |
| --- | --- | --- | --- | --- |
| Paper Model | Open-Ended | VQA v2 | 54.08 | [VQA Challenge](https://visualqa.org/roe.html) |
| My Model | Multiple Choice | VQA v2 | **54.72** | |


- Loss and Accuracy on VQA datasets v2

![train1](./png/train.png)