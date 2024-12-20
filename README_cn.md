## 连续开放域物体检测代码使用指南
### 一、概述

1. 本工作以 [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) 为物体检测架构，并基于第三方实现的 [Open-Grounding-DINO训练代码](https://github.com/longzw1997/Open-GroundingDino) 再次开发。
2. 本工作以 Grounding-DINO 发布的预训练模型参数为训练起点来做开放域类增量微调，最终实现的效果是：实现对增量任务中涉及到的新类别物体的检测（即微调后模型对这些新类的检测性能相比原始的Grounding-DINO 预训练模型更高），同时维持维持了其开放域检测能力（即微调后的模型在其他开放域类别上的检测性能可以和原始的Grounding-DINO 预训练模型持平）。


### 二、环境准备

请参考 [Open-Grounding-DINO](https://github.com/longzw1997/Open-GroundingDino) 中的 **Setup** 来安装所需环境。

### 三、预训练模型参数准备
本工作在做实验时主要是以 Grounding-DINO 官方发布的 [GroundingDINO-T](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) 预训练模型为初始参数。您可通过上述链接下载模型参数文件，并将其参数文件放在本项目的根目录。另外，您也可以在您的业务中使用更大的backbone模型，并下载 [GroundingDINO-B](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth) 预训练参数 （后续需手动修改config中预训练模型路径）

### 四、数据集准备
1. 需要先确保您的业务数据的训练/测试集的annoations是符合COCO物体检测数据集的标注规范 (COCO-style，json格式)。
2. 修改 ```tools/custom2odvg.py``` 文件的第8行的CLASS_NUM变量，修改其的值为您业务中新task中新类别的数目。
3. 运行 ```tools/custom2odvg.py```来转化您原始的COCO-style的annotation文件：```python tools/custom2odvg.py --input YOUR_CUSTOM_COCO_STYLE_TRAIN_ANNO_FILE_PATH --output TRANSFERD_ANNO_FILE_NAME.json```。其中，YOUR_CUSTOM_COCO_STYLE_TRAIN_ANNO_FILE_PATH 为您原始的COCO-style 训练所用的annotation文件路径，TRANSFERD_ANNO_FILE_NAME 为转化后的annotation文件名，注意转换后的文件格式也应为json格式。
4. 将上述转化后的annotation文件移动至本项目路径下的```config``` 文件夹中。
5. 在```config```路径下新建一个json类型的文件，这里用 ```CUSTOM_label_map.json``` 指代。打开该文件，将您业务的新类别 id 和对应的类别名填到里面，如下所示：
    ```json
    {
        "0":"dog",
        "1":"cat",
        "2":"bird",
        ...
    }
    ```
    注意id为字符串格式，且从0开始。类别名和id的对应关系要和您```TRANSFERD_ANNO_FILE_NAME.json``` 一致。
6. 在```config```路径下再新建一个json类型的文件，这里用 ```train_CUSTOM.json``` 指代。打开该文件，填入以下内容：
    ```json
    {
        "train": [
            {
                "root": "TRAIN_IMAGES_ROOT",
                "anno": "/ABSOLUTE/PATH/OF/TRANSFERD_ANNO_FILE_NAME.json",
                "label_map": "/ABSOLUTE/PATH/OF/CUSTOM_label_map.json",
                "dataset_mode": "odvg"
            }
        ],
        "val": [
            {
                "root": "TEST_IMAGES_ROOT",
                "anno": "/ABSOLUTE/PATH/OF/YOUR_CUSTOM_COCO_STYLE_TEST_ANNO_FILE_PATH",
                "label_map": "/ABSOLUTE/PATH/OF/CUSTOM_label_map.json",
                "dataset_mode": "coco"
            }
        ]
    }
     ```
    其中，```TRAIN_IMAGES_ROOT``` 是您训练图像所在文件夹的路径；```/ABSOLUTE/PATH/OF/TRANSFERD_ANNO_FILE_NAME.json``` 是 ```TRANSFERD_ANNO_FILE_NAME.json``` 的绝对路径； ```/ABSOLUTE/PATH/OF/CUSTOM_label_map.json``` 是 CUSTOM_label_map.json 的绝对路径；```TEST_IMAGES_ROOT``` 是您测试图像所在文件夹的路径；```"/ABSOLUTE/PATH/OF/YOUR_CUSTOM_COCO_STYLE_TEST_ANNO_FILE_PATH"``` 是您测试图像对应的coco-style的annotation文件。

### 五、模型的训练
1. 打开该项目路径下的 ```start_train.sh``` 文件，修改第5行为 ```train_CUSTOM.json``` 文件的实际名字和路径，修改第6行具体的模型和log保存路径。
2. 运行 ```start_train.sh```  文件开始训练: ```bash start_train.sh```
3. 若有多个新task，则按照 “**四、数据集准备**” 中的说明，为每个task都维护 ```TRANSFERD_ANNO_FILE_NAME```，```CUSTOM_label_map.json``` 和```train_CUSTOM.json```。并在训练启动脚本中依次添加训练每个task的训练代码，如项目路径下的 ```start_train_more_task.sh``` 所示。

### 六、模型的推理
模型训练完毕后，将会在 ```start_train.sh``` 中的 OUTPUT_DIR 路径下生成模型训练后的checkpoint。
#### 6.1 task key的生成
模型在推理时，本工作基于retrieval的思想来动态选择所训练的模型权重，因此首先为每个task维护一个task key，具体为每个task训练集的图像特征的平均值。步骤如下：
1. 首先来用模型提取每个task的每张训练图像的特征。打开本项目路径下的 ```extract_feat_scripts\extract_feat.sh``` 文件，第5和第7行的参数填该task的coco-style的训练annotation文件路径，第6行参数填该task的训练集图像所在的文件夹路径。第8行参数填您所希望生成feature的路径，我们建议该路径由两层路径构成，第一层为总的 ```FEATRURE_SAVE_ROOT```，第二层为具体对应每个 task 的 ```TASK_NAME``` ，以便后续代码读取。
2. 运行 ```extract_feat_scripts\extract_feat.sh```: ```bash extract_feat_scripts\extract_feat.sh```，等待模型提取每个训练图像的特征，并自动保存到上述填写的保存路径。
3. 重复以上步骤，直到每个task的feature都成功提取并保存
4. 打开本项目根目录下的 ```cal_mean_feature_per_task.py``` 文件，修改第6行变量为您第1步设置的```FEATRURE_SAVE_ROOT```，修改第8行列表元素，分别为您第1步设置的每个task的保存路径名 ```TASK_NAME```。
5. 运行 ```cal_mean_feature_per_task.py```： ```python cal_mean_feature_per_task.py```，所有task的task key将由dict的形式统一保存在 ```FEATRURE_SAVE_ROOT/mean_feat/task_feats.pkl``` 。 

#### 6.2 每个task的PEFT模块提取
模型在推理时，本工作基于retrieval的思想来动态选择所训练的模型参数。为方便模型在线且高效地动态读取各个模型的参数，可先离线地将各个task所训模型的PEFT模块参数统一提取并保存。步骤如下：
1. 打开本项目根目录下的 ```save_prompt_pth.py``` 文件，修改第6行的 ```subtask_key```列表元素，分别为6.1中每个task的feature保存路径名。修改14行的 ```ckpt_list``` 列表元素，分别为 **五**中每个task训练时保存ckpt的路径。
2. 运行 ```save_prompt_pth.py``` 文件： ```python save_prompt_pth.py```, 得到 ```multi_model_prompt_params.pkl```。
3. 打开本项目根目录下的 ```save_lora_pth.py``` 文件，修改第6行的 ```subtask_key```列表元素，分别为6.1中每个task的feature保存路径名。修改14行的 ```ckpt_list``` 列表元素，分别为 **五**中每个task训练时保存ckpt的路径。
4. 运行 ```save_lora_pth.py``` 文件： ```python save_lora_pth.py```, 得到 ```multi_model_lora_params.pkl```。

#### 6.3 执行模型推理
1. 打开本项目目录下的 ```eval_one_image.sh```,  修改第4行参数值 ```-p``` 为您所训练的最后一个task的checkpoint路径，修改第5行```--image_path``` 为您想推理的图像路径，修改第6行 ```--text_prompt``` 为您业务中类别空间中类别名所组成的prompt （类别名用英文句号隔开，句号和前后类别需要有一个空格，且prompt的末尾最后一个字符是英文句号, 例如 ```"CLASS_NAME1 . CLASS_NAME2 . CLASS_NAME3 ."```），修改第12行的 ```-o``` 为您想要设置的可视化输出路径。
2. 运行 ```tools/inference_on_a_image.py``` 文件，执行： ```python tools/inference_on_a_image.py```，运行完毕后，可视化的检测图像将保存到 ```OUTPUT_DIR/pred.jpg```。
3. 您可以修改```tools/inference_on_a_image.py``` 中298行的```retrieval_tau``` 参数值来调整算法的效果。