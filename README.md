# ZooMiami-Trapdoor-Spider-AI
##### **SPIDER TIME-LAPSE AI — Standard Operating Procedures**

\--------------------------------------------------------------------------------------

###### SECTION A — Launching the Spider Time-Lapse App

Follow these steps each time you need to open the spider\_timelapse\_app.



**Step 1 — Open Anaconda Prompt**

Press the Windows key, search for Anaconda Prompt, and click to open it.



**Step 2 — Run the App**

Copy and paste each line below, pressing Enter after each:

conda env list

conda activate yolo26-env

cd C:\\Users\\%USERPROFILE%\\Documents\\yolo

python spider\_timelapse\_app.py



Note: conda env list confirms the environment exists before activating it.

Warning: Do not skip the activate step, running Python outside the environment will cause errors.

\--------------------------------------------------------------------------------------



###### SECTION B — Training a New YOLO AI Model

**Step 1 — Install Anaconda**

Go to [https://www.anaconda.com/download](https://www.anaconda.com/download), download the Windows installer, run it, and follow the on-screen prompts. Once complete, open Anaconda Prompt from the Start menu.



**Step 2 — Create a New Environment and Install Dependencies**

Run the following commands one by one in Anaconda Prompt:

conda create --name yolo26-env python=3.12 -y

conda activate yolo26-env

pip install ultralytics

pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(torch.cuda.get\_device\_name(0))"

pip install label-studio

label-studio start



**Step 3 — Gather and Label Images**

Label Studio will open in your browser. Create a project, upload your images, and label them:

* Draw a tight bounding box around each object in every image
* Make sure the box fits the object closely — avoid excessive padding
* When finished, export 'YOLO with Images' the dataset from Label Studio


The exported .zip file will contain:

* images/ — your image files
* labels/ — YOLO-format annotation .txt files
* classes.txt — the list of your label class names



Extract the .zip file into a folder named my\_dataset.

**Step 4 — Set Up the Folder Structure**

**4.1 — Create the Root Working Directory**

Run these commands in Anaconda Prompt (still inside yolo26-env):
mkdir %USERPROFILE%\\Documents\\yolo

cd %USERPROFILE%\\Documents\\yolo

mkdir data


Note: %USERPROFILE% automatically points to your Windows user folder (e.g., C:\\Users\\Evan).


**4.2 — Split the Dataset into Train and Validation Sets**

Download the automated split script, then run it pointing to your dataset folder:

curl --output train\_val\_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train\_val\_split.py

python train\_val\_split.py --datapath="C:\\\\Users\\\\ChichiPepe\\\\Documents\\\\my\_dataset" --train\_pct=.8

This step may take a while. Do not continue until the prompt is ready for a new command (cursor blinking on an empty line).



**Step 5 — Configure the Training YAML File**

Open Notepad, paste the template below, make the edits listed, then save the file as data.yaml inside your C:\\Users<username>\\Documents\\yolo folder.


path: C:\\Users\\<username>\\Documents\\yolo\\data

train: train\\images

val: validation\\images

nc: 5

names: \["class1", "class2", "class3", "class4", "class5"]


Required edits before saving:

* Change path: to your actual data folder (e.g., C:\\Users\\Evan\\Documents\\yolo\\data)
* Change nc: to the number of classes you labeled (e.g., nc: 4)
* Replace the names: list with your actual class names in the same order as classes.txt (e.g., \["penny","nickel","dime","quarter"])




**Step 6 — Train the Model**

Run the following command from inside the yolo directory:

yolo detect train data=data.yaml model=yolo11m.pt epochs=40 imgsz=640 batch=8




Parameter reference:

* model=yolo11m.pt — Best balance of accuracy vs. speed for wildlife detection
* epochs=40 — Ideal for 1,000+ image datasets; avoids overfitting
* imgsz=640 — More manageable resolution for a 6GB GPU
* batch=8 — Reduces VRAM usage to prevent out-of-memory errors


Training takes a long time. Keep your computer on and do not let it sleep (I recommend to install an Auto Clicker). Trained weights are saved to: runs/detect/train/weights/best.pt


**Step 7 — Test the Trained Model**

First, download the inference script:
curl -o yolo\_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo\_detect.py



Then run detection on your desired source:

Test on a live webcam:

python yolo\_detect.py --model=runs/detect/train/weights/best.pt --source=usb0



Test on an image:

python yolo\_detect.py --model=runs/detect/train/weights/best.pt --source=your\_image.jpg



Test on a video:

python yolo\_detect.py --model=runs/detect/train/weights/best.pt --source=your\_video.mp4

\--------------------------------------------------------------------------------------



###### SECTION C — Fine-Tuning an Existing Model

Use this section when you want to improve a model that has already been trained. For example, when detection is missing certain objects or producing too many false positives. Fine-tuning resumes from your best checkpoint rather than starting from scratch.

When to Fine-Tune vs. Retrain from Scratch

Fine-tune when:

* The model mostly works but needs improvement
* You're adding new images of existing classes
* You're fixing specific missed detections
* You want to save time



Retrain from scratch (Section B) when:

* Starting a brand new project
* Changing or adding new object classes
* Labels or folder structure changed significantly
* Model performance is fundamentally poor




**Step 1 — Add New or Corrected Images**

Before fine-tuning, update your dataset with images that teach the model what it was getting wrong:

* Open Label Studio and add new images to your existing project
* Re-label any images where the model made mistakes
* Export the updated dataset as a new .zip file
* Extract it and re-run the train/val split (same as Section B, Step 4.2)



**Step 2 — Fine-Tune from Your Best Checkpoint**

Navigate to your yolo folder in Anaconda Prompt:

conda activate yolo26-env

cd %USERPROFILE%\\Documents\\yolo



Then run the fine-tune command:

yolo detect train data=data.yaml model=runs/detect/train/weights/best.pt epochs=20 imgsz=640 batch=8 lr0=0.001



Key differences from full training:

* model= points to your previous best.pt instead of a base checkpoint
* epochs=20 is enough — fewer epochs avoids overwriting what the model already learned
* lr0=0.001 uses a lower learning rate so existing weights are adjusted gradually, not overwritten


Each fine-tune run saves to a new folder — runs/detect/train2, train3, etc. Always check which folder your new best.pt is in.


**Step 3 — Verify Improvement**

Test the new model the same way as Section B, Step 7 — but point to the updated weights:

python yolo\_detect.py --model=runs/detect/train2/weights/best.pt --source=your\_test\_video.mp4



*QUICK REFERENCE — Fine-Tune Cheat Sheet*

Copy and paste this block whenever you need to fine-tune quickly:

\# 1. Activate environment

conda activate yolo26-env



\# 2. Navigate to yolo folder

cd %USERPROFILE%\\Documents\\yolo



\# 3. Fine-tune from last best weights

yolo detect train data=data.yaml model=runs/detect/train/weights/best.pt epochs=20 imgsz=640 batch=8 lr0=0.001



\# 4. Test the new model

python yolo\_detect.py --model=runs/detect/train2/weights/best.pt --source=your\_video.mp4



Warning: Always check which train# folder your new weights saved to before testing. YOLO increments the folder number with each run.

\--------------------------------------------------------------------------------------


