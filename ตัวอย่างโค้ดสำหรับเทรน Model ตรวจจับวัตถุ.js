{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45e29dfd",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "#สำหรับเทรน model\n",
    "!pip install ultralytics\n",
    "\n",
    "from ultralytics import YOLO\n",
    "from pathlib import Path\n",
    "\n",
    "model = YOLO(\"yolov8n.pt\")\n",
    "path = Path(r\"/content/drive/MyDrive/DavidGod/warning.v1i.yolov8/data.yaml\") #Set path .yaml\n",
    "\n",
    "\n",
    "if not path.exists():\n",
    "    raise FileNotFoundError(f\"dataset not found\")\n",
    "else:\n",
    "    print(f\"Found {path}\")\n",
    "\n",
    "result = model.train(\n",
    "    data=path,\n",
    "    epochs=150,      # ปรับให้โมเดลมีเวลาเรียนรู้มากขึ้นแต่ไม่นานจน Overfit\n",
    "    imgsz=640,       \n",
    "    batch=-1,        \n",
    "    lr0=0.001,       \n",
    "    lrf=0.01,        \n",
    "    optimizer='AdamW', \n",
    "    augment=True,    \n",
    "    val=True,\n",
    "    plots=True\n",
    ")\n",
    "\n",
    "result = model.val()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
