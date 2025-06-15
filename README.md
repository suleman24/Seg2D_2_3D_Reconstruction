# Seg2D_2_3D_Reconstruction

A Flask web app for image segmentation using SAM and 3D model creation with Trellis API, visualized with Three.js. Features

**Application Flow**

The application processes an image from upload to 3D model visualization as follows:
1. Image Upload: User uploads an image via the web interface (index.html).
2. Point Selection: User clicks a point on the image to indicate the object for segmentation.
3. Segmentation: The Flask backend (main.py) uses the Segment Anything Model (SAM) to segment the object at the clicked coordinates, generating a masked image with a transparent background.
4. Display Segmented Image: The segmented image is displayed in the Three.js canvas as a 2D plane with orbit controls.
5. 3D Model Generation: The segmented image is sent to the Trellis API (trellis.py) to generate a 3D GLB model.
6. 3D Visualization: The GLB model is loaded into the Three.js scene, replacing the 2D plane (when user clicks on 'create 3D'), with lighting, shadows, and auto rotation.
7. Download Options: Users can download the segmented image (PNG) or the 3D model (GLB) via buttons in the interface.

**Prerequisites**

Python 3.8+

PyTorch (CUDA-enabled)

Flask, OpenCV, NumPy, Gradio Client

SAM checkpoint (sam_vit_h_4b8939.pth)

Hugging Face token for Trellis API

**Installation**

Clone repo:git clone https://github.com/suleman24/Seg2D_2_3D_Reconstruction

cd Seg2D_2_3D_Reconstruction

Set up virtual environment:python -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies:pip install flask opencv-python torch numpy gradio-client segment-anything

Download SAM checkpoint and place in root.

Update HFs in trellis.py with your Hugging Face token.

Create uploads directory:mkdir -p static/uploads


**Usage**

Run app:python main.py

Open http://localhost:5005 in browser.

Upload image, click to segment, view 3D model, and download results.

**Structure**

main.py: Flask backend for segmentation and 3D model generation.

trellis.py: Interfaces with Trellis API.

index.html: Frontend with Three.js rendering.

static/uploads/: Stores images and GLB files.


**Acknowledgments**

Segment Anything

Trellis API

Three.js


