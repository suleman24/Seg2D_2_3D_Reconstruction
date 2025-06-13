from gradio_client import Client, handle_file
import time
import shutil
import os
import uuid

HFs = ["tok1", "tok2", "tok3"]

def generate_glb_from_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")

    for i in [0, 1, 2]:  # Try first and second token
        try:
            HF_TOKEN = HFs[i]
            print('*********')
            print(f"Using Hugging Face token : {i+1}")
            print('*********')

            #Uncomment before running the code
            # client = Client("crevelop/Trellis", hf_token=HF_TOKEN)

            # Step 1: Preprocess the image
            trial_id, _ = client.predict(
                image=handle_file(image_path),
                api_name="/preprocess_image"
            )

            # Step 2: Generate 3D model
            _ = client.predict(
                trial_id=trial_id,
                seed=0,
                randomize_seed=True,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3,
                slat_sampling_steps=12,
                api_name="/image_to_3d"
            )

            # Step 3: Extract GLB
            time.sleep(5)
            glb_view_path, glb_download_path = client.predict(
                mesh_simplify=0.95,
                texture_size=1024,
                api_name="/extract_glb"
            )

            # Step 4: Save the GLB to the uploads directory
            output_filename = f"model_{uuid.uuid4()}.glb"
            output_path = os.path.join("static/uploads", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(glb_download_path, output_path)

            return output_path

        except Exception as e:
            print(f"Error with token index {i} ({HF_TOKEN}): {e}")
            if i == 2:
                raise RuntimeError("All tokens failed. Cannot generate GLB model.")

