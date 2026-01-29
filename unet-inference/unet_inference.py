import argparse
import os
import io
import numpy as np
import tensorflow as tf
from google.cloud import storage, aiplatform
from PIL import Image
import imageio.v3 as iio

def run_inference(project_id, location, bucket_images, study_id, model_resource_name):
    print(f"🚀 Iniciando inferencia para el estudio: {study_id}")
    
    aiplatform.init(project=project_id, location=location)
    
    
    model = aiplatform.Model(model_name=model_resource_name)
    model_uri = model.uri
    
    print(f"📦 Cargando modelo desde Registry: {model_resource_name}")
    loaded_model = tf.keras.models.load_model(model_uri)

    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_images)
    prefix = f"staging/{study_id}/"
    blobs = bucket.list_blobs(prefix=prefix)

    
    processed_count = 0
    for blob in blobs:
        if not blob.name.lower().endswith('.png'):
            continue
            
        
        content = blob.download_as_bytes()
        img = iio.imread(io.BytesIO(content))

        
        img_pil = Image.fromarray(img).convert('L').resize((256, 256))
        img_input_norm = np.array(img_pil) / 255.0
        max_on_img = np.max(img_input_norm)
        

       
        if max_on_img > 0 :
            img_input = np.expand_dims(img_input, axis=(0, -1))
            prediction = loaded_model.predict(img_input, verbose=0)
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

        else: 
            mask = np.zeros((256,256),dtype=np.unit8)

        mask_binary = (mask/255).astype(np.unit8)  
        img_segmented = (img_input* mask_binary).astype(np.unit8)

        mask_path = blob.name.replace("staging/", "results/").replace(".png", "_mask.png")
        
        img_seg_path = blob.name.replace("staging/","results_seg/").replace(".png","_seg.png")

        output_blob_mask = bucket.blob(mask_path)
        output_blob_seg = bucket.blob(img_seg_path)
        buffer_mask = io.BytesIO()
        buffer_seg = io.BytesIO()
        Image.fromarray(mask).save(buffer_mask, format="PNG")
        Image.fromarray(img_segmented).save(buffer_seg, format="PNG")

        output_blob_mask.upload_from_string(buffer_mask.getvalue(), content_type="image/png")
        output_blob_seg.upload_from_string(buffer_seg.getvalue(), content_type="image/png")
        
        processed_count += 1
        print(f"✅ Máscara guardada: {mask_path}")
        print(f"✅ Segmentación guardada: {mask_path}")
    print(f"🏁 Proceso terminado. Total de imágenes procesadas: {processed_count}")

if __name__ == "__main__":
    # Configuración de argumentos para que el contenedor sea parametrizable
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--bucket_images', type=str, required=True)
    parser.add_argument('--study_id', type=str, required=True)
    parser.add_argument('--model_resource_name', type=str, required=True)
    
    args = parser.parse_args()
    
    run_inference(
        project_id=args.project_id,
        location=args.location,
        bucket_images=args.bucket_images,
        study_id=args.study_id,
        model_resource_name=args.model_resource_name
    )