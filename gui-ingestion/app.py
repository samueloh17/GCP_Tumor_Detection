import os
import uuid
import nibabel as nb
import numpy as np
from flask import Flask, request, jsonify, render_template
from google.cloud import storage
import io
import imageio
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_utils.logger_config import get_logger



EPSILON = 1e-8

# Instanciar el logger para este servicio específico
logger = get_logger("ingestion_service")



app = Flask(__name__)

# CONFIGURACIÓN DINÁMICA
# Si no encuentra la variable, por defecto es 'LOCAL'
ENV = os.getenv("APP_ENV", "LOCAL")
BUCKET_NAME = os.getenv("BUCKET_NAME", "mi-bucket-tesis")
LOCAL_STORAGE_PATH = "local_storage" # Carpeta para pruebas en tu PC

# Crear carpeta local si no existe
if ENV == "LOCAL" and not os.path.exists(LOCAL_STORAGE_PATH):
    os.makedirs(LOCAL_STORAGE_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/upload', methods=['POST'])
def upload_nifti():
    
    file = request.files.get('file')
    study_id = str(uuid.uuid4())
    logger.info(f"THE PROCESS FOR THE {study_id} STARTS")
    
    temp_nii = f"temp_{study_id}.nii"
    file.save(temp_nii)
    
    try:
        data = nb.load(temp_nii)
        imgs = data.get_fdata()
        
        if ENV == "CLOUD":
            logger.info("PROCES ON CLOUD ENV")
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            for i, img in enumerate(imgs, start = 1):
                img_buffer = io.BytesIO()
                img_min = np.min(img)
                img_max = np.max(img)
                img_range = img_max - img_min
                if img_range == 0:
                    img_final = np.zeros(img.shape, dtype=np.uint8)
                    logger.warning(f"Slice {i} is empty or constant. Generating black image.")
                else:
        
                    img_norm = (img - img_min) / (img_range + EPSILON)
                    img_final = (img_norm * 255).astype(np.uint8)

                img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
                img_final = (img_norm * 255).astype(np.uint8)
                imageio.imsave(img_buffer, img_final, format = "PNG")
                img_buffer.seek(0)
                blob = bucket.blob(f"staging/{study_id}/slice_{i}.npy")
                blob.upload_from_file(img_buffer,content_type="imge/png")
                pass
            logger.info("PROCESS COMPLETED SUCCESSS")
            return jsonify({"status": "Cloud Upload Success", "id": study_id})
        
        else:
            logger.info("PROCES ON LOCAL ENV")
            study_folder = os.path.join(LOCAL_STORAGE_PATH, study_id)
            os.makedirs(study_folder)
            
            for i, img in enumerate(imgs, start = 1):
                img_buffer = io.BytesIO()
                img_min = np.min(img)
                img_max = np.max(img)
                img_range = img_max - img_min
                if img_range == 0:
                    img_final = np.zeros(img.shape, dtype=np.uint8)
                    logger.warning(f"Slice {i} is empty or constant. Generating black image.")
                else:
                    img_norm = (img - img_min) / (img_range + EPSILON)
                    img_final = (img_norm * 255).astype(np.uint8)

                    
                slice_path = os.path.join(study_folder, f"slice_{i}.png")
                imageio.imsave(slice_path,img_final, format="PNG")
            
            logger.info("PROCESS COMPLETED SUCCESSS")
            return jsonify({
                "status": "Local Save Success",
                "path": os.path.abspath(study_folder),
                "slices": data.shape[2]
            })
        

    except Exception as e : 
        logger.error(f"CRITICAL ERROR IN STUDY {study_id}, {e}",exc_info = True)

    finally:
        if os.path.exists(temp_nii):
            os.remove(temp_nii)
            logger.info(f"REMOVE THE TEMPROAL {temp_nii}")

    
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

