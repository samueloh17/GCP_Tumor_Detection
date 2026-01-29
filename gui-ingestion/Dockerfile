# 1. Usar una imagen ligera de Python
FROM python:3.12-slim

# 2. Configurar variables de entorno para que Python no genere archivos .pyc
ENV PYTHONDONTWRITEBYTECODE 1 #WE SAID TO PYTHON THAT DONT GENERATE __PYCHACHE__, THIS HELPUS TO IMPROVE THE PERFOMANCE
ENV PYTHONUNBUFFERED 1 #WE USE THIS TO SEND ALL THE LOG TO THE CONSOLE EVEN Y SOMTHING FAIL 

# 3. Establecer el directorio de trabajo
WORKDIR /app

# 4. Copiar los requisitos e instalarlos
# Nota: Copiamos requirements desde la carpeta del servicio
COPY gui-ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 5. Copiar el código del servicio y las utilidades compartidas
COPY gui-ingestion/ ./gui-ingestion/
COPY shared_utils/ ./shared_utils/

# 6. Exponer el puerto que usa Flask
EXPOSE 8080

# 7. Comando para ejecutar la aplicación con Gunicorn (estándar de producción)
# Instalamos gunicorn si no está en tus requirements
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "gui-ingestion.app:app"]