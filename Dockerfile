FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6
    
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload", "--port", "8000"]
