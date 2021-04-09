FROM python:3.7
WORKDIR /opt/app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install gunicorn

COPY . .

EXPOSE 5000

CMD ["python3", "api.py"]

ENTRYPOINT ["./gunicorn.sh"] 