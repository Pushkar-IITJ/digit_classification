FROM python:3.9

WORKDIR /digit_classification

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

# RUN pip3 install -r /digit_classification/requirements.txt

# VOLUME /digit_classification/models

ENV FLASK_APP=api/app.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]