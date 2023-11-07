FROM python:3.9

WORKDIR /digit_classification

COPY . /digit_classification/

RUN pip3 install -r /digit_classification/requirements.txt

VOLUME /digit_classification/models

# CMD ["pytest"]
ENV FLASK_APP=api/app.py
CMD ["flask", "run", "--host=0.0.0.0"]
# CMD ["python", "plot_digits_classification.py"]