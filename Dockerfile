FROM python:3.10.6-buster
COPY ML_psy /ML_psy
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn ML_psy.api.fast:app --host 0.0.0.0