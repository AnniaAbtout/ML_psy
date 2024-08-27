FROM python:3.10.6-buster
# Set the working directory
WORKDIR /ML_psy
# Copy only the requirements file first to leverage Docker cache
COPY requirements_prod.txt requirements.txt
# Install the dependencies
RUN pip install -r requirements.txt
# Copy the rest of the application code
COPY . /ML_psy
# Command to run your application
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
