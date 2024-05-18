# Base image using Python 3.9
FROM python:3.9

# Set the working directory in the container to /code
WORKDIR /code

# Set environment variables for the cache directory
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV HF_HOME=/tmp/huggingface

# Create the directory and set permissions
RUN mkdir -p /tmp/huggingface/transformers
RUN chmod -R 777 /tmp/huggingface

# Copy the requirements file into the working directory
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files in the current directory on the host to the working directory in the container
COPY . /code

# Install unzip, unzip the archives, rearrange files, then clean up
RUN apt-get update \
    && apt-get install -y unzip \
    && unzip odoo_vector_store.zip -d /code \
    && unzip Data_Source_Pdf.zip -d /code/temp_data \
    && mkdir -p /code/data \
    && mv /code/temp_data/2021/* /code/data \
    && mv /code/temp_data/2022/* /code/data \
    && rm -rf /code/temp_data \
    && rm odoo_vector_store.zip \
    && rm Data_Source_Pdf.zip \
    # && rm -rf /code/odoo_vector_store2/index \
    && apt-get purge -y --auto-remove unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
