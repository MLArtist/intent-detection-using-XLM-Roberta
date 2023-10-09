# Use an official Python runtime as a parent image
FROM huggingface/transformers-pytorch-cpu:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app/.

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

#stop server from going into sleep
CMD ["sleep", "infinity"]