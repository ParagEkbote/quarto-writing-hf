ARG QUARTO_VERSION="1.4.550"

# Use the Quarto base image
FROM ghcr.io/quarto-dev/quarto:${QUARTO_VERSION} AS builder

COPY src /app
WORKDIR /app

# Install Python requirements
USER root
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

RUN quarto render .

EXPOSE 7860
CMD ["python3", "-m", "http.server", "7860", "--directory", "_site"]
