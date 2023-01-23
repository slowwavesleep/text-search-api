# starting from an existing image
# this service would be really impractical without gpu capabilities
FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04

WORKDIR /bgremover

COPY . .

# the rest is already installed in the base image
RUN pip install Flask
RUN pip install transformers

ENV FLASK_APP=api
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD ["flask", "run"]
