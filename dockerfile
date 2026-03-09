FROM ubuntu:latest

RUN apt-get update
RUN echo "Hello from Docker container"

CMD ["echo", "Container started successfully"]