FROM golang:1.18-bullseye

ENV APP_HOME /ai-image-recognition-telegrambot

# 1. Install the TensorFlow C Library (v2.9.0).
RUN curl -L https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-$(uname -m)-2.9.0.tar.gz \
    | tar xz --directory /usr/local \
    && ldconfig

# 2. Install the Protocol Buffers Library and Compiler.
RUN apt-get update && apt-get -y install --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    unzip

# 3. Download the Inception model.
RUN mkdir -p "$APP_HOME/model" && \
  curl -o "$APP_HOME/inception5h.zip" -s "http://download.tensorflow.org/models/inception5h.zip" && \
  unzip "$APP_HOME/inception5h.zip" -d "$APP_HOME/model"

# 4. Install and Setup the TensorFlow Go API.
RUN git clone --branch=v2.9.0 https://github.com/tensorflow/tensorflow.git /go/src/github.com/tensorflow/tensorflow \
    && cd /go/src/github.com/tensorflow/tensorflow \
    && go mod init github.com/tensorflow/tensorflow \
    && (cd tensorflow/go/op && go generate) \
    && go mod tidy \
    && go test ./...

# Build the  Program.
WORKDIR "$APP_HOME"
COPY *.go ./
COPY .env ./
RUN go mod init bot \
    && go mod edit -require github.com/tensorflow/tensorflow@v2.9.0+incompatible \
    && go mod edit -replace github.com/tensorflow/tensorflow=/go/src/github.com/tensorflow/tensorflow \
    && go mod tidy \
    && go build

ENTRYPOINT ["/ai-image-recognition-telegrambot/bot"]