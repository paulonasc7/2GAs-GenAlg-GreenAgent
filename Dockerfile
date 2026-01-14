FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential make \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN make -C examples cvrp

RUN pip install --no-cache-dir ./green-agent-template-main

EXPOSE 9009
CMD ["python", "green-agent-template-main/src/server.py", "--host", "0.0.0.0", "--port", "9009"]
