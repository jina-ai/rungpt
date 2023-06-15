FROM jinaai/jina:3.17.0-py310-perf

COPY . /workspace/
WORKDIR /workspace

RUN python3 -m pip install -e .

RUN echo $'!Gateway\
py_modules:\
  open_gpt.serve.gateway\
with:\
  cors: False' > config.yml


ENTRYPOINT ["jina", "gateway", "--uses", "config.yml"]