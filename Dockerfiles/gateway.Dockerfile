FROM jinaai/jina:3.17.0-py310-perf

COPY . /workspace/
WORKDIR /workspace

RUN python3 -m pip install -e .


RUN echo "\
!Gateway\n\
py_modules:\n\
  - open_gpt.serve.gateway\n\
with:\n\
  cors: False\n\
" > config.yml


ENTRYPOINT ["jina", "gateway", "--uses", "config.yml"]