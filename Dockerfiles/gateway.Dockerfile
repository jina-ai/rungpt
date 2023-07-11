FROM jinaai/jina:3.18.0-py310-standard

COPY . /open_gpt/
WORKDIR /open_gpt

RUN python3 -m pip install -e .


RUN echo "\
!Gateway\n\
py_modules:\n\
  - open_gpt.serve.gateway\n\
" > /tmp/config.yml


ENTRYPOINT ["jina", "gateway", "--uses", "/tmp/config.yml"]