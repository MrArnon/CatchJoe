FROM python

RUN useradd -m catchjoe
RUN mkdir /home/catchjoe/app
RUN chown -R catchjoe:catchjoe /home/catchjoe/

COPY --chown=catchjoe . /home/catchjoe/app/

WORKDIR /home/catchjoe/app

USER catchjoe

RUN pip install -r requirements.txt