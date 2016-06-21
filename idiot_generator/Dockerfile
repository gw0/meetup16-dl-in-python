# Train a text generating LSTM on Slovenian poems and prose
#
# Usage:
#   docker run -d $(ls /dev/nvidia* | xargs -I{} echo '--device={}') -v /srv/storage/meetup16-idiot:/srv/out --name meetup16-idiot meetup16-idiot
#   docker logs -f meetup16-idiot

FROM gw000/keras:1.0.4-py2-tf-gpu

ADD slovenian-poetry/ ./slovenian-poetry/
ADD slovenian-prose/ ./slovenian-prose/
ADD idiot/ ./idiot/
ADD idiot-rnn.py ./idiot-rnn.py
RUN mkdir ./out \
 && chmod +x ./idiot-rnn.py

CMD ["/srv/idiot-rnn.py"]
