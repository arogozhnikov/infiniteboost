FROM ubuntu:16.04

# installing apt things
RUN apt-get -y update; \
    apt-get install -y python-dev libblas-dev libatlas-dev liblapack-dev gfortran g++ python-pip git ;\
    apt-get install -y libpng-dev libjpeg8-dev libfreetype6-dev libxft-dev ;\
    apt-get clean; \
    apt-get autoclean; \
    apt-get autoremove    

# installing python things
RUN pip install --upgrade pip ;\
    pip --no-cache-dir install numpy==1.12.1 scipy==0.19.0 pandas==0.19.2 matplotlib==2.0.1 hep_ml==0.4.0 scikit-learn==0.18.1 jupyter joblib==0.11 nose tables

VOLUME ["/notebooks"]
EXPOSE 8890
CMD ["/bin/bash", "--login", "-c", "cd /notebooks && jupyter-notebook --allow-root --ip=* --port=8890 --NotebookApp.token="]

# running
# sudo docker run -it --rm -v /home/axelr/experiments:/notebooks -p 8890:8890 arogozhnikov/pmle:0.01

# uploading
# sudo docker build -t pmle .
# sudo docker tag pmle arogozhnikov/pmle:0.01
# sudo docker push arogozhnikov/pmle:0.01






