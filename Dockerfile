FROM ucsdets/datahub-base-notebook:2021.3-stable

USER root

RUN conda create -n sortinghat_docker python=3.7 poetry

USER jovyan
WORKDIR /home/jovyan

RUN git clone https://github.com/TanveerMittal/Sortinghat_Benchmark_Replication.git
RUN cd Sortinghat_Benchmark_Replication
RUN poetry install