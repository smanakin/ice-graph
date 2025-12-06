# 1. Use the official lightweight Miniconda image
FROM continuumio/miniconda3
# 2. Set the working directory
WORKDIR /app
# 3. Copy the environment file
COPY environment.yml .
# 4. Create the environment
# Notice we don't need Mamba anymore because the Conda list is tiny!
# We just run 'conda env create' and it will trigger the internal Pip install.
RUN conda env create -f environment.yml && \
    conda clean -afy
# 5. Add the new environment to the PATH
# This ensures that when you type 'python', it uses the 'icegraph' python
ENV PATH /opt/conda/envs/icegraph/bin:$PATH
# 6. Copy your code
COPY . .
# 7. Keep the container alive
CMD ["/bin/bash"]
