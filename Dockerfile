FROM python:latest

RUN mkdir /Data61_WesternWater

WORKDIR /Data61_WesternWater

COPY ["requirements.txt", "/Data61_WesternWater"]

#VOLUME /Data61_WesternWater

# Copy the current directory contents into the container at /app
#ADD . /Data61_WesternWater

#FROM microsoft/dotnet-framework:4.6.2

#ADD https://download.microsoft.com/download/6/A/A/6AA4EDFF-645B-48C5-81CC-ED5963AEAD48/vc_redist.x64.exe /vc_redist.x64.exe
#RUN C:\vc_redist.x64.exe /quiet /install

# ADD some files to the images in this layer
RUN pip install -r requirements.txt

#ADD WaterMain_Main_Code.py /
#RUN /bin/sh /tmp/mysql-setup.sh

# Adding this will expose mysql on a random host port. It's recommended to avoid this. Other containers on the same 
# host can use the service without it.
#EXPOSE 3306
CMD ["python", "WaterMain_Main_Code.py"]